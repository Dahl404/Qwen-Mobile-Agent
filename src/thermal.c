/* thermal.c — device governor: faithful C port of the original jerry2
 * Python ThermalMonitor (jerry2-py/jerry_core/thermal.py), which was proven
 * accurate on this device. Same sensors, same thresholds, same hysteresis,
 * same spike guard, same worker caps.
 *
 *   temp = max(robust CPU-zone temp, battery temp)
 *   robust(): drop readings <20°C or >105°C (dead/bad sensors); if the
 *             hottest sane zone is >12°C above the second-hottest it's a
 *             glitch -> use the second-hottest; else the max.
 *
 *   levels: 0 cool · 1 warm · 2 hot · 3 critical
 *     UP      = [0, 70, 80, 88]   temp to ENTER a level
 *     DOWN    = [0, 66, 74, 82]   temp to LEAVE a level
 *     workers = {0: all, 1: 6, 2: 4, 3: 2}
 *     affinity: only level 3 pins to the efficiency cores (0-3)
 *
 *   - spike guard: TWO consecutive hot polls before throttling up, so a
 *     single glitching zone never pins the engine to little cores
 *   - cooldown: one level at a time via DOWN hysteresis
 *   - 8 s poll
 *
 * The worker cap goes through qma_pool_set_max(), which the eval pool reads
 * on EVERY pool_run — and prefill (prompt processing) and decode share that
 * pool, with prefill chunked into per-chunk pool_runs. So capping workers
 * caps prompt processing, which is the hottest phase.
 *
 * Level changes print to stderr (visible, off the output stream). Disable
 * with QMA_NOTHERMAL=1.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <dirent.h>
#include <time.h>
#include "thermal.h"
#include "qma.h"

#define POLL_INTERVAL_S  8.0

/* thresholds (°C) with hysteresis — EXACTLY the original jerry2 values */
static const double UP[4]   = { 0.0, 70.0, 80.0, 88.0 };
static const double DOWN[4] = { 0.0, 66.0, 74.0, 82.0 };
/* worker caps per level; 0 = all workers (restore) */
static const int WORKERS_BY_LEVEL[4] = { 0, 6, 4, 2 };
/* little-cluster CPUs (efficiency cores) on this 8-core device */
static const int EFF_CORES[4] = { 0, 1, 2, 3 };

static volatile int g_stop = 0;
static int g_level = 0;
static int g_hot_streak = 0;
static pthread_t g_thread;

/* ---- sensor reads (identical semantics to the Python originals) ---- */

/* all readable cpu-* / cpuss-* zone temps (°C); falls back to every readable
 * zone when no cpu zones exist. Returns the count read. */
static int read_thermal_zones(double *out, int cap) {
    const char *base = "/sys/class/thermal";
    DIR *d = opendir(base);
    if (!d) return 0;
    struct dirent *e;
    int n = 0, ncpu = 0;
    double tmp[64];
    int is_cpu[64];
    while ((e = readdir(d)) != NULL && n < 64) {
        if (strncmp(e->d_name, "thermal_zone", 12) != 0) continue;
        char p[256];
        snprintf(p, sizeof(p), "%s/%s/type", base, e->d_name);
        FILE *f = fopen(p, "r");
        if (!f) continue;
        char t[64] = "";
        if (fscanf(f, "%63s", t) != 1) { fclose(f); continue; }
        fclose(f);
        snprintf(p, sizeof(p), "%s/%s/temp", base, e->d_name);
        f = fopen(p, "r");
        if (!f) continue;
        long v = 0;
        if (fscanf(f, "%ld", &v) == 1) {
            tmp[n] = (double)v / 1000.0;
            is_cpu[n] = (strncmp(t, "cpu", 3) == 0);
            if (is_cpu[n]) ncpu++;
            n++;
        }
        fclose(f);
    }
    closedir(d);
    if (ncpu > 0) {
        int k = 0;
        for (int i = 0; i < n && k < cap; i++)
            if (is_cpu[i]) out[k++] = tmp[i];
        return k;
    }
    int k = 0;
    for (int i = 0; i < n && k < cap; i++) out[k++] = tmp[i];
    return k;
}

/* _robust_temp: filter 20..105, glitch-guard (>12°C gap -> second-hottest),
 * else max. 0.0 when nothing sane. */
static double robust_temp(const double *temps, int n) {
    double sane[64];
    int ns = 0;
    for (int i = 0; i < n; i++)
        if (temps[i] >= 20.0 && temps[i] <= 105.0) sane[ns++] = temps[i];
    if (ns == 0) return 0.0;
    if (ns >= 2) {
        /* insertion sort (tiny n) to find the two hottest */
        double a[64];
        memcpy(a, sane, sizeof(double) * ns);
        for (int i = 1; i < ns; i++) {
            double key = a[i];
            int j = i - 1;
            while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; }
            a[j + 1] = key;
        }
        if (a[ns - 1] - a[ns - 2] > 12.0) return a[ns - 2];
    }
    double mx = sane[0];
    for (int i = 1; i < ns; i++) if (sane[i] > mx) mx = sane[i];
    return mx;
}

/* battery °C via termux-battery-status (tenths handling), zone fallback */
static double read_battery(void) {
    FILE *f = popen("termux-battery-status 2>/dev/null", "r");
    if (f) {
        char buf[1024];
        size_t n = fread(buf, 1, sizeof(buf) - 1, f);
        pclose(f);
        buf[n] = 0;
        char *t = strstr(buf, "\"temperature\"");
        if (t) {
            t = strchr(t, ':');
            if (t) {
                double v = atof(t + 1);
                if (v > 100.0) v /= 10.0;   /* tenths of °C */
                return v;
            }
        }
    }
    /* fallback: thermal_zone type "battery" */
    const char *base = "/sys/class/thermal";
    DIR *d = opendir(base);
    if (!d) return 0.0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (strncmp(e->d_name, "thermal_zone", 12) != 0) continue;
        char p[256];
        snprintf(p, sizeof(p), "%s/%s/type", base, e->d_name);
        FILE *tf = fopen(p, "r");
        if (!tf) continue;
        char t[64] = "";
        if (fscanf(tf, "%63s", t) != 1) { fclose(tf); continue; }
        fclose(tf);
        if (strcmp(t, "battery") != 0) continue;
        snprintf(p, sizeof(p), "%s/%s/temp", base, e->d_name);
        FILE *vf = fopen(p, "r");
        if (!vf) continue;
        long v = 0;
        double r = 0.0;
        if (fscanf(vf, "%ld", &v) == 1) r = (double)v / 1000.0;
        fclose(vf);
        closedir(d);
        return r;
    }
    closedir(d);
    return 0.0;
}

/* ---- process-wide affinity (all threads), like os.sched_setaffinity ---- */
static void set_proc_affinity(int eff_only) {
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu < 1) ncpu = 8;
    DIR *d = opendir("/proc/self/task");
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        cpu_set_t set;
        CPU_ZERO(&set);
        if (eff_only) {
            for (int i = 0; i < 4; i++) CPU_SET(EFF_CORES[i], &set);
        } else {
            for (long i = 0; i < ncpu; i++) CPU_SET((int)i, &set);
        }
        sched_setaffinity((pid_t)atoi(e->d_name), sizeof(set), &set);
    }
    closedir(d);
}

static void apply_level(int level) {
    if (level == g_level) return;
    qma_pool_set_max(WORKERS_BY_LEVEL[level]);   /* 0 = all workers */
    set_proc_affinity(level >= 3);               /* eff cores only at 3 */
    g_level = level;
    fprintf(stderr, "[thermal] level %d%s\n", level,
            level == 3 ? " — efficiency cores, 2 workers" :
            level > 0 ? " — trimming workers" : " — full speed again");
}

static void *thermal_loop(void *arg) {
    (void)arg;
    while (!g_stop) {
        double zones[64];
        int nz = read_thermal_zones(zones, 64);
        double cpu = robust_temp(zones, nz);
        double bat = read_battery();
        double temp = (bat > cpu) ? bat : cpu;

        /* spike-guarded up-transition: TWO consecutive hot polls */
        int up_level = 0;
        if (temp >= UP[3]) up_level = 3;
        else if (temp >= UP[2]) up_level = 2;
        else if (temp >= UP[1]) up_level = 1;

        int new_level = g_level;
        if (up_level > g_level) {
            g_hot_streak++;
            if (g_hot_streak >= 2) {
                new_level = up_level;
                g_hot_streak = 0;
            }
        } else {
            g_hot_streak = 0;
        }

        /* cooldown via DOWN hysteresis, one level at a time */
        if (g_level > 0 && temp < DOWN[g_level])
            new_level = g_level - 1;

        apply_level(new_level);

        for (int i = 0; i < (int)(POLL_INTERVAL_S / 0.5) && !g_stop; i++)
            usleep(500000);
    }
    return NULL;
}

void thermal_start(void) {
    if (getenv("QMA_NOTHERMAL")) return;   /* also checked by the caller */
    g_stop = 0;
    g_level = 0;
    g_hot_streak = 0;
    if (pthread_create(&g_thread, NULL, thermal_loop, NULL) != 0)
        return;
    pthread_detach(g_thread);
}

void thermal_stop(void) {
    g_stop = 1;
    qma_pool_set_max(0);        /* restore all workers */
    set_proc_affinity(0);       /* restore full affinity */
}
