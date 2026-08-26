/* platform_id.c — portable platform fingerprint (os-arch-cpu).
 *
 * The self-hosting binary is compiled with -mcpu/-march=native, so it is
 * tuned for ONE machine. To support "copy the binary to any device and it
 * rebuilds for that device", the build machine's identity is stamped into
 * the internal tree (.platform-id) and compared at boot against the
 * current machine's identity (qma_platform_id). A mismatch means the
 * binary was built for a different platform, and the engine recompiles
 * itself from the embedded source, tuned for the machine it's now on.
 *
 * Used two ways:
 *   - compiled into the engine as qma_platform_id(), and
 *   - built standalone (-DPLATFORM_ID_MAIN) by the Makefile to stamp
 *     .platform-id before the tree is embedded.
 */
#include <stdio.h>
#include <string.h>
#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>   /* sysctlbyname for the CPU name */
#endif

static const char *os_name(void) {
#if defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "other";
#endif
}

static const char *arch_name(void) {
#if defined(__aarch64__)
    return "aarch64";
#elif defined(__x86_64__)
    return "x86_64";
#else
    return "other";
#endif
}

/* Stable identifier for the CPU we are running on: ARM implementer+part or
   x86 model name on Linux, the chip name on macOS. Two machines with the
   same string are safe to share a -mcpu/-march=native binary. */
static const char *cpu_name(char *buf, size_t cap) {
#if defined(__linux__)
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            const char *v;
            if (strncmp(line, "CPU part", 8) == 0 ||
                strncmp(line, "model name", 10) == 0) {
                v = strchr(line, ':');
                if (!v) continue;
                v++;
                while (*v == ' ' || *v == '\t') v++;
                size_t l = strlen(v);
                while (l > 0 && (v[l-1] == '\n' || v[l-1] == ' ' || v[l-1] == '\t')) l--;
                if (l >= cap) l = cap - 1;
                memcpy(buf, v, l);
                buf[l] = 0;
                fclose(f);
                return buf;
            }
        }
        fclose(f);
    }
    return "unknown";
#elif defined(__APPLE__)
    {
        size_t len = cap;
        if (sysctlbyname("machdep.cpu.brand_string", buf, &len, NULL, 0) == 0 &&
            len > 0 && buf[0])
            return buf;
        len = cap;
        if (sysctlbyname("hw.model", buf, &len, NULL, 0) == 0 && len > 0 && buf[0])
            return buf;
    }
    return "unknown";
#else
    (void)buf; (void)cap;
    return "unknown";
#endif
}

/* full platform id: "os-arch-cpu" (compared at boot for the self-rebuild) */
void qma_platform_id(char *buf, size_t cap) {
    char c[512];
    snprintf(buf, cap, "%s-%s-%s", os_name(), arch_name(), cpu_name(c, sizeof(c)));
}

#ifdef PLATFORM_ID_MAIN
/* build helper: prints the build machine's platform id on stdout. */
int main(void) {
    char b[512];
    qma_platform_id(b, sizeof(b));
    printf("%s\n", b);
    return 0;
}
#endif
