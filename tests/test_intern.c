/* tests/test_intern.c — internal-tree blob round trip:
   embed a small tree, extract it, verify every file byte-for-byte,
   and probe the generation serial. Links intern.c + selfctx.c only. */
#define main test_intern_main_unused
#include "../src/intern.c"
#undef main

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

static int g_fail = 0;
static void check(const char *what, int ok) {
    if (!ok) { g_fail++; fprintf(stderr, "FAIL %s\n", what); }
    else      fprintf(stderr, "ok   %s\n", what);
}

int main(void) {
    char root[256];
    snprintf(root, sizeof(root), "/data/data/com.termux/files/home/.qma_it_%d", (int)getpid());
    char tree[300], blob[300], out[300];
    snprintf(tree, sizeof(tree), "%s/tree", root);
    snprintf(blob, sizeof(blob), "%s/blob.bin", root);
    snprintf(out,  sizeof(out),  "%s/out", root);
    mkdir(root, 0700);
    mkdir(tree, 0700);

    char sub[320], f1[320], f2[320];
    snprintf(sub, sizeof(sub), "%s/sub", tree);
    snprintf(f1,  sizeof(f1),  "%s/a.c", tree);
    snprintf(f2,  sizeof(f2),  "%s/sub/b.txt", tree);
    mkdir(sub, 0700);
    FILE *f = fopen(f1, "w");
    fprintf(f, "int main(void){return 0;}\n/* padding padding padding padding */\n");
    fclose(f);
    f = fopen(f2, "w");
    fprintf(f, "hello internal tree\nline two\n");
    fclose(f);

    int fd = open(blob, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    check("open blob", fd >= 0);
    int rc = intern_embed_dir(fd, tree, 0, 7, 1234567890);
    close(fd);
    check("embed", rc == 0);

    mkdir(out, 0700);
    rc = intern_extract_file(blob, out);
    check("extract", rc == 0);

    char o1[320], o2[320];
    snprintf(o1, sizeof(o1), "%s/a.c", out);
    snprintf(o2, sizeof(o2), "%s/sub/b.txt", out);
    check("a.c present", access(o1, R_OK) == 0);
    check("sub/b.txt present", access(o2, R_OK) == 0);

    {
        char b1[4096], b2[4096];
        FILE *g = fopen(f1, "rb"); size_t n1 = fread(b1, 1, sizeof(b1), g); fclose(g);
        g = fopen(o1, "rb");      size_t n2 = fread(b2, 1, sizeof(b2), g); fclose(g);
        check("a.c bytes", n1 == n2 && memcmp(b1, b2, n1) == 0);
        g = fopen(f2, "rb"); n1 = fread(b1, 1, sizeof(b1), g); fclose(g);
        g = fopen(o2, "rb"); n2 = fread(b2, 1, sizeof(b2), g); fclose(g);
        check("b.txt bytes", n1 == n2 && memcmp(b1, b2, n1) == 0);
    }

    uint64_t gen = 0, bt = 0;
    check("probe gen+bt", intern_probe(blob, &gen, &bt) == 1 && gen == 7 && bt == 1234567890);

    char cmd[512];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", root);
    system(cmd);
    fprintf(stderr, g_fail ? "FAILED: %d\n" : "PASS: intern round-trip\n", g_fail);
    return g_fail ? 1 : 0;
}
