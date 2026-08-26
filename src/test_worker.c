/* test_worker.c - standalone LFM worker test */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "worker.h"
#include "lfm.h"

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [file_path]\n", prog);
    fprintf(stderr, "  file_path: document to load (default: test.txt)\n");
}

int main(int argc, char **argv) {
    const char *model_path = "/data/data/com.termux/files/home/projects/models/lfm2.5:2.6b:Q4_K_M.gguf";
    const char *file_path = "test.txt";
    if (argc > 1) file_path = argv[1];

    fprintf(stderr, "[test] Loading worker model from %s\n", model_path);
    if (worker_model_load(model_path) != 0) {
        fprintf(stderr, "[test] Failed to load worker model\n");
        return 1;
    }

    fprintf(stderr, "[test] Reading file: %s\n", file_path);
    FILE *f = fopen(file_path, "rb");
    if (!f) {
        perror("[test] fopen");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *doc = malloc((size_t)len + 1);
    if (!doc) {
        perror("[test] malloc");
        fclose(f);
        return 1;
    }
    size_t rd = fread(doc, 1, (size_t)len, f);
    doc[rd] = 0;
    fclose(f);
    fprintf(stderr, "[test] Read %zu bytes\n", rd);

    fprintf(stderr, "[test] Spawning worker...\n");
    int id = worker_spawn("test_worker", NULL, doc);
    free(doc);
    if (id < 0) {
        fprintf(stderr, "[test] worker_spawn failed (rc=%d)\n", id);
        return 1;
    }
    fprintf(stderr, "[test] Worker spawned (id=%d)\n", id);

    const char *question = "Summarize the document in one sentence.";
    fprintf(stderr, "[test] Asking: %s\n", question);
    char *answer = worker_ask(id, question, 1, 0); // ephemeral
    if (!answer) {
        fprintf(stderr, "[test] worker_ask returned NULL\n");
        worker_close(id);
        return 1;
    }
    printf("Answer: %s\n", answer);
    free(answer);

    worker_close(id);
    fprintf(stderr, "[test] Done.\n");
    return 0;
}
