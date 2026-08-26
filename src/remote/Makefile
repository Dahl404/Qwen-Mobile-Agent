# remote — qma's headless autonomous remote agent.
# A standalone binary built from the LFM engine + the self-hosting shell.
# No UI, no chat: it reads a task spec, executes autonomously, packs its
# data + memory + KV into a return binary, and (per the return directive)
# writes it back for qma to re-ingest.
#
# Build:   make            -> remote (model-less base, ~1.5 MB)
# Breed:   qma dispatches  -> remote + LFM weights + task + internal tree
CC      ?= clang
CFLAGS  ?= -O3 -std=c99 -Wall -Wextra -Wno-unused-parameter
CFLAGS  += -mcpu=native -ffast-math -fomit-frame-pointer -pthread
LDFLAGS ?= -lm -pthread -ldl

LFM    = ../lfm/lfm_gguf.c ../lfm/lfm_tokenizer.c ../lfm/lfm_nn.c ../lfm/lfm_sampler.c
REMOTE = remote_main.c remote_tools.c json.c selfctx.c intern.c platform_id.c

all: remote

remote: $(LFM) $(REMOTE) remote.h selfctx.h intern.h ../lfm/lfm.h
	$(CC) $(CFLAGS) -I ../lfm -I .. -I . -o $@ $(LFM) $(REMOTE) $(LDFLAGS)

clean:
	rm -f remote

.PHONY: all clean
