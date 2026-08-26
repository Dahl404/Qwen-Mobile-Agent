# qma — qwen mobile agent, unified C agent: engine + agent loop + tools in ONE binary.
# No Python, no HTTP server, no separate processes.

CC      ?= clang
CFLAGS  ?= -O3 -std=c99 -Wall -Wextra -Wno-unused-parameter
CFLAGS  += -mcpu=native -ffast-math -fomit-frame-pointer -pthread -I src/lfm -I src
LDFLAGS ?= -lm -pthread -ldl

ENGINE = src/gguf.c src/tokenizer.c src/nn.c src/quants.c src/sampler.c \
         src/ecache.c src/q8k.c src/qkerns.c src/kvq.c src/intern.c \
         src/lfm/lfm_gguf.c src/lfm/lfm_tokenizer.c src/lfm/lfm_nn.c \
         src/lfm/lfm_sampler.c src/worker.c
AGENT  = src/json.c src/toolparse.c src/tools.c src/thermal.c src/selfctx.c src/agent.c src/cl.c
HDR    = src/qma.h src/ecache.h src/kvq.h src/json.h src/toolparse.h src/tools.h src/thermal.h src/selfctx.h src/intern.h src/cl.h src/worker.h src/lfm/lfm.h

all: qma

qma: $(ENGINE) $(AGENT) $(HDR)
	$(CC) $(CFLAGS) -o $@ $(ENGINE) $(AGENT) $(LDFLAGS)
	./$@ --embed-internal . $@

# tests (no engine core needed)
tests/test_toolparse: tests/test_toolparse.c src/toolparse.c src/toolparse.h src/json.c src/qma.h
	$(CC) $(CFLAGS) -I src -o $@ tests/test_toolparse.c src/toolparse.c src/json.c $(LDFLAGS)

tests/test_intern: tests/test_intern.c src/intern.c src/intern.h src/selfctx.c src/selfctx.h
	$(CC) $(CFLAGS) -I src -o $@ tests/test_intern.c src/selfctx.c $(LDFLAGS)

tests/cltest: tests/cltest.c src/cl.c src/cl.h src/quants.c src/qma.h
	$(CC) $(CFLAGS) -I src -o $@ tests/cltest.c src/cl.c src/quants.c $(LDFLAGS)

tests/test_q8kgemm: tests/test_q8kgemm.c src/q8k.c src/qkerns.c src/quants.c src/qma.h
	$(CC) $(CFLAGS) -I src -o $@ tests/test_q8kgemm.c src/q8k.c src/qkerns.c src/quants.c $(LDFLAGS)

test: tests/test_toolparse tests/test_intern tests/cltest tests/test_q8kgemm
	./tests/test_toolparse
	./tests/test_intern
	./tests/cltest
	./tests/test_q8kgemm

clean:
	rm -f qma tests/test_toolparse tests/test_intern tests/cltest

.PHONY: all test clean
