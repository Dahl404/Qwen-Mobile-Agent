# qma — qwen mobile agent, unified C agent: engine + agent loop + tools in ONE binary.
# No Python, no HTTP server, no separate processes.

CC      ?= clang
CFLAGS  ?= -O3 -std=c99 -Wall -Wextra -Wno-unused-parameter
CFLAGS  += -mcpu=native -ffast-math -fomit-frame-pointer -pthread
LDFLAGS ?= -lm -pthread

ENGINE = src/gguf.c src/tokenizer.c src/nn.c src/quants.c src/sampler.c \
         src/ecache.c src/q8k.c src/kvq.c
AGENT  = src/json.c src/toolparse.c src/grammar.c src/tools.c src/thermal.c src/selfctx.c src/agent.c
HDR    = src/qma.h src/ecache.h src/kvq.h src/json.h src/toolparse.h src/grammar.h src/tools.h src/thermal.h src/selfctx.h

all: qma

qma: $(ENGINE) $(AGENT) $(HDR)
	$(CC) $(CFLAGS) -o $@ $(ENGINE) $(AGENT) $(LDFLAGS)

# parser unit tests (no engine core needed — types only)
work/test_toolparse: work/test_toolparse.c src/json.c src/toolparse.c src/json.h src/toolparse.h src/qma.h
	$(CC) $(CFLAGS) -I src -o $@ work/test_toolparse.c src/json.c src/toolparse.c $(LDFLAGS)

test: work/test_toolparse work/test_selfctx work/test_grammar
	./work/test_toolparse
	./work/test_selfctx
	./work/test_grammar

clean:
	rm -f qma work/test_toolparse

.PHONY: all test clean

# selfctx snapshot mechanics tests (no engine core needed)
work/test_selfctx: work/test_selfctx.c src/selfctx.c src/selfctx.h
	$(CC) $(CFLAGS) -I src -o $@ work/test_selfctx.c src/selfctx.c $(LDFLAGS)

# tool-call grammar DFA tests
work/test_grammar: work/test_grammar.c src/grammar.c src/grammar.h src/qma.h
	$(CC) $(CFLAGS) -I src -o $@ work/test_grammar.c src/grammar.c $(LDFLAGS)
