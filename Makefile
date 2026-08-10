# qma — qwen mobile agent, unified C agent: engine + agent loop + tools in ONE binary.
# No Python, no HTTP server, no separate processes.

CC      ?= clang
CFLAGS  ?= -O3 -std=c99 -Wall -Wextra -Wno-unused-parameter
CFLAGS  += -mcpu=native -ffast-math -fomit-frame-pointer -pthread
LDFLAGS ?= -lm -pthread

ENGINE = src/gguf.c src/tokenizer.c src/nn.c src/quants.c src/sampler.c \
         src/ecache.c src/q8k.c src/kvq.c src/intern.c
AGENT  = src/json.c src/toolparse.c src/grammar.c src/tools.c src/thermal.c src/selfctx.c src/agent.c
HDR    = src/qma.h src/ecache.h src/kvq.h src/json.h src/toolparse.h src/grammar.h src/tools.h src/thermal.h src/selfctx.h src/intern.h

all: qma

qma: $(ENGINE) $(AGENT) $(HDR)
	$(CC) $(CFLAGS) -o $@ $(ENGINE) $(AGENT) $(LDFLAGS)
	./$@ --embed-internal . $@

# tests (no engine core needed)
tests/test_grammar: tests/test_grammar.c src/grammar.c src/grammar.h src/qma.h
	$(CC) $(CFLAGS) -I src -o $@ tests/test_grammar.c src/grammar.c $(LDFLAGS)

tests/test_intern: tests/test_intern.c src/intern.c src/intern.h src/selfctx.c src/selfctx.h
	$(CC) $(CFLAGS) -I src -o $@ tests/test_intern.c src/selfctx.c $(LDFLAGS)

test: tests/test_grammar tests/test_intern
	./tests/test_grammar
	./tests/test_intern

clean:
	rm -f qma tests/test_grammar tests/test_intern

.PHONY: all test clean
