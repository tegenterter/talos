# Talos - a UCI-compatible chess engine.
#
# Targets here mirror CLAUDE.md's Commands section; keep the two in sync.

GO     ?= go
EXE    := $(shell $(GO) env GOEXE)
BINARY := talos$(EXE)

# datagen defaults - override on the command line, e.g.:
#   make datagen DATAGEN_GAMES=20000 DATAGEN_THREADS=8 DATAGEN_OUT=run1.plain
DATAGEN_OUT     ?= datagen.plain
DATAGEN_GAMES   ?= 1000
DATAGEN_NODES   ?= 5000
DATAGEN_THREADS ?= 1
DATAGEN_SEED    ?= 1

.PHONY: all build run test test-short vet fmt fmt-check bench datagen pack ci clean

all: build

build:
	$(GO) build -o $(BINARY) .

# Runs the UCI loop, reading commands from stdin - "make run" then type
# "uci", "position startpos", "go depth 6", etc.
run:
	$(GO) run .

test:
	$(GO) test ./...

# The slow, timing-sensitive tests (parallel speedup, whole-endgame
# conversions) skip themselves under -short, for a fast inner dev loop.
test-short:
	$(GO) test -short ./...

vet:
	$(GO) vet ./...

fmt:
	gofmt -w .

# Fails (non-zero exit) and lists offending files if anything is unformatted,
# for CI or a pre-push check - "make fmt" above is the one that fixes it.
fmt-check:
	@files="$$(gofmt -l .)"; \
	if [ -n "$$files" ]; then \
		echo "gofmt needed on:"; echo "$$files"; exit 1; \
	fi

# Deterministic, single-threaded, fixed-position benchmark: the node total is
# a regression signal for search-behaviour changes, nps for raw speed. See
# CLAUDE.md's "Measuring strength" section before reading anything into it.
bench:
	$(GO) run . bench

# Self-play data generation for NNUE training (internal/datagen). Fixed nodes
# per move, not fixed time, so a run reproduces exactly from its seed.
datagen: build
	./$(BINARY) datagen \
		-out $(DATAGEN_OUT) \
		-games $(DATAGEN_GAMES) \
		-nodes $(DATAGEN_NODES) \
		-threads $(DATAGEN_THREADS) \
		-seed $(DATAGEN_SEED)

# Converts a datagen dump into the fixed-size binary records a trainer
# memory-maps. Separate from datagen so the text dump stays the archival
# artifact and packing can be redone whenever the feature set changes.
#   make pack PACK_IN=run1.plain PACK_OUT=run1.bin
pack: build
	@if [ -z "$(PACK_IN)" ]; then \
		echo "usage: make pack PACK_IN=<file.plain> [PACK_OUT=<file.bin>]" >&2; \
		exit 2; \
	fi
	./$(BINARY) pack -in $(PACK_IN) -out $(if $(PACK_OUT),$(PACK_OUT),$(PACK_IN:.plain=.bin))

# What CI (or a pre-push check) should run: formatting, vet, and the full
# test suite, in order of how cheap they are to fail on.
ci: fmt-check vet test

clean:
	rm -f $(BINARY)
