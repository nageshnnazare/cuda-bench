# Makefile: builds every standalone learning program.
#
#   make                   build every program into bin/
#   make matmul            build one dir in src
#   make run               build, then run every program (self-test, then timing) and append to reports/timings.csv
#   make ARCH=sm_75        choose the GPU architecture (default: detected with nvidia-smi, else sm_75)
#   make clean             delete bin/ (keeps reports/)

NVCC     ?= nvcc
DETECTED := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
# sm_75 is the Tesla T4. WMMA needs sm_70 or newer.
ARCH     ?= $(if $(DETECTED),sm_$(DETECTED),sm_75)

BIN     := bin
RESULTS := reports/timings.csv

# Every program is one .cu file. The helpers in common/ are included by relative path, so no -I flags are needed.
COMMON := include/common/helpers.cuh
SRCS   := $(wildcard src/matmul/*.cu)
BINS   := $(patsubst src/matmul/%.cu,$(BIN)/matmul_%,$(SRCS))

# -lineinfo lets Nsight Compute map SASS back to source lines (needed for --import-source).
# -Xptxas=-v prints the registers used by each kernel. -warn-spills warns when registers spill to local memory.
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -Iinclude/ -lineinfo -Xptxas=-v,-warn-spills
LIBS      :=

# The compile step, shared by all the rules below. $< is the .cu file and $@ is the binary.
# On failure it prints the whole ptxas log. On success it prints only warnings, register counts and spills.
build = $(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS) 2> $@.ptxas.log || { cat $@.ptxas.log; exit 1; }; grep -E "warning|spill|registers" $@.ptxas.log | grep -v " 0 bytes spill" || true

.PHONY: all run clean matmul

all: $(BINS)

$(BIN): ; mkdir -p $@

$(BIN)/matmul_%: src/matmul/%.cu $(COMMON) | $(BIN)
	$(build)

# One target per topic, for building only that part.
matmul:    $(filter $(BIN)/matmul_%,$(BINS))

# Run every program with its default mode: the self-test against the CPU, then timing. Each timing line is
# appended to $(RESULTS). Stops at the first program that fails.
run: all
	mkdir -p reports
	rm -f $(RESULTS)
	@for b in $(BINS); do echo "== $$b"; ./$$b --csv $(RESULTS) || exit 1; done

clean:
	rm -rf $(BIN)

