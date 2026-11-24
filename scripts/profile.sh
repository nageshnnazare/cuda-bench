#!/usr/bin/env bash
# Profile the programs with Nsight Compute (ncu).

# Each invocation writes one run folder, reports/<UTC time>_<GPU>_clock-<mode>/, so earlier runs are kept.
# The folder is meant to be committed with the code. For each program it contains:
#   <name>.ncu-rep    full report (--set full, with source): open it in the Nsight Compute GUI
#   <name>.log        everything ncu printed for that program, both passes, as it ran
#   <name>.csv        the key metrics, raw from ncu, one row per kernel launch
#   <name>.md         the same metrics as a markdown table (../scripts/ncu_summary.py)
#   <name>.ptxas.log  the compiler's register and spill report for the build of this program
# Once per run it contains:
#   environment.txt   the date, the GPU, the driver, the CUDA and ncu versions, and the git commit with any local changes
#
# Usage:
#   ./profile.sh                          every program in bin/
#   ./profile.sh matmul_01_naive          only the named programs (names as in bin/)
#   NCU_CLOCK=none ./profile.sh           profile at boost clocks (default: base clocks)
#   RUN_DIR=reports/my-run ./profile.sh   write into a folder you name (an existing folder is overwritten)
#
# Each program runs with --profile, so ncu sees exactly one launch of the kernel: no self-test and no timing loop.
# ncu usually needs root or the perf-counter permission: https://developer.nvidia.com/ERR_NVGPUCTRPERM
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -gt 0 ]; then
  for name in "$@"; do
    make -s "bin/$name"
  done
else
  make -s all        # build first, so bin/ is current (a no-op when nothing changed)
fi
mkdir -p reports

# ncu locks the GPU to its base clock by default, so the numbers are reproducible. On a T4 that is about 585 MHz
# against about 1.5 GHz at boost, so a kernel can look slower here than in the timing output. Use
# NCU_CLOCK=none to match the timings. Counters are the same either way.
CLOCK="${NCU_CLOCK:-base}"

# One folder per run, named by the time, the GPU and the clock mode, so a second run never overwrites the first.
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | tr -s ' ' '-' | tr -cd 'A-Za-z0-9._-' || true)"
RUN_DIR="${RUN_DIR:-reports/${STAMP}_${GPU:-no-gpu}_clock-${CLOCK}}"
mkdir -p "$RUN_DIR"

# The metrics in the compact table. Each one is explained in PROFILING.md, section 4.
METRICS=$(paste -sd, - <<'EOF'
gpu__time_duration.sum
sm__throughput.avg.pct_of_peak_sustained_elapsed
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
dram__bytes_read.sum
dram__bytes_write.sum
sm__warps_active.avg.pct_of_peak_sustained_active
launch__registers_per_thread
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
EOF
)

# Record the setup, so a log can be read later without knowing the machine it came from.
{
  echo "date_utc:      $STAMP"
  echo "host:          $(hostname)"
  echo "clock_control: $CLOCK"
  echo "command:       $0 $*"
  echo "run_dir:       $RUN_DIR"
  echo
  echo "== git (repo root is the parent of learn/)"
  if git -C .. rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "commit: $(git -C .. rev-parse HEAD)"
    echo "local changes (git status --short, empty means clean):"
    git -C .. status --short || true
  else
    echo "not a git checkout"
  fi
  echo
  echo "== nvidia-smi"
  nvidia-smi 2>&1 || echo "nvidia-smi failed"
  echo
  echo "== nvcc"
  nvcc --version 2>&1 || echo "nvcc not found"
  echo
  echo "== ncu"
  ncu --version 2>&1 || echo "ncu not found"
} > "$RUN_DIR/environment.txt"

# Choose the programs: the names given on the command line, or every executable in bin/.
BINARIES=()
if [ "$#" -gt 0 ]; then
  for name in "$@"; do
    BINARIES+=("bin/$name")
  done
else
  for f in bin/*; do
    case "$f" in
      *.ptxas.log) continue ;;   # the compiler logs are not programs
    esac
    if [ -x "$f" ] && [ -f "$f" ]; then
      BINARIES+=("$f")
    fi
  done
fi

if [ "${#BINARIES[@]}" -eq 0 ]; then
  echo "no programs found in bin/; run make first" >&2
  exit 1
fi

for bin in "${BINARIES[@]}"; do
  name="$(basename "$bin")"
  log="$RUN_DIR/$name.log"
  echo "== $name"
  # Everything ncu prints for this program goes to its log, and to the terminal as well.
  {
    echo "### full report: ncu --set full --import-source yes"
    # 1. Full report, for the GUI: every section, with source lines. Slow, because ncu replays each kernel many times.
    ncu --clock-control "$CLOCK" --set full --import-source yes -f -o "$RUN_DIR/$name" "$bin" --profile
    echo "### metric table: ncu --metrics (the raw CSV is written to $name.csv)"
    # 2. Compact table of the key metrics, as raw CSV on stdout. ncu's own messages go to the log.
    ncu --clock-control "$CLOCK" --metrics "$METRICS" --csv --page raw --print-units base "$bin" --profile \
      > "$RUN_DIR/$name.csv"
  } 2>&1 | tee "$log"
  # 3. The compiler's register and spill report for this program, kept next to its profile.
  if [ -f "bin/$name.ptxas.log" ]; then
    cp "bin/$name.ptxas.log" "$RUN_DIR/$name.ptxas.log"
  else
    echo "no bin/$name.ptxas.log to copy (run make to build it)" | tee -a "$log"
  fi
  # 4. The same table as markdown, with sectors per request computed for you.
  python3 ../scripts/summarize_profile.py "$RUN_DIR/$name.csv" > "$RUN_DIR/$name.md"
  cat "$RUN_DIR/$name.md"
done

echo "run folder: $RUN_DIR"
