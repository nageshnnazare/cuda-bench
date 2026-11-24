"""Turn `ncu --csv --page raw` output into a readable markdown table.

    python3 scripts/summarize_profile.py reports/run/bench_matmul.csv
"""
import csv
import sys

COLUMNS = [
    ("gpu__time_duration.sum", "time (us)", lambda v: v / 1e3),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM %", None),
    ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "Mem %", None),
    ("dram__bytes_read.sum", "DRAM rd (MB)", lambda v: v / 1e6),
    ("dram__bytes_write.sum", "DRAM wr (MB)", lambda v: v / 1e6),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "occupancy %", None),
    ("launch__registers_per_thread", "regs", None),
    ("sectors/req", "sectors/req", None),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "smem conflicts", None),
    ("sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active", "tensor %", None),
]


def num(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def main(path):
    with open(path, newline="") as f:
        # ncu prints its own log lines (==PROF==) before the CSV header.
        lines = [l for l in f if l.startswith('"')]
    rows = list(csv.DictReader(lines))
    rows = [r for r in rows if r.get("ID", "").strip() not in ("", None)]  # drop the units row

    print("| kernel | " + " | ".join(c[1] for c in COLUMNS) + " |")
    print("|---" * (len(COLUMNS) + 1) + "|")
    for r in rows:
        name = r.get("Kernel Name", "?")
        name = name.split("(")[0]
        if len(name) > 48:
            name = name[:45] + "..."
        sectors = num(r.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"))
        reqs = num(r.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"))
        r["sectors/req"] = f"{sectors / reqs:.2f}" if sectors and reqs else ""
        cells = []
        for key, _, fn in COLUMNS:
            v = num(r.get(key, ""))
            if v is None:
                cells.append(r.get(key, "") or "")
            else:
                v = fn(v) if fn else v
                cells.append(f"{v:,.1f}" if abs(v) >= 1 or v == 0 else f"{v:.3f}")
        print(f"| `{name}` | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1])
