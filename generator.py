#!/usr/bin/env python3
"""
TBPP-C-LF instance generator.

Generates ONE instance with n items and conflict density d, in the same JSON
format as the existing instances (minus 'pi', plus per-item 'h').

Model conventions encoded here
------------------------------
* Item i has size w_i, half-open activity interval [s_i, e_i), split budget k_i
  (max number of SPLITS -> at most k_i + 1 fragments), per-fragment overhead
  h_i = eta * C, and minimum fragment size delta_i = w_i / a.
* Conflicts exist ONLY between items whose intervals overlap. Items with
  disjoint time spans are never in conflict (they may share a bin).
* d is the EFFECTIVE conflict density: each temporally overlapping pair becomes
  a conflict with probability d. Both effective and raw density are recorded.
* No 'pi' field: fragmentation is priced by the capacity overhead h, not by an
  objective penalty.
* No duplicate items: no two items share the same full definition
  (s, e, w, k). Intervals may repeat - several items can run over the same
  time span, they just cannot be identical items.

Usage (Anaconda Prompt)
-----------------------
    cd C:\\path\\to\\your\\folder
    python gen_instance.py --n 200 --d 0.5

    # how many instances: omit --count for 1, or ask for k of them
    python gen_instance.py --n 200 --d 0.5 --count 20

    # a whole conflict-density study in one command
    python gen_instance.py --n 200 --d 0.1 0.3 0.5 0.7 0.9 --count 10

    python gen_instance.py --n 50 --d 0.2 --C 120 --eta 0.05 --a 8 --seed 7
    python gen_instance.py --n 1000 --d 0.8 --w-min 30 --w-max 60
    --T 127 --dur-min 40 --dur-max 80 // to have higher bins end of it

Output layout: instances are grouped by conflict density.

    dinst/                      <- --out-dir (default "dinst")
      0.1/                      <- one folder per density, created on demand
        n200_d0.1_C100.json
      0.5/
        n200_d0.5_C100.json

File name: n{n}_d{d}_C{C}.json, with _s{seed} appended when --count > 1.
An existing file is never overwritten silently (_v2, _v3, ... unless --overwrite).
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

FORMAT_VERSION = "4.0"


# ----------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------
def make_items(n, C, T, w_min, w_max, dur_min, dur_max, k_min, k_max, eta, a, rng):
    """Draw n items. Items are unique as ITEMS, i.e. no two items share the
    same full definition (s, e, w, k). Intervals themselves may repeat: several
    different items are allowed to run over the very same time span."""
    h = eta * C

    # is the sample space big enough for n distinct items?
    n_w = w_max - w_min + 1
    n_k = k_max - k_min + 1
    n_intervals = sum(max(0, T - dur + 1) for dur in range(dur_min, dur_max + 1))
    space = n_intervals * n_w * n_k
    if space < n:
        raise SystemExit(
            f"cannot draw {n} distinct items: only {space} unique "
            f"(interval, size, k) combinations exist for T={T}, duration "
            f"{dur_min}-{dur_max}, size {w_min}-{w_max}, k {k_min}-{k_max}. "
            f"Increase --T, --horizon-ratio, or the duration/size/k ranges."
        )

    seen = set()
    items = []
    attempts = 0
    max_attempts = 200 * n + 10000
    while len(items) < n:
        attempts += 1
        if attempts > max_attempts:
            raise SystemExit(
                f"gave up after {max_attempts} attempts with only {len(items)}/{n} "
                f"distinct items (sample space too tight). Widen --T or the "
                f"duration/size ranges."
            )
        dur = rng.randint(dur_min, dur_max)
        s = rng.randint(0, T - dur)
        w = rng.randint(w_min, w_max)
        k = rng.randint(k_min, k_max)
        key = (s, s + dur, w, k)          # the item, not just its interval
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "id": len(items),
            "s": s,
            "e": s + dur,
            "w": w,
            "k": k,
            "h": round(h, 6),
            "delta": round(w / a, 6),
        })

    items.sort(key=lambda it: (it["s"], it["e"], it["w"]))   # readable order
    for new_id, it in enumerate(items):
        it["id"] = new_id
    return items


def overlapping_pairs(items):
    """Pairs (i, j), i < j, whose half-open intervals [s, e) overlap."""
    n = len(items)
    out = []
    for i in range(n):
        si, ei = items[i]["s"], items[i]["e"]
        for j in range(i + 1, n):
            if si < items[j]["e"] and items[j]["s"] < ei:
                out.append((i, j))
    return out


def make_conflicts(pairs, d, rng):
    """Conflicts ONLY over temporally overlapping pairs, each kept w.p. d."""
    if d <= 0:
        return []
    return [[i, j] for (i, j) in pairs if rng.random() < d]


# ----------------------------------------------------------------------
# statistics / feasibility
# ----------------------------------------------------------------------
def maximal_cliques(items):
    """Maximal cliques of the interval graph = maximal sets of simultaneously
    active items, which occur at item start times."""
    starts = sorted({it["s"] for it in items})
    seen, cands = set(), []
    for t in starts:
        act = frozenset(it["id"] for it in items if it["s"] <= t < it["e"])
        if act and act not in seen:
            seen.add(act)
            cands.append(act)
    cands.sort(key=len, reverse=True)
    out = []
    for c in cands:
        if not any(c < m for m in out):
            out.append(c)
    return out


def check_feasibility(items, C):
    """Every item must be placeable under its own overhead / delta / k budget."""
    errs = []
    for it in items:
        h, w, k, delta = it["h"], it["w"], it["k"], it["delta"]
        cap = C - h                                   # usable room per fragment
        if cap <= 0:
            errs.append(f"item {it['id']}: h={h} >= C={C}")
            continue
        need = math.ceil(w / cap)                     # fragments required
        allowed = k + 1                               # fragments permitted
        if delta > 0:
            allowed = min(allowed, math.floor(w / delta + 1e-9))
        if need > allowed:
            errs.append(
                f"item {it['id']}: needs >= {need} fragments (w={w}, C-h={cap:g}) "
                f"but k/delta allow at most {allowed}"
            )
        if delta > cap:
            errs.append(f"item {it['id']}: delta={delta:g} > C-h={cap:g}")
    return errs


def build_stats(items, conflicts, pairs, C):
    n = len(items)
    cliques = maximal_cliques(items)
    n_pairs = n * (n - 1) // 2

    # peak-load lower bound on bins (valid relaxation: ignores h and conflicts)
    lb_peak = max(math.ceil(sum(items[i]["w"] for i in K) / C) for K in cliques)

    # overhead-aware bound: each item in a clique costs at least w_i + h_i
    # (>= one fragment present for its whole interval)
    lb_h = max(
        math.ceil(sum(items[i]["w"] + items[i]["h"]
                      * math.ceil(items[i]["w"] / (C - items[i]["h"]))
                      for i in K) / C)
        for K in cliques
    )
    return {
        "n_overlap_pairs": len(pairs),
        "n_conflicts": len(conflicts),
        "eff_conflict_density": round(len(conflicts) / len(pairs), 4) if pairs else 0.0,
        "raw_conflict_density": round(len(conflicts) / n_pairs, 4) if n_pairs else 0.0,
        "n_maximal_cliques": len(cliques),
        "max_clique_size": max(len(K) for K in cliques),
        "avg_clique_size": round(sum(len(K) for K in cliques) / len(cliques), 2),
        "lb_peak_bins": lb_peak,
        "lb_peak_bins_overhead": lb_h,
        "horizon": [min(it["s"] for it in items), max(it["e"] for it in items)],
    }


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Generate one TBPP-C-LF instance (JSON).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # required
    p.add_argument("--n", type=int, required=True, help="number of items")
    p.add_argument("--d", type=float, required=True, nargs="+",
                   help="conflict density over TEMPORALLY OVERLAPPING pairs (0..1). "
                        "Several values allowed: --d 0.1 0.3 0.5")
    # problem parameters
    p.add_argument("--C", type=float, default=100, help="bin capacity")
    p.add_argument("--eta", type=float, default=0.1,
                   help="per-fragment overhead fraction: h = eta * C")
    p.add_argument("--a", type=float, default=10,
                   help="minimum fragment size divisor: delta_i = w_i / a")
    p.add_argument("--k-min", type=int, default=0, help="minimum split budget k_i")
    p.add_argument("--k-max", type=int, default=3, help="maximum split budget k_i")
    # size distribution (absolute; defaults derived from C)
    p.add_argument("--w-min", type=int, default=None,
                   help="minimum item size (default: 0.25*C)")
    p.add_argument("--w-max", type=int, default=None,
                   help="maximum item size (default: 0.55*C)")
    # time distribution (absolute; defaults derived from T)
    p.add_argument("--T", type=int, default=None,
                   help="time horizon (default: round(horizon-ratio * n))")
    p.add_argument("--horizon-ratio", type=float, default=1.27,
                   help="T = round(horizon_ratio * n) when --T is not given")
    p.add_argument("--dur-min", type=int, default=None,
                   help="minimum item duration (default: max(1, 0.08*T))")
    p.add_argument("--dur-max", type=int, default=None,
                   help="maximum item duration (default: max(2, 0.20*T))")
    # misc
    p.add_argument("--seed", type=int, default=0,
                   help="random seed (same seed + parameters = same instance)")
    p.add_argument("--count", "-m", "--replicates", type=int, default=1,
                   dest="count",
                   help="how many instances to generate per density. "
                        "Omit it and you get 1; --count 20 gives 20, using "
                        "seeds seed, seed+1, ... so every one is different "
                        "and reproducible")
    p.add_argument("--out-dir", default="dinst",
                   help="root output folder; one sub-folder per density is "
                        "created inside it")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite an existing file instead of adding _v2, _v3, ...")
    p.add_argument("--out", default=None,
                   help="explicit output path (overrides the automatic file name)")
    args = p.parse_args()

    # ---- validate inputs ----
    if args.n < 2:
        sys.exit("--n must be at least 2")
    for d in args.d:
        if not 0.0 <= d <= 1.0:
            sys.exit(f"--d values must be between 0 and 1 (got {d})")
    if args.C <= 0:
        sys.exit("--C must be positive")
    if not 0.0 <= args.eta < 1.0:
        sys.exit("--eta must be in [0, 1)")
    if args.a <= 0:
        sys.exit("--a must be positive")
    if args.k_min < 0 or args.k_max < args.k_min:
        sys.exit("need 0 <= --k-min <= --k-max")
    if args.count < 1:
        sys.exit("--count must be at least 1")
    if args.out and (len(args.d) > 1 or args.count > 1):
        sys.exit("--out names a single file; drop it when generating several "
                 "instances (use --out-dir instead)")

    C = float(args.C)
    h = args.eta * C

    # ---- derived defaults (independent of d) ----
    T = args.T if args.T else max(4, round(args.horizon_ratio * args.n))
    dur_min = args.dur_min if args.dur_min else max(1, round(0.08 * T))
    dur_max = args.dur_max if args.dur_max else max(dur_min + 1, round(0.20 * T))
    w_min = args.w_min if args.w_min else max(1, round(0.25 * C))
    w_max = args.w_max if args.w_max else max(w_min + 1, round(0.55 * C))

    if dur_max > T:
        sys.exit(f"--dur-max ({dur_max}) cannot exceed the horizon T ({T})")
    if w_max > C - h:
        print(f"note: w-max ({w_max}) exceeds C-h ({C - h:g}), so the largest items "
              f"must be split; feasibility is checked below.")

    print(f"items {args.n} | C {C:g} | h = eta*C = {h:g} | delta_i = w_i/{args.a:g} "
          f"| k {args.k_min}..{args.k_max}")
    print(f"sizes {w_min}..{w_max} | horizon T {T} | durations {dur_min}..{dur_max}")
    print(f"densities {[f'{d:g}' for d in args.d]} x {args.count} instance(s) each "
          f"-> {args.out_dir}/<density>/\n")

    written = []
    for d in args.d:
        d_str = f"{d:g}"
        # one folder per density: created on first use, reused afterwards
        dens_dir = Path(args.out_dir) / d_str
        dens_dir.mkdir(parents=True, exist_ok=True)

        for r in range(args.count):
            seed = args.seed + r
            rng = random.Random(seed)

            items = make_items(args.n, C, T, w_min, w_max, dur_min, dur_max,
                               args.k_min, args.k_max, args.eta, args.a, rng)
            pairs = overlapping_pairs(items)
            conflicts = make_conflicts(pairs, d, rng)

            errs = check_feasibility(items, C)
            if errs:
                print("INFEASIBLE instance under these parameters:")
                for e in errs[:10]:
                    print("  -", e)
                if len(errs) > 10:
                    print(f"  ... and {len(errs) - 10} more")
                sys.exit("nothing written. Lower --eta, raise --k-max, or raise --a.")

            stats = build_stats(items, conflicts, pairs, C)

            c_str = f"{C:g}"
            base = f"n{args.n}_d{d_str}_C{c_str}"
            name = base if args.count == 1 else f"{base}_s{seed}"
            inst = {
                "name": name,
                "format_version": FORMAT_VERSION,
                "problem": "TBPP-C-LF",
                "capacity": C,
                "n_items": args.n,
                "items": items,
                "conflicts": conflicts,
                "meta": {
                    # kept for compatibility with solve_tbppclf.py, which reads these
                    "tier": f"n{args.n}",
                    "class": f"d{d_str}",
                    "params": {
                        "n": args.n, "d_target": d, "C": C,
                        "eta": args.eta, "h": round(h, 6),
                        "a": args.a, "delta_rule": "delta_i = w_i / a",
                        "k_range": [args.k_min, args.k_max],
                        "w_range": [w_min, w_max],
                        "duration_range": [dur_min, dur_max],
                        "T": T,
                    },
                    "generator": {"script": "gen_instance.py", "seed": seed,
                                  "conflict_rule": "overlapping pairs only"},
                    "stats": stats,
                },
            }

            if args.out:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = dens_dir / f"{name}.json"
                # never silently overwrite an existing instance
                if out_path.exists() and not args.overwrite:
                    stem, v = out_path.stem, 2
                    while out_path.exists():
                        out_path = dens_dir / f"{stem}_v{v}.json"
                        v += 1
            out_path.write_text(json.dumps(inst, indent=1))
            written.append((out_path, stats))

            print(f"  {out_path}   conflicts {stats['n_conflicts']:>7} / "
                  f"{stats['n_overlap_pairs']:<7} eff.density "
                  f"{stats['eff_conflict_density']:<7} cliques "
                  f"{stats['n_maximal_cliques']:<4} LB "
                  f"{stats['lb_peak_bins']}/{stats['lb_peak_bins_overhead']}")

    print(f"\nwrote {len(written)} instance(s) under {args.out_dir}/")
    if len(written) > 1:
        eff = [s["eff_conflict_density"] for _, s in written]
        print(f"effective densities: {min(eff):g} .. {max(eff):g}")
    print("LB shown as peak-load/overhead-aware.")


if __name__ == "__main__":
    main()