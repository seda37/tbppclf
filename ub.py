#!/usr/bin/env python3
"""
Upper bounds for TBPP-C-LF by multi-start construction + bin-emptying local search.

Every solution produced here is FEASIBLE and is verified before being reported:
  - fragments of item i sum to w_i, and there are at most k_i + 1 of them
  - if i is split, every fragment is at least delta_i
  - each fragment occupies its bin for the whole interval [s_i, e_i)
  - at every event time, sum over items alive in bin j of (mass + h_i) <= C
  - conflicting items with overlapping intervals never share a bin

Pair with tbpp_lb.py to measure tightness:
    python tbpp_lb.py inst1 --csv lb.csv
    python tbpp_ub.py inst1 --csv ub.csv --lb lb.csv

Usage
    python tbpp_ub.py instance.json
    python tbpp_ub.py inst1 --csv ub.csv
    python tbpp_ub.py inst1 --starts 40 --lb lb.csv --csv ub.csv

Requires: nothing beyond the standard library (numpy/scipy not needed).
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import time

EPS = 1e-7


# ----------------------------------------------------------------------------
# instance
# ----------------------------------------------------------------------------

class Instance:
    def __init__(self, path):
        with open(path) as fh:
            d = json.load(fh)
        self.path = path
        self.name = d.get("name", os.path.basename(path))
        self.C = float(d["capacity"])
        self.items = {int(it["id"]): it for it in d["items"]}
        self.ids = sorted(self.items)
        self.meta = d.get("meta", {})
        self.adj = {i: set() for i in self.ids}
        for a, b in d["conflicts"]:
            self.adj[int(a)].add(int(b))
            self.adj[int(b)].add(int(a))
        # capacity only needs checking at start times
        self.events = sorted({self.items[i]["s"] for i in self.ids})
        self.eidx = {t: p for p, t in enumerate(self.events)}
        # event positions during which item i is alive
        self.span = {}
        for i in self.ids:
            s, e = self.items[i]["s"], self.items[i]["e"]
            self.span[i] = [p for p, t in enumerate(self.events) if s <= t < e]
        self.fmin = {i: math.ceil(self.items[i]["w"] / (self.C - self.items[i]["h"]))
                     for i in self.ids}

    def overlap(self, i, j):
        a, b = self.items[i], self.items[j]
        return a["s"] < b["e"] and b["s"] < a["e"]

    def conflict(self, i, j):
        return j in self.adj[i]


# ----------------------------------------------------------------------------
# a bin
# ----------------------------------------------------------------------------

class Bin:
    __slots__ = ("load", "mass")

    def __init__(self, n_events):
        self.load = [0.0] * n_events      # consumed capacity per event
        self.mass = {}                     # item id -> mass placed here

    def clone(self, n_events):
        b = Bin(n_events)
        b.load = list(self.load)
        b.mass = dict(self.mass)
        return b

    def blocked_by_conflict(self, inst, i):
        for j in self.mass:
            if j != i and inst.conflict(i, j) and inst.overlap(i, j):
                return True
        return False

    def room_for(self, inst, i):
        """Max additional mass of item i this bin can take (0 if none)."""
        if self.blocked_by_conflict(inst, i):
            return 0.0
        it = inst.items[i]
        extra_h = 0.0 if i in self.mass else it["h"]
        free = inst.C - it["h"] if i not in self.mass else inst.C
        for p in inst.span[i]:
            free = min(free, inst.C - self.load[p] - extra_h)
        cap = (inst.C - it["h"]) - self.mass.get(i, 0.0)
        return max(0.0, min(free, cap))

    def put(self, inst, i, amount):
        it = inst.items[i]
        if i not in self.mass:
            for p in inst.span[i]:
                self.load[p] += it["h"]
            self.mass[i] = 0.0
        for p in inst.span[i]:
            self.load[p] += amount
        self.mass[i] += amount

    def remove(self, inst, i):
        if i not in self.mass:
            return 0.0
        it = inst.items[i]
        amt = self.mass.pop(i)
        for p in inst.span[i]:
            self.load[p] -= amt + it["h"]
        return amt


# ----------------------------------------------------------------------------
# placement
# ----------------------------------------------------------------------------

def place(inst, bins, i, allow_new=True):
    """Place item i across existing bins (and new ones if allowed).
    Returns True on success; leaves bins untouched on failure."""
    it = inst.items[i]
    budget = it["k"] + 1
    delta = it["delta"]
    snapshot = [(b, dict(b.mass), list(b.load)) for b in bins]
    n0 = len(bins)

    rem = float(it["w"])
    used = 0

    # try a whole, unsplit placement first: tightest bin that fits it all
    cands = []
    for b in bins:
        r = b.room_for(inst, i)
        if r >= rem - EPS:
            cands.append((r, b))
    if cands:
        cands.sort(key=lambda x: x[0])       # best fit: least leftover
        cands[0][1].put(inst, i, rem)
        return True

    # otherwise split greedily into the roomiest bins
    if budget > 1:
        room = sorted(((b.room_for(inst, i), b) for b in bins),
                      key=lambda x: -x[0])
        for r, b in room:
            if rem <= EPS or used >= budget:
                break
            if r < delta - EPS:
                continue
            take = min(r, rem)
            tail = rem - take
            if tail > EPS and tail < delta - EPS:
                take = rem - delta          # leave a legal tail
                if take < delta - EPS:
                    continue
            if used == budget - 1 and rem - take > EPS:
                continue                     # last allowed fragment must finish it
            b.put(inst, i, take)
            rem -= take
            used += 1

    # open new bins for whatever is left
    while rem > EPS:
        if used >= budget or not allow_new:
            for b, m, l in snapshot:         # roll back
                b.mass, b.load = m, l
            del bins[n0:]
            return False
        cap = inst.C - it["h"]
        take = min(rem, cap)
        tail = rem - take
        if tail > EPS and tail < delta - EPS:
            take = rem - delta
        if used == budget - 1:
            take = rem
            if take > cap + EPS:
                for b, m, l in snapshot:
                    b.mass, b.load = m, l
                del bins[n0:]
                return False
        nb = Bin(len(inst.events))
        nb.put(inst, i, take)
        bins.append(nb)
        rem -= take
        used += 1
    return True


def construct(inst, order):
    bins = []
    for i in order:
        if not place(inst, bins, i):
            return None
    return bins


# ----------------------------------------------------------------------------
# local search: try to empty bins
# ----------------------------------------------------------------------------

def _try_drain(inst, bins, j, rng, shuffle=False):
    """Attempt to empty bin j into the others. Returns True and mutates bins
    on success; otherwise restores everything and returns False."""
    victim = bins[j]
    others = [b for q, b in enumerate(bins) if q != j]
    backup = [(b, dict(b.mass), list(b.load)) for b in others]
    contents = sorted(victim.mass.items(), key=lambda kv: -kv[1])
    if shuffle:
        rng.shuffle(contents)
    for i, amt in contents:
        elsewhere = sum(1 for b in others if i in b.mass)
        budget = inst.items[i]["k"] + 1 - elsewhere
        if budget <= 0:
            break
        rem = amt
        room = [(b.room_for(inst, i), b) for b in others]
        if shuffle:
            rng.shuffle(room)
            room.sort(key=lambda x: -x[0] * (0.75 + 0.5 * rng.random()))
        else:
            room.sort(key=lambda x: -x[0])
        used = 0
        for r, b in room:
            if rem <= EPS or used >= budget:
                break
            if r <= EPS:
                continue
            take = min(r, rem)
            delta = inst.items[i]["delta"]
            tail = rem - take
            if i not in b.mass and take < delta - EPS and tail > EPS:
                continue
            if tail > EPS and tail < delta - EPS:
                take = rem - delta
                if take <= EPS:
                    continue
            b.put(inst, i, take)
            rem -= take
            used += 1
        if rem > EPS:
            break
    else:
        bins.pop(j)
        return True
    for b, m, l in backup:
        b.mass, b.load = m, l
    return False


def empty_bins(inst, bins, rounds=40, rng=None):
    """Repeatedly try to empty a bin and redistribute its contents.
    Deterministic passes first, then randomized retries."""
    rng = rng or random.Random(0)
    for rd in range(rounds):
        if len(bins) <= 1:
            break
        order = sorted(range(len(bins)), key=lambda j: sum(bins[j].mass.values()))
        moved = False
        for j in order:
            if _try_drain(inst, bins, j, rng, shuffle=False):
                moved = True
                break
        if not moved:
            for _ in range(3 * len(bins)):
                j = rng.randrange(len(bins))
                if _try_drain(inst, bins, j, rng, shuffle=True):
                    moved = True
                    break
        if not moved:
            break
    return bins


def _legacy_unused(inst, bins, rounds=6, rng=None):
    rng = rng or random.Random(0)
    for _ in range(rounds):
        if len(bins) <= 1:
            break
        order = sorted(range(len(bins)), key=lambda j: sum(bins[j].mass.values()))
        moved = False
        for j in order[:max(1, len(bins) // 3)]:
            victim = bins[j]
            others = [b for q, b in enumerate(bins) if q != j]
            backup = [(b, dict(b.mass), list(b.load)) for b in others]
            contents = sorted(victim.mass.items(), key=lambda kv: -kv[1])
            ok = True
            for i, amt in contents:
                # how many fragments does i already have elsewhere?
                elsewhere = sum(1 for b in others if i in b.mass)
                budget = inst.items[i]["k"] + 1 - elsewhere
                if budget <= 0:
                    ok = False
                    break
                rem = amt
                room = sorted(((b.room_for(inst, i), b) for b in others),
                              key=lambda x: -x[0])
                used = 0
                for r, b in room:
                    if rem <= EPS or used >= budget:
                        break
                    if r <= EPS:
                        continue
                    take = min(r, rem)
                    b.put(inst, i, take)
                    rem -= take
                    used += 1
                if rem > EPS:
                    ok = False
                    break
            if ok:
                bins.pop(j)
                moved = True
                break
            for b, m, l in backup:
                b.mass, b.load = m, l
        if not moved:
            break
    return bins


# ----------------------------------------------------------------------------
# verification -- never report an unverified solution
# ----------------------------------------------------------------------------

def verify(inst, bins):
    frags = {i: 0 for i in inst.ids}
    total = {i: 0.0 for i in inst.ids}
    for b in bins:
        for i, m in b.mass.items():
            if m < -EPS:
                return "negative mass on item %d" % i
            frags[i] += 1
            total[i] += m
        # conflicts
        members = list(b.mass)
        for a in range(len(members)):
            for c in range(a + 1, len(members)):
                x, y = members[a], members[c]
                if inst.conflict(x, y) and inst.overlap(x, y):
                    return "conflict %d/%d share a bin" % (x, y)
        # capacity at every event
        for p, t in enumerate(inst.events):
            load = sum(m + inst.items[i]["h"]
                       for i, m in b.mass.items()
                       if inst.items[i]["s"] <= t < inst.items[i]["e"])
            if load > inst.C + 1e-6:
                return "capacity exceeded at t=%d (%.4f > %g)" % (t, load, inst.C)
    for i in inst.ids:
        it = inst.items[i]
        if abs(total[i] - it["w"]) > 1e-6:
            return "item %d mass %.4f != %g" % (i, total[i], it["w"])
        if frags[i] > it["k"] + 1:
            return "item %d has %d fragments, budget %d" % (i, frags[i], it["k"] + 1)
        if frags[i] > 1:
            for b in bins:
                if i in b.mass and b.mass[i] < it["delta"] - 1e-6:
                    return "item %d fragment %.4f below delta %g" % (i, b.mass[i], it["delta"])
    return None


# ----------------------------------------------------------------------------
# driver for one instance
# ----------------------------------------------------------------------------

def solve(inst, starts=24, seed=0, time_limit=60.0):
    rng = random.Random(seed)
    it = inst.items
    orders = [
        sorted(inst.ids, key=lambda i: -it[i]["w"]),
        sorted(inst.ids, key=lambda i: -(it[i]["w"] + it[i]["h"])),
        sorted(inst.ids, key=lambda i: (-len(inst.adj[i]), -it[i]["w"])),
        sorted(inst.ids, key=lambda i: (it[i]["s"], -it[i]["w"])),
        sorted(inst.ids, key=lambda i: -(it[i]["e"] - it[i]["s"])),
        sorted(inst.ids, key=lambda i: (it[i]["k"], -it[i]["w"])),
    ]
    best, best_bins, tried = None, None, 0
    t0 = time.time()
    for s in range(max(len(orders), starts)):
        if time.time() - t0 > time_limit:
            break
        if s < len(orders):
            order = orders[s]
        else:
            base = orders[rng.randrange(len(orders))]
            order = list(base)
            for _ in range(max(2, len(order) // 8)):     # light perturbation
                a = rng.randrange(len(order)); b = rng.randrange(len(order))
                order[a], order[b] = order[b], order[a]
        bins = construct(inst, order)
        if bins is None:
            continue
        tried += 1
        bins = empty_bins(inst, bins, rng=rng)
        if best is None or len(bins) < best:
            err = verify(inst, bins)
            if err:
                continue                      # never accept an invalid solution
            best, best_bins = len(bins), bins
    return best, best_bins, tried


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def collect(pattern):
    if os.path.isdir(pattern):
        root = os.path.abspath(pattern)
        paths = sorted(glob.glob(os.path.join(pattern, "**", "*.json"), recursive=True))
        return root, [p for p in paths if os.path.isfile(p)]
    paths = sorted(glob.glob(pattern))
    if not paths and os.path.exists(pattern):
        paths = [pattern]
    return None, [p for p in paths if os.path.isfile(p)]


def group_of(path, root):
    if root is None:
        return os.path.basename(os.path.dirname(os.path.abspath(path))) or "-"
    rel = os.path.relpath(os.path.dirname(os.path.abspath(path)), root)
    return "-" if rel in (".", "") else rel.replace(os.sep, "/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", help="instance file, glob, or folder (searched recursively)")
    ap.add_argument("--starts", type=int, default=24, help="multi-start count (default 24)")
    ap.add_argument("--time-limit", type=float, default=60.0, help="seconds per instance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lb", help="CSV from tbpp_lb.py; merges LB_mc and reports the gap")
    ap.add_argument("--csv", help="write results here")
    args = ap.parse_args()

    root, paths = collect(args.pattern)
    if not paths:
        print("no instances found under: %s" % args.pattern)
        print("hint: quote globs, e.g.  python tbpp_ub.py 'inst1/*.json'")
        return

    lbmap = {}
    if args.lb:
        try:
            with open(args.lb) as fh:
                for r in csv.DictReader(fh):
                    lbmap[r["name"]] = int(r["LB_mc"])
        except Exception as exc:
            print("could not read %s: %s" % (args.lb, exc))

    print("found %d instances\n" % len(paths))
    hdr = ("%-16s %-24s %5s %6s %6s %6s %8s %7s" %
           ("folder", "instance", "n", "UB", "LB_mc", "gap", "gap%", "sec"))
    print(hdr); print("-" * len(hdr))

    rows = []
    for p in paths:
        inst = Instance(p)
        t0 = time.time()
        ub, bins, tried = solve(inst, starts=args.starts, seed=args.seed,
                                time_limit=args.time_limit)
        secs = time.time() - t0
        if ub is None:
            print("%-16s %-24s  no feasible solution found" %
                  (group_of(p, root)[:16], inst.name[:24]))
            continue
        lb = lbmap.get(inst.name)
        gap = (ub - lb) if lb is not None else None
        gpct = (100.0 * gap / ub) if (gap is not None and ub) else None
        rows.append(dict(group=group_of(p, root), name=inst.name, n=len(inst.ids),
                         UB=ub, LB_mc=(lb if lb is not None else ""),
                         gap=(gap if gap is not None else ""),
                         gap_pct=(round(gpct, 2) if gpct is not None else ""),
                         starts_ok=tried, sec=round(secs, 2)))
        print("%-16s %-24s %5d %6d %6s %6s %8s %7.1f" %
              (group_of(p, root)[:16], inst.name[:24], len(inst.ids), ub,
               lb if lb is not None else "-",
               gap if gap is not None else "-",
               ("%.1f%%" % gpct) if gpct is not None else "-", secs))

    if rows:
        print("-" * len(hdr))
        gaps = [r["gap"] for r in rows if r["gap"] != ""]
        if gaps:
            groups = sorted({r["group"] for r in rows})
            if len(groups) > 1:
                print("%-16s %5s %8s %8s %8s %8s" %
                      ("folder", "N", "mean UB", "mean LB", "mean gap", "closed"))
                for g in groups:
                    sub = [r for r in rows if r["group"] == g and r["gap"] != ""]
                    if not sub:
                        continue
                    print("%-16s %5d %8.2f %8.2f %8.2f %8d" %
                          (g[:16], len(sub),
                           sum(r["UB"] for r in sub) / len(sub),
                           sum(r["LB_mc"] for r in sub) / len(sub),
                           sum(r["gap"] for r in sub) / len(sub),
                           sum(1 for r in sub if r["gap"] == 0)))
                print("-" * len(hdr))
            print("instances %d   mean gap %.2f bins   proven optimal %d/%d"
                  % (len(rows), sum(gaps) / len(gaps),
                     sum(1 for g in gaps if g == 0), len(gaps)))
        else:
            print("instances %d   mean UB %.2f  (pass --lb to see gaps)"
                  % (len(rows), sum(r["UB"] for r in rows) / len(rows)))

    if args.csv and rows:
        keys = list(rows[0])
        for attempt in (args.csv, os.path.join(os.path.expanduser("~"),
                                               os.path.basename(args.csv))):
            try:
                with open(attempt, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=keys)
                    w.writeheader()
                    w.writerows(rows)
                print("wrote %s" % os.path.abspath(attempt))
                break
            except PermissionError:
                print("PermissionError writing %s (open in Excel?)" % attempt)
            except OSError as exc:
                print("could not write %s: %s" % (attempt, exc))


if __name__ == "__main__":
    main()