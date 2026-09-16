#!/usr/bin/env python3
"""
Lower bounds for TBPP-C-LF (temporal bin packing, conflicts, limited fragmentation).

Computes four bounds and reports them side by side:

  LB_w    weighing         : peak over event times of  ceil( sum w_i / C )
  LB_o    weighing+overhead: same, but each item also pays h_i * fmin_i
  LB_q    clique           : sum of fmin_i over a maximum clique
  LB_mc   MC-LB            : open the clique's bins, absorb what legally fits,
                             count the stranded mass  (this is the only bound
                             that sees conflicts and capacity together)

VALIDITY.  MC-LB is obtained from the real problem by deletion only:
  R1  delete every item not alive at the chosen event time t*
  R2  partition bins into "belongs to clique item q" and "no clique item";
      the latter are replaced by a continuous mass count
  R3  drop conflicts between outside items and leftover bins
Original constraints (split budgets k_i, min fragment size, conflicts among
outside items in clique bins) are KEPT, which is always safe.  Nothing is
added that the real problem does not already impose, so no interchange
argument is required.

Note: the number of bins opened per clique item is a decision variable in
[fmin_q, k_q+1], not a fixed constant.  That is what removes the need for
Ekici's Theorem 1 style proof, which does not survive per-fragment overhead.

Usage
  python tbpp_lb.py instance.json
  python tbpp_lb.py 'instances/*.json' --csv results.csv
  python tbpp_lb.py 'instances/*.json' --events 5 --greedy-clique

Requires: numpy, scipy (>=1.9 for scipy.optimize.milp)
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


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
        self.fmin = {i: math.ceil(self.items[i]["w"] / (self.C - self.items[i]["h"]))
                     for i in self.ids}
        self.events = sorted({self.items[i]["s"] for i in self.ids})

    def alive(self, i, t):
        return self.items[i]["s"] <= t < self.items[i]["e"]

    def active(self, t):
        return [i for i in self.ids if self.alive(i, t)]

    def conflict(self, i, j):
        return j in self.adj[i]

    def check_feasible(self):
        """Standing assumption: oversized items must be splittable enough."""
        bad = []
        for i in self.ids:
            it = self.items[i]
            cap = self.C - it["h"]
            if cap <= 0:
                bad.append((i, "h >= C"))
            elif self.fmin[i] > min(it["k"] + 1, math.floor(it["w"] / it["delta"])):
                bad.append((i, "needs %d fragments, budget %d" % (self.fmin[i], it["k"] + 1)))
            elif it["delta"] > cap:
                bad.append((i, "delta > C-h"))
        return bad


# ----------------------------------------------------------------------------
# cliques
# ----------------------------------------------------------------------------

def max_clique(adj, nodes, time_limit=10.0):
    """Bron-Kerbosch with pivoting and a size bound. Falls back to the best
    clique found so far if time_limit is hit."""
    best = [[]]
    deadline = time.time() + time_limit
    timed_out = [False]

    def expand(R, P, X):
        if timed_out[0]:
            return
        if time.time() > deadline:
            timed_out[0] = True
            return
        if not P and not X:
            if len(R) > len(best[0]):
                best[0] = list(R)
            return
        if len(R) + len(P) <= len(best[0]):
            return
        pivot = max(P | X, key=lambda u: len(adj[u] & P))
        for v in list(P - adj[pivot]):
            expand(R + [v], P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}

    sys.setrecursionlimit(20000)
    expand([], set(nodes), set())
    return best[0], timed_out[0]


def greedy_clique(adj, nodes):
    """Johnson's first heuristic: repeatedly add the most connected vertex."""
    nodes = list(nodes)
    deg = {i: len(adj[i] & set(nodes)) for i in nodes}
    Q = []
    for i in sorted(nodes, key=lambda x: -deg[x]):
        if all(j in adj[i] for j in Q):
            Q.append(i)
    return Q


# ----------------------------------------------------------------------------
# simple bounds
# ----------------------------------------------------------------------------

def weighing_bounds(inst):
    lb_w = lb_o = 0
    t_w = t_o = None
    peak = 0
    for t in inst.events:
        S = inst.active(t)
        peak = max(peak, len(S))
        w = math.ceil(sum(inst.items[i]["w"] for i in S) / inst.C - 1e-9)
        o = math.ceil(sum(inst.items[i]["w"] + inst.items[i]["h"] * inst.fmin[i]
                          for i in S) / inst.C - 1e-9)
        if w > lb_w:
            lb_w, t_w = w, t
        if o > lb_o:
            lb_o, t_o = o, t
    return lb_w, lb_o, t_o, peak


# ----------------------------------------------------------------------------
# MC-LB
# ----------------------------------------------------------------------------

class _Model:
    """Thin builder around scipy.optimize.milp."""

    def __init__(self):
        self.cost, self.integ, self.lo, self.up = [], [], [], []
        self.rows = []

    def var(self, cost=0.0, integer=False, lo=0.0, up=np.inf):
        self.cost.append(cost)
        self.integ.append(1 if integer else 0)
        self.lo.append(lo)
        self.up.append(up)
        return len(self.cost) - 1

    def row(self, coefs, lo=-np.inf, up=np.inf):
        self.rows.append((dict(coefs), lo, up))

    def solve(self, time_limit=None):
        n = len(self.cost)
        A = np.zeros((len(self.rows), n))
        rl = np.empty(len(self.rows))
        ru = np.empty(len(self.rows))
        for r, (coefs, lo, up) in enumerate(self.rows):
            for k, v in coefs.items():
                A[r, k] += v
            rl[r], ru[r] = lo, up
        opts = {}
        if time_limit:
            opts["time_limit"] = time_limit
        return milp(c=np.array(self.cost, float),
                    constraints=LinearConstraint(A, rl, ru),
                    integrality=np.array(self.integ),
                    bounds=Bounds(np.array(self.lo, float), np.array(self.up, float)),
                    options=opts)


def mc_lb_at(inst, t, Q, outside_conflicts=True, time_limit=60.0):
    """MC-LB evaluated at event time t with clique Q (subset of items alive at t).

    Returns (bound, detail_dict) or (None, reason)."""
    C = inst.C
    S = set(inst.active(t))
    Q = [q for q in Q if q in S]
    if not Q:
        return None, "empty clique"
    O = [i for i in S if i not in Q]

    m = _Model()

    # --- bins belonging to clique items -------------------------------------
    # r ranges over 0 .. k_q, i.e. at most k_q + 1 bins may hold item q.
    # y is a DECISION, not a constant: the model may open more than fmin_q.
    y, g = {}, {}
    bins = []
    for q in Q:
        nmax = inst.items[q]["k"] + 1
        for r in range(nmax):
            b = len(bins)
            bins.append((q, r))
            y[b] = m.var(cost=1.0, integer=True, lo=0, up=1)   # bin opened
            g[b] = m.var(lo=0, up=C - inst.items[q]["h"])      # mass of q here

    # --- outside items in clique bins ---------------------------------------
    f, u = {}, {}
    for b, (q, r) in enumerate(bins):
        for o in O:
            if inst.conflict(o, q):
                continue                       # locked out: no variable at all
            f[(o, b)] = m.var(lo=0, up=inst.items[o]["w"])
            u[(o, b)] = m.var(integer=True, lo=0, up=1)

    # --- leftover -----------------------------------------------------------
    L = {o: m.var(lo=0, up=inst.items[o]["w"]) for o in O}    # stranded mass
    v = {o: m.var(integer=True, lo=0, up=1) for o in O}       # leftover nonempty
    z = m.var(cost=1.0, integer=True, lo=0, up=len(O) + len(bins) + 5)

    # --- clique item fully placed across its own bins -----------------------
    for q in Q:
        idx = [b for b, (qq, r) in enumerate(bins) if qq == q]
        m.row({g[b]: 1.0 for b in idx}, inst.items[q]["w"], inst.items[q]["w"])
        for b in idx:
            m.row({g[b]: 1.0, y[b]: -(C - inst.items[q]["h"])}, up=0.0)
        # symmetry breaking among the copies of q
        for b1, b2 in zip(idx, idx[1:]):
            m.row({y[b1]: 1.0, y[b2]: -1.0}, lo=0.0)

    # --- bin capacity, overhead charged per fragment ------------------------
    for b, (q, r) in enumerate(bins):
        co = {g[b]: 1.0, y[b]: inst.items[q]["h"] - C}
        for o in O:
            if (o, b) in f:
                co[f[(o, b)]] = 1.0
                co[u[(o, b)]] = inst.items[o]["h"]
        m.row(co, up=0.0)

    # --- outside item mass conservation -------------------------------------
    for o in O:
        co = {L[o]: 1.0}
        for b in range(len(bins)):
            if (o, b) in f:
                co[f[(o, b)]] = 1.0
        m.row(co, inst.items[o]["w"], inst.items[o]["w"])

    # --- linking, min fragment size, bin must be open -----------------------
    for (o, b), fi in f.items():
        ui = u[(o, b)]
        m.row({fi: 1.0, ui: -min(inst.items[o]["w"], C - inst.items[o]["h"])}, up=0.0)
        m.row({fi: 1.0, ui: -inst.items[o]["delta"]}, lo=0.0)
        m.row({ui: 1.0, y[b]: -1.0}, up=0.0)

    # --- split budgets (original constraints, kept) -------------------------
    for o in O:
        us = {u[(o, b)]: 1.0 for b in range(len(bins)) if (o, b) in f}
        cap = C - inst.items[o]["h"]
        budget = inst.items[o]["k"] + 1
        # fragments in clique bins + at least one more if anything is stranded
        co = dict(us)
        co[v[o]] = co.get(v[o], 0) + 1.0
        m.row(co, up=budget)
        # stranded mass must fit in the remaining fragment allowance
        co2 = {k: val * cap for k, val in us.items()}
        co2[L[o]] = co2.get(L[o], 0) + 1.0
        m.row(co2, up=budget * cap)
        # link v to L
        m.row({L[o]: 1.0, v[o]: -inst.items[o]["w"]}, up=0.0)

    # --- conflicts among outside items inside a clique bin ------------------
    if outside_conflicts:
        for b in range(len(bins)):
            for a_ in range(len(O)):
                for b_ in range(a_ + 1, len(O)):
                    o1, o2 = O[a_], O[b_]
                    if inst.conflict(o1, o2) and (o1, b) in f and (o2, b) in f:
                        m.row({u[(o1, b)]: 1.0, u[(o2, b)]: 1.0}, up=1.0)

    # --- stranded mass needs extra bins (each leftover pays its overhead) ---
    co = {}
    for o in O:
        co[L[o]] = 1.0
        co[v[o]] = inst.items[o]["h"]
    co[z] = -C
    m.row(co, up=0.0)

    res = m.solve(time_limit=time_limit)
    if not res.success or res.x is None:
        return None, "solver: %s" % getattr(res, "message", "failed")

    lb = math.ceil(res.fun - 1e-6)
    detail = dict(t=t, clique=len(Q), outside=len(O),
                  bins_opened=int(round(sum(res.x[y[b]] for b in range(len(bins))))),
                  extra_bins=int(round(res.x[z])),
                  nvars=len(m.cost), nrows=len(m.rows))
    return lb, detail


def mc_lb(inst, n_events=3, use_greedy=False, outside_conflicts=True,
          time_limit=60.0, clique_time=10.0):
    """Evaluate MC-LB at the most promising event times, return the best."""
    # rank events by the overhead-weighing value: where capacity is tightest
    ranked = sorted(inst.events,
                    key=lambda t: -sum(inst.items[i]["w"] + inst.items[i]["h"] * inst.fmin[i]
                                       for i in inst.active(t)))
    cands = ranked[:max(1, n_events)]

    # the global max clique is simultaneously alive whenever conflicts only
    # exist between overlapping pairs; add its window to the candidate list
    if use_greedy:
        Qg = greedy_clique(inst.adj, inst.ids)
        cut = False
    else:
        Qg, cut = max_clique(inst.adj, inst.ids, time_limit=clique_time)
    if Qg:
        lo = max(inst.items[i]["s"] for i in Qg)
        hi = min(inst.items[i]["e"] for i in Qg)
        if lo < hi and lo not in cands:
            cands.append(lo)

    best, best_detail = 0, None
    for t in cands:
        S = inst.active(t)
        if use_greedy:
            Q = greedy_clique(inst.adj, S)
        else:
            Q, _ = max_clique(inst.adj, S, time_limit=clique_time)
        val, det = mc_lb_at(inst, t, Q, outside_conflicts=outside_conflicts,
                            time_limit=time_limit)
        if val is not None and val > best:
            best, best_detail = val, det
    return best, best_detail, len(Qg), cut


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def group_of(path, root):
    """Folder label for an instance: path relative to the search root."""
    if root is None:
        return os.path.basename(os.path.dirname(os.path.abspath(path))) or "-"
    rel = os.path.relpath(os.path.dirname(os.path.abspath(path)), root)
    return "-" if rel in (".", "") else rel.replace(os.sep, "/")


def run_one(path, args, root=None):
    inst = Instance(path)
    infeas = inst.check_feasible()
    t0 = time.time()
    lb_w, lb_o, t_o, peak = weighing_bounds(inst)
    t_simple = time.time() - t0

    t0 = time.time()
    lb_mc, det, clique_size, cut = mc_lb(
        inst, n_events=args.events, use_greedy=args.greedy_clique,
        outside_conflicts=not args.no_outside_conflicts,
        time_limit=args.time_limit, clique_time=args.clique_time)
    t_mc = time.time() - t0

    Q, _ = ((greedy_clique(inst.adj, inst.ids), False) if args.greedy_clique
            else max_clique(inst.adj, inst.ids, time_limit=args.clique_time))
    lb_q = sum(inst.fmin[i] for i in Q)

    simple = max(lb_w, lb_o, lb_q)
    return dict(
        group=group_of(path, root),
        name=inst.name,
        n=len(inst.ids),
        C=inst.C,
        density=inst.meta.get("stats", {}).get("eff_conflict_density", ""),
        peak_active=peak,
        clique=len(Q),
        LB_w=lb_w,
        LB_o=lb_o,
        LB_q=lb_q,
        LB_mc=lb_mc,
        best_simple=simple,
        gain=lb_mc - simple,
        infeasible_items=len(infeas),
        clique_timeout=int(cut),
        t_simple=round(t_simple, 2),
        t_mc=round(t_mc, 2),
        detail=det,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", help="instance file or glob, e.g. 'inst/*.json'")
    ap.add_argument("--events", type=int, default=3,
                    help="how many candidate event times to evaluate (default 3)")
    ap.add_argument("--time-limit", type=float, default=60.0,
                    help="MILP time limit per event, seconds (default 60)")
    ap.add_argument("--clique-time", type=float, default=10.0,
                    help="max-clique time limit, seconds (default 10)")
    ap.add_argument("--greedy-clique", action="store_true",
                    help="use Johnson's greedy clique instead of exact")
    ap.add_argument("--no-outside-conflicts", action="store_true",
                    help="drop conflicts among outside items (weaker, faster)")
    ap.add_argument("--csv", help="write results to this CSV file")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pat = args.pattern
    root = None
    if os.path.isdir(pat):
        root = os.path.abspath(pat)
        paths = sorted(glob.glob(os.path.join(pat, "**", "*.json"), recursive=True))
        if not paths:
            print("no .json files found anywhere under: %s" % pat)
            return
        ndirs = len({os.path.dirname(q) for q in paths})
        print("found %d instances in %d folder(s) under %s\n" % (len(paths), ndirs, pat))
    else:
        paths = sorted(glob.glob(pat))
        if not paths:
            if os.path.exists(pat):
                paths = [pat]
            else:
                print("no files match: %s" % pat)
                print("hint: quote globs, e.g.  python tbpp_lb.py 'inst1/*.json'")
                return
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("nothing to do")
        return
    rows = []
    hdr = ("%-18s %-24s %5s %6s %7s %6s %6s %6s %6s %7s %6s" %
           ("folder", "instance", "n", "dens", "clique", "LB_w", "LB_o",
            "LB_q", "LB_mc", "gain", "sec"))
    print(hdr)
    print("-" * len(hdr))
    for p in paths:
        try:
            r = run_one(p, args, root=root)
        except Exception as exc:
            print("%-26s  ERROR: %s" % (os.path.basename(p)[:26], exc))
            continue
        rows.append(r)
        dens = ("%.2f" % r["density"]) if r["density"] != "" else "-"
        flag = " *" if r["gain"] > 0 else ""
        print("%-18s %-24s %5d %6s %7d %6d %6d %6d %6d %+7d %6.1f%s" %
              (r["group"][:18], r["name"][:24], r["n"], dens, r["clique"],
               r["LB_w"], r["LB_o"], r["LB_q"], r["LB_mc"], r["gain"],
               r["t_mc"], flag))
        if args.verbose and r["detail"]:
            print("      %s" % r["detail"])
        if r["infeasible_items"]:
            print("      WARNING: %d items violate the splittability assumption"
                  % r["infeasible_items"])
        if r["clique_timeout"]:
            print("      note: max-clique hit its time limit, LB_q may be loose")

    if rows:
        gains = [r["gain"] for r in rows]
        print("-" * len(hdr))
        groups = sorted({r["group"] for r in rows})
        if len(groups) > 1:
            print("%-18s %5s %7s %8s %8s %8s" %
                  ("folder", "n", "mean d", "mean LB_mc", "mean gain", "n better"))
            for g in groups:
                sub = [r for r in rows if r["group"] == g]
                ds = [r["density"] for r in sub if r["density"] != ""]
                print("%-18s %5d %7s %8.2f %+8.2f %8d" %
                      (g[:18], len(sub),
                       ("%.2f" % (sum(ds) / len(ds))) if ds else "-",
                       sum(r["LB_mc"] for r in sub) / len(sub),
                       sum(r["gain"] for r in sub) / len(sub),
                       sum(1 for r in sub if r["gain"] > 0)))
            print("-" * len(hdr))
        print("instances: %d   MC-LB strictly better on %d   mean gain %+.2f   max gain %+d"
              % (len(rows), sum(1 for g in gains if g > 0),
                 sum(gains) / len(gains), max(gains)))

    if args.csv and rows:
        keys = [k for k in rows[0] if k != "detail"]
        target = args.csv
        for attempt in (target, os.path.join(os.path.expanduser("~"),
                                             os.path.basename(target))):
            try:
                with open(attempt, "w", newline="") as fh:
                    wtr = csv.DictWriter(fh, fieldnames=keys)
                    wtr.writeheader()
                    for r in rows:
                        wtr.writerow({k: r[k] for k in keys})
                print("wrote %s" % os.path.abspath(attempt))
                break
            except PermissionError:
                print("PermissionError writing %s" % attempt)
                print("  the file is probably open in Excel, or the folder is not writable")
            except OSError as exc:
                print("could not write %s: %s" % (attempt, exc))
        else:
            print("results not saved; the table above is still valid")


if __name__ == "__main__":
    main()