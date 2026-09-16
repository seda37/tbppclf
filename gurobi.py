#!/usr/bin/env python3
"""
tbpp_exact.py -- exact MIP for TBPP-C-LF (min number of distinct bins).

Replaces gurobi.py. Self-contained: only gurobipy + stdlib. Reads the
gen_instance.py format_version 4.x files directly (builds the maximal cliques
from the intervals itself), so loader.py is optional.

MODEL -- exactly the spec, nothing added to the objective:

    min  sum_j y_j                                        (distinct bins)
    s.t. sum_j x_ij = 1                                   (whole item placed)
         w_i x_ij <= min(w_i, C - h_i) u_ij               (lifted VUB)
         w_i x_ij >= delta_i u_ij                         (min fragment size)
         sum_{i in K} (w_i x_ij + h_i u_ij) <= C y_j      (clique capacity)
         u_ij + u_i'j <= 1        for (i,i') in Ebar      (overlapping conflicts)
         sum_j u_ij <= k_i + 1                            (split budget)
         u_ij <= y_j ,  y_j >= y_{j+1}                    (linking + symmetry)


Valid inequalities added (all derived from the data, not from the heuristic,
so they stay in even when the warm start is switched off):
    sum_j y_j >= LB,  y_j = 1 for j < LB,
    sum_j u_ij >= ceil(w_i / (C - h_i))

Usage:
    python gurobi.py inst.json
    python gurobi.py main/ --timelimit 300 --csv runs.csv
    python gurobi.py main/ --no-warmstart          # honest greedy-vs-MIP
    python gurobi.py inst.json --free-splits       # reference: k_i = inf
    python gurobi.py inst.json --eta 0.2           # override h_i = eta * C

One line of output per instance, reporting the final incumbent only.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

TOL = 1e-5
FTOL = 1e-9          # guard against float noise inside ceil()/floor()


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def build_cliques(items):
    """Maximal cliques of the interval graph. An interval graph's maximal
    cliques are exactly the sets of items alive at some START time, minus the
    ones strictly contained in another such set."""
    starts = sorted({it["s"] for it in items})
    raw = []
    for t in starts:
        K = frozenset(i for i, it in enumerate(items) if it["s"] <= t < it["e"])
        if K:
            raw.append((t, K))
    sets = [K for _, K in raw]
    out, seen = [], set()
    for t, K in raw:
        if any(K < K2 for K2 in sets) or K in seen:
            continue
        seen.add(K)
        out.append((t, tuple(sorted(K))))
    out.sort(key=lambda p: p[0])
    return [K for _, K in out], [t for t, _ in out]


def load_instance(path):
    """Read + normalise an instance. Accepts 'conflicts' or 'conflict_pairs',
    builds 'cliques' when the file does not carry them."""
    inst = json.loads(Path(path).read_text())

    if "conflict_pairs" not in inst:
        inst["conflict_pairs"] = inst.get("conflicts", [])
    inst["conflict_pairs"] = [tuple(sorted(map(int, e)))
                              for e in inst["conflict_pairs"]]

    inst["capacity"] = float(inst["capacity"])
    for idx, it in enumerate(inst["items"]):
        for f in ("s", "e", "w", "h", "delta"):
            if f in it:
                it[f] = float(it[f])
        it["delta"] = float(it.get("delta", 0.0))
        it["k"] = int(it["k"])

    if not inst.get("cliques"):
        inst["cliques"], inst["clique_times"] = build_cliques(inst["items"])
    else:
        inst["cliques"] = [tuple(sorted(map(int, K))) for K in inst["cliques"]]
        inst.setdefault("clique_times",
                        [min(inst["items"][i]["s"] for i in K)
                         for K in inst["cliques"]])

    inst.setdefault("n_items", len(inst["items"]))
    inst.setdefault("name", Path(path).stem)
    return inst


def apply_overhead(inst, eta=None):
    """If eta is given, overwrite h_i = eta * C (for the overhead sweep);
    otherwise keep the h_i stored in the file. Then check the standing
    feasibility assumption. Returns the common h, or None if heterogeneous."""
    C = inst["capacity"]
    missing = [i for i, it in enumerate(inst["items"]) if "h" not in it]
    if missing and eta is None:
        raise ValueError(f"items {missing[:5]} carry no 'h' field; either fix "
                         f"the instance or pass --eta")

    errs = []
    for i, it in enumerate(inst["items"]):
        if eta is not None:
            it["h"] = eta * C
        h = float(it["h"])
        it["h"] = h
        if h < 0 or h >= C:
            errs.append(f"item {i}: overhead h={h:g} outside [0, C={C:g})")
            continue
        cap = C - h
        m_lo = math.ceil(it["w"] / cap - FTOL)          # fragments needed
        m_hi = it["k"] + 1                              # fragments allowed
        if it["delta"] > 0:
            m_hi = min(m_hi, math.floor(it["w"] / it["delta"] + FTOL))
        if m_lo > m_hi:
            errs.append(f"item {i}: needs >={m_lo} fragments (w={it['w']:g}, "
                        f"C-h={cap:g}) but k/delta allow at most {m_hi}")
        if it["delta"] > cap:
            errs.append(f"item {i}: delta={it['delta']:g} > C-h={cap:g}")
    if errs:
        raise ValueError("infeasible by construction:\n  " + "\n  ".join(errs))

    hs = sorted({it["h"] for it in inst["items"]})
    return hs[0] if len(hs) == 1 else None


# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
def min_frags(it, C):
    """Smallest number of fragments item i can possibly be packed into."""
    return max(1, math.ceil(it["w"] / (C - it["h"]) - FTOL))


def greedy_clique_number(nodes, adj):
    """Cheap lower bound on the clique number of the conflict subgraph induced
    on `nodes`. Mutually conflicting items need one bin each, so this is a
    valid bin lower bound at that time step."""
    nodes = sorted(nodes, key=lambda v: -len(adj[v] & set(nodes)))
    cl = []
    for v in nodes:
        if all(u in adj[v] for u in cl):
            cl.append(v)
    return max(1, len(cl))


class _Packing:
    """Incremental packing state shared by the greedy constructors."""

    def __init__(self, inst):
        self.inst, self.C = inst, inst["capacity"]
        self.load, self.members = [], []
        self.assign = {i: [] for i in range(len(inst["items"]))}

    def new_bin(self):
        self.load.append({})
        self.members.append({})
        return len(self.load) - 1

    def residual(self, i, j):
        """Free capacity for a new fragment of item i in bin j, BEFORE paying
        h_i; -inf if a conflicting item already sits in that bin."""
        inst = self.inst
        if set(self.members[j]) & inst["_conf_adj"][i]:
            return float("-inf")
        free = min(self.C - self.load[j].get(t, 0.0)
                   for t in inst["_cliques_of"][i])
        return free - inst["items"][i]["h"]

    def place(self, i, j, amount):
        it = self.inst["items"][i]
        for t in self.inst["_cliques_of"][i]:
            self.load[j][t] = self.load[j].get(t, 0.0) + amount + it["h"]
        self.members[j][i] = self.members[j].get(i, 0) + 1
        self.assign[i].append((j, amount))

    def result(self):
        used = sorted({b for fr in self.assign.values() for b, _ in fr})
        remap = {b: r for r, b in enumerate(used)}
        return len(used), {i: [(remap[b], a) for b, a in fr]
                           for i, fr in self.assign.items()}


def greedy_whole_then_fresh(inst):
    """The original heuristic: first-fit whole (decreasing w); if the item does
    not fit whole anywhere, split it evenly over brand-new bins."""
    items, C = inst["items"], inst["capacity"]
    P = _Packing(inst)
    for i in sorted(range(len(items)), key=lambda i: -items[i]["w"]):
        it = items[i]
        if it["w"] <= C - it["h"] + TOL:
            for j in range(len(P.load)):
                if P.residual(i, j) >= it["w"] - TOL:
                    P.place(i, j, it["w"])
                    break
            else:
                P.place(i, P.new_bin(), it["w"])
        else:
            m = min_frags(it, C)
            for _ in range(m):
                P.place(i, P.new_bin(), it["w"] / m)
    return P.result()


def greedy_split_fit(inst):
    """Fragment-aware constructor: whole worst-fit first, else fill existing
    bins with legal fragments (respecting k_i, delta_i, conflicts and every
    clique the item touches) and only then open a bin."""
    items, C = inst["items"], inst["capacity"]
    P = _Packing(inst)
    order = sorted(range(len(items)), key=lambda i: (-items[i]["w"],
                                                     items[i]["k"]))
    for i in order:
        it = items[i]
        cap_i, budget = C - it["h"], it["k"] + 1
        if it["w"] <= cap_i + TOL:
            cands = [(P.residual(i, j), j) for j in range(len(P.load))]
            cands = [c for c in cands if c[0] >= it["w"] - TOL]
            if cands:
                P.place(i, max(cands)[1], it["w"])     # worst-fit: balances
                continue
        rem, used = it["w"], 0
        if budget > 1:
            for j in sorted(range(len(P.load)), key=lambda j: -P.residual(i, j)):
                if used >= budget - 1 or rem <= TOL:
                    break
                r = min(P.residual(i, j), cap_i)
                if r < max(it["delta"], TOL):
                    continue
                amt = min(r, rem)
                if TOL < rem - amt < it["delta"]:
                    amt = rem - it["delta"]            # leave a legal tail
                if amt < it["delta"] - TOL or amt <= TOL:
                    continue
                P.place(i, j, amt)
                rem -= amt
                used += 1
        while rem > TOL:                               # tail into fresh bins
            amt = min(rem, cap_i)
            if TOL < rem - amt < it["delta"]:
                amt = rem - it["delta"]
            P.place(i, P.new_bin(), amt)
            rem -= amt
    return P.result()


def preprocess(inst, eta=None):
    inst["_h_common"] = apply_overhead(inst, eta)
    items, C = inst["items"], inst["capacity"]
    n = len(items)

    # effective conflicts: a pair only matters if the intervals overlap
    adj, pairs = {i: set() for i in range(n)}, []
    for i, j in inst["conflict_pairs"]:
        if items[i]["s"] < items[j]["e"] and items[j]["s"] < items[i]["e"]:
            adj[i].add(j)
            adj[j].add(i)
            pairs.append((i, j))
    inst["_conf_adj"], inst["_conf_pairs"] = adj, pairs
    inst["_n_conf_dropped"] = len(inst["conflict_pairs"]) - len(pairs)

    cliques_of = {i: [] for i in range(n)}
    for t, K in enumerate(inst["cliques"]):
        for i in K:
            cliques_of[i].append(t)
    inst["_cliques_of"] = cliques_of
    inst["_min_frags"] = {i: min_frags(items[i], C) for i in range(n)}

    # ---- lower bound, per clique, max over cliques ---------------------
    # material: every item in K occupies at least w_i + h_i * (its minimum
    # fragment count) at that time step.  conflict: a mutually conflicting
    # set inside K needs one bin per member.
    # NOTE the -FTOL: without it a load of exactly 3.0000000000000004 rounds
    # up to 4 and the LB cuts off the true optimum.
    lb_mat = lb_conf = 1
    for K in inst["cliques"]:
        dem = sum(items[i]["w"] + items[i]["h"] * inst["_min_frags"][i]
                  for i in K)
        lb_mat = max(lb_mat, math.ceil(dem / C - FTOL))
        lb_conf = max(lb_conf, greedy_clique_number(K, adj))
    inst["_lb_material"], inst["_lb_conflict"] = lb_mat, lb_conf
    inst["_lb_bins"] = max(lb_mat, lb_conf, 1)

    g1, g2 = greedy_whole_then_fresh(inst), greedy_split_fit(inst)
    inst["_greedy_whole"], inst["_greedy_splitfit"] = g1[0], g2[0]
    inst["_ub_bins"], inst["_ub_assign"] = min([g1, g2], key=lambda g: g[0])
    return inst


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def build_model(inst, B=None, quiet=True, mip_start=True, threads=0,
                free_splits=False):
    import gurobipy as gp
    from gurobipy import GRB

    C, items = inst["capacity"], inst["items"]
    n, cliques, E = len(items), inst["cliques"], inst["_conf_pairs"]
    B = B or inst["_ub_bins"]

    m = gp.Model(inst["name"])
    m.Params.OutputFlag = 0 if quiet else 1
    if threads:
        m.Params.Threads = threads

    x = m.addVars(n, B, lb=0.0, ub=1.0, name="x")     # portion of i in bin b
    u = m.addVars(n, B, vtype=GRB.BINARY, name="u")   # i has a fragment in b
    y = m.addVars(B, vtype=GRB.BINARY, name="y")      # bin b used

    m.addConstrs((x.sum(i, "*") == 1 for i in range(n)), "assign")
    m.addConstrs((items[i]["w"] * x[i, b]
                  <= min(items[i]["w"], C - items[i]["h"]) * u[i, b]
                  for i in range(n) for b in range(B)), "vub")
    m.addConstrs((items[i]["w"] * x[i, b] >= items[i]["delta"] * u[i, b]
                  for i in range(n) for b in range(B)), "minfrag")
    # Iterate over the clique INDEX t, not over the clique itself: addConstrs
    # keys its tupledict by the generator's loop variables, and a clique is a
    # sequence -> "unhashable type" as soon as it is not a tuple.
    m.addConstrs((gp.quicksum(items[i]["w"] * x[i, b] + items[i]["h"] * u[i, b]
                              for i in cliques[t]) <= C * y[b]
                  for t in range(len(cliques)) for b in range(B)), "cap")
    m.addConstrs((u[i, b] + u[j, b] <= 1
                  for (i, j) in E for b in range(B)), "conflict")
    if not free_splits:
        m.addConstrs((u.sum(i, "*") <= items[i]["k"] + 1
                      for i in range(n)), "splits")
    m.addConstrs((u[i, b] <= y[b] for i in range(n) for b in range(B)), "used")
    m.addConstrs((y[b] >= y[b + 1] for b in range(B - 1)), "order")

    # --- valid inequalities (data-derived, independent of the heuristic) ---
    m.addConstr(y.sum() >= inst["_lb_bins"], "lb_bins")
    for b in range(min(inst["_lb_bins"], B)):
        y[b].lb = 1.0                 # valid because y is ordered
    m.addConstrs((u.sum(i, "*") >= inst["_min_frags"][i]
                  for i in range(n)), "minsplits")

    m.setObjective(y.sum(), GRB.MINIMIZE)

    if mip_start:
        for b in range(B):
            y[b].Start = 0.0
        for b in sorted({b for fr in inst["_ub_assign"].values()
                         for b, _ in fr}):
            if b < B:
                y[b].Start = 1.0
        for i in range(n):
            st = dict(inst["_ub_assign"][i])
            for b in range(B):
                u[i, b].Start = 1.0 if b in st else 0.0
                x[i, b].Start = st.get(b, 0.0) / items[i]["w"]
    return m, x, u, y, B


def extract_solution(inst, x, u, B):
    items = inst["items"]
    frag = {i: [] for i in range(len(items))}
    for i in range(len(items)):
        for b in range(B):
            if u[i, b].X > 0.5 and x[i, b].X > 1e-7:
                frag[i].append((b, items[i]["w"] * x[i, b].X))
    peak = 0
    for K in inst["cliques"]:
        peak = max(peak, len({b for i in K for b, _ in frag[i]}))
    used = len({b for fr in frag.values() for b, _ in fr})
    return frag, used, peak, sum(len(v) for v in frag.values())


def validate(inst, frag, free_splits=False):
    """Independent feasibility check, straight from the data."""
    C, items = inst["capacity"], inst["items"]
    errs = []
    for i, fr in frag.items():
        it = items[i]
        if abs(sum(s for _, s in fr) - it["w"]) > TOL * max(1.0, it["w"]):
            errs.append(f"item {i}: fragment sizes do not sum to w")
        if not free_splits and len(fr) > it["k"] + 1:
            errs.append(f"item {i}: {len(fr)} fragments > k+1={it['k'] + 1}")
        if len({b for b, _ in fr}) != len(fr):
            errs.append(f"item {i}: two fragments in the same bin")
        for b, s in fr:
            if s > min(it["w"], C - it["h"]) + TOL:
                errs.append(f"item {i} bin {b}: fragment {s:.4f} > C-h")
            if len(fr) > 1 and s < it["delta"] - TOL:
                errs.append(f"item {i} bin {b}: fragment {s:.4f} < delta")
    for t, K in enumerate(inst["cliques"]):
        load = {}
        for i in K:
            for b, s in frag[i]:
                load[b] = load.get(b, 0.0) + s + items[i]["h"]
        for b, ld in load.items():
            if ld > C + TOL:
                errs.append(f"clique {t}: load {ld:.4f} > C in bin {b}")
    bins_of = {i: {b for b, _ in fr} for i, fr in frag.items()}
    for (i, j) in inst["_conf_pairs"]:
        if bins_of[i] & bins_of[j]:
            errs.append(f"conflict ({i},{j}) shares a bin")
    return errs


# ----------------------------------------------------------------------
# One instance -> one row
# ----------------------------------------------------------------------
def run_instance(path, args):
    t0 = time.time()
    inst = load_instance(path)
    meta = inst.get("meta", {})
    row = {"instance": inst["name"], "file": str(path),
           "n": inst["n_items"], "C": inst["capacity"],
           "tier": meta.get("tier", ""), "class": meta.get("class", ""),
           "seed": meta.get("generator", {}).get("seed", "")}
    try:
        preprocess(inst, eta=args.eta)
    except ValueError as exc:
        row.update({"status": "INFEASIBLE_DATA",
                    "note": str(exc).splitlines()[1].strip()
                    if "\n" in str(exc) else str(exc),
                    "runtime_s": round(time.time() - t0, 2)})
        return row

    row.update({
        "h": inst["_h_common"] if inst["_h_common"] is not None else "per-item",
        "conf_eff": len(inst["_conf_pairs"]),
        "conf_dropped": inst["_n_conf_dropped"],
        "n_cliques": len(inst["cliques"]),
        "max_clique": max(len(K) for K in inst["cliques"]),
        "lb_bins": inst["_lb_bins"],
        "lb_material": inst["_lb_material"],
        "lb_conflict": inst["_lb_conflict"],
        "greedy_whole": inst["_greedy_whole"],
        "greedy_splitfit": inst["_greedy_splitfit"],
        "greedy": inst["_ub_bins"],
    })

    import gurobipy as gp
    from gurobipy import GRB

    B = args.bins or inst["_ub_bins"]
    m, x, u, y, B = build_model(inst, B=B, quiet=not args.verbose,
                                mip_start=not args.no_warmstart,
                                threads=args.threads,
                                free_splits=args.free_splits)
    if args.timelimit:
        m.Params.TimeLimit = args.timelimit
    m.Params.MIPGap = args.mipgap
    # the objective counts bins, so it is integral: a remaining absolute gap
    # below 1 already proves optimality
    m.Params.MIPGapAbs = args.mipgapabs
    m.Params.Seed = args.seed
    if args.symmetry is not None:
        m.Params.Symmetry = args.symmetry
    if args.mipfocus is not None:
        m.Params.MIPFocus = args.mipfocus
    m.optimize()

    row.update({
        "B_slots": B, "warm_start": not args.no_warmstart,
        "free_splits": args.free_splits,
        "status": {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
                   GRB.TIME_LIMIT: "TIME_LIMIT",
                   GRB.INTERRUPTED: "INTERRUPTED"}.get(m.Status, str(m.Status)),
        "runtime_s": round(m.Runtime, 2), "nodes": int(m.NodeCount),
    })
    try:
        row["root_lp"] = round(m.ObjBoundC, 4)
    except Exception:
        row["root_lp"] = ""

    # ObjBound / MIPGap are +-inf when the run stops before any bound exists
    # (very short limits, huge models). Never let that reach round()/ceil().
    def finite(v, nd=4):
        return round(v, nd) if v is not None and abs(v) != float("inf") else ""

    if m.SolCount == 0:
        # no incumbent from Gurobi: the greedy solution is still the best
        # known feasible answer, so report it as such rather than as a blank
        row.update({"incumbent": "",
                    "bound": "" if m.Status == GRB.INFEASIBLE
                    else finite(m.ObjBound),
                    "best_known": inst["_ub_bins"], "source": "greedy",
                    "note": "solver found no incumbent"})
        m.dispose()
        return row

    frag, used, peak, nfrag = extract_solution(inst, x, u, B)
    errs = validate(inst, frag, free_splits=args.free_splits)
    inc, bnd = m.ObjVal, m.ObjBound
    safe_bnd = max(bnd, inst["_lb_bins"]) if abs(bnd) != float("inf") \
        else inst["_lb_bins"]              # fall back on the preprocessing LB
    row.update({
        "incumbent": int(round(inc)),          # best feasible bin count found
        "bound": finite(bnd),                  # proven lower bound
        "gap_pct": finite(100 * m.MIPGap, 2),
        "gap_abs": int(round(inc)) - math.ceil(safe_bnd - FTOL),
        "proven_optimal": bool(m.Status == GRB.OPTIMAL),
        "best_known": int(round(inc)), "source": "mip",
        "bins_used": used, "peak_bins": peak,
        "n_fragments": nfrag,
        "split_items": sum(1 for fr in frag.values() if len(fr) > 1),
        "vs_greedy": inst["_ub_bins"] - int(round(inc)),
        "valid": "PASS" if not errs else "FAIL",
        "n_violations": len(errs),
    })
    if args.sol:
        Path(args.sol).parent.mkdir(parents=True, exist_ok=True)
        Path(args.sol).write_text(json.dumps(
            {**{k: v for k, v in row.items()},
             "fragments": {str(i): [(b, round(s, 6)) for b, s in fr]
                           for i, fr in frag.items()},
             "violations": errs}, indent=1, default=str))
    m.dispose()
    return row


def fmt(row):
    """Exactly one line per instance."""
    name = row["instance"][:26]
    if "lb_bins" not in row:          # never reached the solver
        return (f"{name:26s} n={str(row.get('n', '?')):>4}  "
                f"{row.get('status', 'ERROR'):16s} {row.get('note', '')[:70]}")
    head = (f"{name:26s} n={row['n']:>4} C={row['C']:g} h={row['h']}  "
            f"LB {row['lb_bins']:>3}  greedy {row['greedy']:>3}  | ")
    if row.get("incumbent") == "":
        return head + (f"{row['status']:10s} no incumbent  "
                       f"bound {row.get('bound') or 'n/a'}  "
                       f"best known {row['best_known']} (greedy)  "
                       f"{row['runtime_s']}s")
    return head + (f"{row['status']:10s} "
                   f"inc {row['incumbent']:>3}  "
                   f"bound {str(row['bound'] if row['bound'] != '' else 'n/a'):>6}  "
                   f"gap {row['gap_abs']:>2}  "
                   f"vs.greedy {row['vs_greedy']:>+3}  "
                   f"peak {row['peak_bins']:>3}  frags {row['n_fragments']:>4}  "
                   f"split {row['split_items']:>3}  "
                   f"{row['runtime_s']:>7.1f}s  [{row['valid']}]")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def expand(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "**", "*.json"),
                                    recursive=True))
        else:
            g = sorted(glob.glob(p))
            out += g if g else [p]
    return [p for p in out if not p.endswith("_sol.json")]


def main():
    ap = argparse.ArgumentParser(
        description="Exact MIP for TBPP-C-LF: min number of distinct bins.")
    ap.add_argument("instances", nargs="+",
                    help="files, globs or a directory of instance JSONs")
    ap.add_argument("--timelimit", type=float, default=300,
                    help="seconds per instance (0 = no limit)")
    ap.add_argument("--csv", default=None, help="append one row per instance")
    ap.add_argument("--sol", default=None,
                    help="write the fragment layout of a single instance here")
    ap.add_argument("--bins", type=int, default=None,
                    help="override the number of bin slots B")
    ap.add_argument("--eta", type=float, default=None,
                    help="override every h_i with eta * C (overhead sweep); "
                         "default keeps the h stored in the instance")
    ap.add_argument("--no-warmstart", action="store_true",
                    help="do NOT seed the greedy solution as a MIP start. Use "
                         "this whenever the run is a greedy-vs-solver "
                         "comparison, otherwise the incumbent is <= greedy by "
                         "construction and the comparison is circular.")
    ap.add_argument("--free-splits", action="store_true",
                    help="reference variant: drop the k_i split budget "
                         "(unlimited fragmentation). Not the default model.")
    ap.add_argument("--mipgap", type=float, default=1e-6)
    ap.add_argument("--mipgapabs", type=float, default=0.999)
    ap.add_argument("--symmetry", type=int, default=2,
                    help="Gurobi Symmetry parameter (2 = aggressive)")
    ap.add_argument("--mipfocus", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1,
                    help="fixed for reproducible timings; 0 = all cores")
    ap.add_argument("--verbose", action="store_true", help="show Gurobi's log")
    args = ap.parse_args()

    files = expand(args.instances)
    rows, t0 = [], time.time()
    for p in files:
        try:
            row = run_instance(p, args)
        except Exception as exc:
            row = {"instance": Path(p).stem, "file": str(p),
                   "status": "ERROR", "note": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        print(fmt(row), flush=True)

    if args.csv:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        p = Path(args.csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        new = not p.exists()
        with p.open("a", newline="") as f:
            w = csv.DictWriter(f, fields, extrasaction="ignore")
            if new:
                w.writeheader()
            for r in rows:
                w.writerow(r)

    solved = sum(1 for r in rows if r.get("proven_optimal"))
    print(f"-- {len(rows)} instance(s), {solved} proven optimal, "
          f"{time.time() - t0:.1f}s total", file=sys.stderr)


if __name__ == "__main__":
    main()