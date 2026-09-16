#!/usr/bin/env python3
"""
run_experiment.py -- self-contained runner for TBPP-C-LF instances (format v4.0)
Temporal Bin Packing with Conflicts and Limited Fragmentation.

Dependencies: numpy (required), scipy (only if --mclb is used).
Keep mclb.py in the same folder to enable the tight MC-LB bound.

  python srff.py                          # ./instances, else ./
  python srff.py data/                    # directory (recursive)
  python srff.py 'data/n50_d0.4_*.json'   # glob
  python srff.py data/ --tune --csv out.csv
  python srff.py data/ --mclb --mclb-events 3   # tight gap

Reads any JSON with the schema
  {capacity, n_items, items:[{id,s,e,w,k,h,delta}], conflicts:[[i,j],...], meta}
Runs the SRPH chain heuristic + FFD(alpha) baseline, verifies every solution
independently, prints a per-instance table and an aggregate summary.
Run --help for all flags.
"""
import argparse
import csv as csvmod
import glob
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

INF = float('inf')

# Optional tight lower bound (conflict + capacity aware). Requires mclb.py + scipy.
try:
    import mclb as _mclb
    _HAVE_MCLB = True
except Exception:
    _HAVE_MCLB = False


@dataclass
class Instance:
    name: str
    n: int
    C: float
    w: np.ndarray
    s: np.ndarray
    e: np.ndarray
    k: np.ndarray
    delta: np.ndarray
    h: np.ndarray
    conf: list                                    # raw conflict neighbour sets
    meta: dict = field(default_factory=dict)
    cliques: list = field(default_factory=list)   # maximal cliques, interval graph
    Cof: list = field(default_factory=list)       # Cof[i] = clique indices of i
    Ebar: list = field(default_factory=list)      # overlap-gated conflicts
    m: np.ndarray = None                          # min fragments ceil(w/(C-h))
    f: np.ndarray = None                          # max fragments min(k+1, w/delta)
    sigma: np.ndarray = None                      # split slack f - m

    # ---------------------------------------------------------------- build
    def build(self):
        n = self.n
        sets = []
        for t in sorted(set(self.s.tolist())):
            act = frozenset(j for j in range(n) if self.s[j] <= t < self.e[j])
            if act:
                sets.append(act)
        maximal = []
        for A in sets:
            if not any(A < B for B in sets) and A not in maximal:
                maximal.append(A)
        self.cliques = maximal
        self.Cof = [set() for _ in range(n)]
        for q, K in enumerate(self.cliques):
            for i in K:
                self.Cof[i].add(q)

        self.Ebar = [set() for _ in range(n)]
        for i in range(n):
            for j in self.conf[i]:
                if self.s[i] < self.e[j] and self.s[j] < self.e[i]:
                    self.Ebar[i].add(j)

        cap = self.C - self.h
        self.m = np.ceil(self.w / cap - 1e-9).astype(int)
        self.f = np.minimum(self.k + 1,
                            np.floor(self.w / self.delta + 1e-9)).astype(int)
        self.sigma = self.f - self.m
        return self

    # ------------------------------------------------------------ integrity
    def feasibility_check(self):
        """spec standing assumption: m_i <= f_i and delta_i <= C - h_i"""
        return [int(i) for i in range(self.n)
                if self.m[i] > self.f[i] or self.delta[i] > self.C - self.h[i]]

    def diagnose(self):
        """report which modelling terms are actually alive on this instance."""
        raw = sum(len(c) for c in self.conf) // 2
        eff = sum(len(c) for c in self.Ebar) // 2
        disjoint_conf = sum(1 for i in range(self.n) for j in self.conf[i]
                            if j > i and not (self.Cof[i] & self.Cof[j]))
        return dict(
            name=self.name, n=self.n, C=self.C,
            n_cliques=len(self.cliques),
            max_clique=max(len(K) for K in self.cliques),
            avg_clique=round(float(np.mean([len(K) for K in self.cliques])), 2),
            raw_conflicts=raw, eff_conflicts=eff,
            conflicting_but_disjoint=disjoint_conf,      # 0 -> free-lunch dead
            forced_split_items=int((self.m > 1).sum()),  # 0 -> m_i == 1
            atomic_items=int((self.f == 1).sum()),       # k_i = 0
            mean_breadth=round(float(np.mean([len(self.Cof[i])
                                              for i in range(self.n)])), 2),
            infeasible_items=self.feasibility_check(),
            lb_clique=self.clique_lower_bound(),
        )

    # ---------------------------------------------------------------- bound
    def clique_lower_bound(self):
        best = 0
        for K in self.cliques:
            load = sum(self.w[i] + self.h[i] * self.m[i] for i in K)
            best = max(best, int(np.ceil(load / self.C - 1e-9)))
        return max(best, int(self.m.max()))


# -------------------------------------------------------------------- I/O
def load_instance(path):
    with open(path) as fh:
        d = json.load(fh)
    items = sorted(d['items'], key=lambda it: it['id'])
    n = int(d.get('n_items', len(items)))
    idx = {it['id']: p for p, it in enumerate(items)}

    def col(key, dtype=float):
        return np.array([it[key] for it in items], dtype=dtype)

    conf = [set() for _ in range(n)]
    for a, b in d.get('conflicts', []):
        pa, pb = idx[a], idx[b]
        conf[pa].add(pb)
        conf[pb].add(pa)

    return Instance(
        name=d.get('name', os.path.basename(path)),
        n=n, C=float(d['capacity']),
        w=col('w'), s=col('s'), e=col('e'),
        k=col('k', int), delta=col('delta'), h=col('h'),
        conf=conf, meta=d.get('meta', {}),
    ).build()


def load_folder(pattern):
    return [load_instance(p) for p in sorted(glob.glob(pattern))]


# ---------------------------------------------- MC-LB adapter (tight bound)
def _mclb_items(inst):
    """Build the item-object list mclb.py expects from the array-based Instance.
    Position index == item id here (conf is stored in position space)."""
    class _It:
        __slots__ = ('id', 'w', 's', 'e', 'k', 'h', 'delta', 'conflicts')
    out = []
    for p in range(inst.n):
        it = _It()
        it.id = p
        it.w = float(inst.w[p]); it.s = float(inst.s[p]); it.e = float(inst.e[p])
        it.k = int(inst.k[p]); it.h = float(inst.h[p]); it.delta = float(inst.delta[p])
        it.conflicts = set(inst.conf[p])
        out.append(it)
    return out


def mclb_bound(inst, events=1, exact=False, milp_time=30.0):
    """Tight conflict+capacity lower bound; None if mclb.py is unavailable."""
    if not _HAVE_MCLB:
        return None
    val, _ = _mclb.mclb_value(_mclb_items(inst), inst.C, n_events=events,
                              greedy=not exact, milp_time=milp_time)
    return val


# --------------------------------------------------- rigidity index R_i
def rigidity(inst, alpha=(0.25, 0.35, 0.20, 0.20)):
    """Temporal generalization of Muritiba's surrogate weight ws_i.

    Reduces to a1*w_i/wbar + a2*d_i/dbar when h=0, k=0 and intervals coincide.
    On this dataset m_i == 1, so phi_i = (w_i + h_i)/C and sigma_i == k_i.
    """
    a1, a2, a3, a4 = alpha
    n, C = inst.n, inst.C
    phi = (inst.w + inst.m)**2 / C
    dbar = np.array([sum(phi[j] for j in inst.Ebar[i]) for i in range(n)])
    brd = np.array([len(inst.Cof[i]) for i in range(n)], float)
    sig = inst.sigma.astype(float)

    def nz(v):
        mu = float(v.mean())
        return v / mu if abs(mu) > 1e-12 else np.zeros_like(v)

    return a1 * nz(phi) + a2 * nz(dbar) + a3 * nz(brd) - a4 * nz(sig), phi, dbar


# --------------------------------------------- pairwise compatibility gamma
def build_gamma(inst, phi, mu_block=0.15, free_lunch=1.4):
    """gamma[i][j]: -inf = hard block (conflict during a shared clique).

    NOTE on this dataset: (conflict AND temporally disjoint) is impossible by
    construction, since conflicts were generated over overlapping pairs only.
    The branch is kept for generality and its firing count is recorded on
    build_gamma.free_lunch_hits so experiments can report that it never fires.
    """
    n, C = inst.n, inst.C
    G = np.zeros((n, n))
    build_gamma.free_lunch_hits = 0
    for i in range(n):
        wi, hi = inst.w[i], inst.h[i]
        for j in range(n):
            if i == j:
                continue
            if j in inst.Ebar[i]:
                G[i, j] = -INF
            elif inst.Cof[i] & inst.Cof[j]:
                tot = (wi + hi + inst.w[j] + inst.h[j]) / C
                G[i, j] = 1.0 - abs(1.0 - tot) if tot <= 1.0 else -0.3 * (tot - 1.0)
            elif j in inst.conf[i]:
                G[i, j] = free_lunch
                build_gamma.free_lunch_hits += 1
            else:
                G[i, j] = 1.0                            # idle-time bin reuse
    beta = np.array([sum(phi[u] for u in inst.Ebar[j]) for j in range(n)])
    if beta.mean() > 0:
        beta = beta / beta.mean()
    return G - mu_block * beta[None, :], beta


class Bin:
    __slots__ = ('r', 'block', 'occ', 'order')

    def __init__(self, Q, C):
        self.r = np.full(Q, C)      # residual per clique
        self.block = [set() for _ in range(Q)]
        self.occ = set()            # items with a fragment here
        self.order = []             # insertion order (chain anchor)


def fits(inst, b, i, amount):
    """can a fragment of size `amount` of item i go into bin b?"""
    if i in b.occ:
        return False
    need = amount + inst.h[i]
    for q in inst.Cof[i]:
        if i in b.block[q] or b.r[q] < need - 1e-9:
            return False
    return True


def capacity_for(inst, b, i):
    """largest fragment of i that b can take (before budget checks)."""
    if i in b.occ:
        return 0.0
    cap = INF
    for q in inst.Cof[i]:
        if i in b.block[q]:
            return 0.0
        cap = min(cap, b.r[q])
    return max(0.0, cap - inst.h[i])


def place(inst, b, i, amount):
    need = amount + inst.h[i]
    for q in inst.Cof[i]:
        b.r[q] -= need
        b.block[q].update(inst.Ebar[i])
    b.occ.add(i)
    b.order.append(i)


def anchor_of(inst, b, j, mode):
    """mode 'last' = last inserted; 'overlap' = occupant sharing most cliques."""
    if not b.order:
        return None
    if mode == 'last':
        return b.order[-1]
    best, bv = b.order[-1], -1
    for i in b.order:
        v = len(inst.Cof[i] & inst.Cof[j])
        if v > bv:
            best, bv = i, v
    return best


def gamma_eff(inst, G, b, j, theta, anchor_mode):
    """theta=1 : chain (anchor only).  theta=0 : consensus (min over occupants)."""
    if not b.occ:
        return 0.0
    a = anchor_of(inst, b, j, anchor_mode)
    g_chain = G[a, j]
    if theta >= 1.0 - 1e-9:
        return g_chain
    g_cons = min(G[i, j] for i in b.occ)
    if theta <= 1e-9:
        return g_cons
    if g_chain == -INF or g_cons == -INF:
        return -INF
    return theta * g_chain + (1 - theta) * g_cons


def fit_score(inst, b, j, amount):
    """best-fit measured across j's own cliques (tight = good)."""
    tot = sum(b.r[q] for q in inst.Cof[j])
    if tot <= 1e-9:
        return 0.0
    return (amount + inst.h[j]) * len(inst.Cof[j]) / tot


def verify(inst, bins, frags, placed_amounts):
    """independent check: capacity per clique, conflicts, budget, min frag."""
    errs = []
    for bi, b in enumerate(bins):
        for q, K in enumerate(inst.cliques):
            load = sum(placed_amounts[(bi, i)] + inst.h[i]
                       for i in b.occ if q in inst.Cof[i])
            if load > inst.C + 1e-6:
                errs.append(f"capacity bin{bi} clique{q}: {load:.2f}>{inst.C}")
            act = [i for i in b.occ if q in inst.Cof[i]]
            for x in act:
                for y in act:
                    if x != y and y in inst.Ebar[x]:
                        errs.append(f"conflict {x},{y} bin{bi} clique{q}")
    for i in range(inst.n):
        tot = sum(v for (bi, j), v in placed_amounts.items() if j == i)
        if abs(tot - inst.w[i]) > 1e-5:
            errs.append(f"size item{i}: {tot:.3f} vs {inst.w[i]:.3f}")
        if frags[i] > inst.f[i]:
            errs.append(f"budget item{i}: {frags[i]}>{inst.f[i]}")
        if frags[i] > 1:
            for (bi, j), v in placed_amounts.items():
                if j == i and v < inst.delta[i] - 1e-6:
                    errs.append(f"minfrag item{i}: {v:.3f}<{inst.delta[i]:.3f}")
    return errs


def pack(inst, R, G, theta=1.0, lam=0.5, anchor_mode='last',
         seed_split='fill', lookahead=False, Bmax=None):
    """Sequential Rigidity Packing Heuristic.

    Returns (#bins used, #fragments total, feasible flag).
    """
    Q = len(inst.cliques)
    Bmax = Bmax or (inst.n * int(inst.m.max()) + 5)
    bins = []
    rem = inst.w.astype(float).copy()
    frags = np.zeros(inst.n, int)
    unplaced = set(range(inst.n))

    def new_bin():
        b = Bin(Q, inst.C)
        bins.append(b)
        return b

    def can_take_more_frags(i):
        return frags[i] < inst.f[i]

    amounts = {}

    def put(b, i, amount):
        if b not in bins:
            bins.append(b)
        amounts[(bins.index(b), i)] = amounts.get((bins.index(b), i), 0.0) + amount
        place(inst, b, i, amount)
        rem[i] -= amount
        frags[i] += 1
        if rem[i] <= 1e-7:
            rem[i] = 0.0
            unplaced.discard(i)

    def feasible_amount(b, i):
        """largest amount of i placeable in b respecting delta & budget."""
        cap = capacity_for(inst, b, i)
        if cap < inst.delta[i] - 1e-9 and cap < rem[i] - 1e-9:
            return 0.0
        if not can_take_more_frags(i):
            return 0.0
        amt = min(rem[i], cap)
        if amt <= 1e-9:
            return 0.0
        if amt < rem[i] - 1e-9:
            # partial: the remainder must still be splittable in the budget left
            budget_left = inst.f[i] - frags[i] - 1
            if budget_left <= 0:
                return 0.0
            max_rest = budget_left * (inst.C - inst.h[i])
            need_amt = rem[i] - max_rest          # lower bound on this fragment
            if need_amt > cap + 1e-9:
                return 0.0
            amt = max(amt, need_amt)
            leftover = rem[i] - amt
            if 1e-9 < leftover < inst.delta[i] - 1e-9:
                amt = rem[i] - inst.delta[i]
            if amt < inst.delta[i] - 1e-9 or amt > cap + 1e-9:
                return 0.0
        return amt

    guard = 0
    while unplaced and guard < 100000:
        guard += 1
        # ---------- pick the globally hardest remaining item as seed --------
        seed = max(unplaced, key=lambda i: R[i])

        # ---------- open bins for the seed ---------------------------------
        opened = []
        while rem[seed] > 1e-7:
            # prefer an existing bin that can host a fragment of the seed
            cands = [b for b in bins if feasible_amount(b, seed) > 0]
            if cands:
                b = max(cands, key=lambda b: capacity_for(inst, b, seed))
            else:
                if len(bins) >= Bmax:
                    return None, None, False
                b = new_bin()
            amt = feasible_amount(b, seed)
            if amt <= 0:
                if len(bins) >= Bmax:
                    return None, None, False
                b = new_bin()
                amt = feasible_amount(b, seed)
                if amt <= 0:
                    return None, None, False
            if seed_split == 'even' and rem[seed] > inst.C:
                need = int(np.ceil(rem[seed] / (inst.C - inst.h[seed])))
                amt = min(amt, rem[seed] / max(need, 1))
                if amt < inst.delta[seed]:
                    amt = feasible_amount(b, seed)
            put(b, seed, amt)
            opened.append(b)

        # ---------- fill each opened bin by chaining ------------------------
        for b in opened:
            while True:
                best, bestv, bestamt = None, -INF, 0.0
                for j in unplaced:
                    amt = feasible_amount(b, j)
                    if amt <= 0:
                        continue
                    g = gamma_eff(inst, G, b, j, theta, anchor_mode)
                    if g == -INF:
                        continue
                    v = lam * R[j] + (1 - lam) * (g + fit_score(inst, b, j, amt))
                    if lookahead:
                        # one-step: best friend available after j
                        nxt = -INF
                        for u in unplaced:
                            if u == j:
                                continue
                            if G[j, u] > nxt and capacity_for(inst, b, u) > 0:
                                nxt = G[j, u]
                        if nxt > -INF:
                            v += 0.3 * nxt
                    if v > bestv:
                        best, bestv, bestamt = j, v, amt
                if best is None:
                    break
                put(b, best, bestamt)

    # ---------- recycling pass for anything left (safety) -------------------
    guard = 0
    while unplaced and guard < 100000:
        guard += 1
        i = min(unplaced, key=lambda i: R[i])
        cands = [b for b in bins if feasible_amount(b, i) > 0]
        if cands:
            b = max(cands, key=lambda b: feasible_amount(b, i))
        else:
            if len(bins) >= Bmax:
                return None, None, False
            b = new_bin()
        amt = feasible_amount(b, i)
        if amt <= 0:
            return None, None, False
        put(b, i, amt)

    used = sum(1 for b in bins if b.occ)
    pack.last_bins = bins
    pack.last_frags = frags
    pack.last_amounts = amounts
    return used, int(frags.sum()), True


# ----------------------------------------------------------------------
# Baseline: Muritiba-style surrogate-weight First-Fit Decreasing, temporalized
# ----------------------------------------------------------------------
def ffd_surrogate(inst, alpha):
    """ws_i = a*w_i/wbar + (1-a)*d_i/dbar ; first-fit with fragmentation."""
    n = inst.n
    d = np.array([len(inst.Ebar[i]) for i in range(n)], float)
    ws = alpha * inst.w / inst.w.mean() + (1 - alpha) * (
        d / d.mean() if d.mean() > 0 else d)
    order = np.argsort(-ws)
    Q = len(inst.cliques)
    bins, rem, frags = [], inst.w.astype(float).copy(), np.zeros(n, int)

    for i in order:
        while rem[i] > 1e-7:
            placed = False
            for b in bins:
                cap = capacity_for(inst, b, i)
                if cap <= 1e-9 or frags[i] >= inst.f[i]:
                    continue
                amt = min(rem[i], cap)
                if amt < rem[i] - 1e-9:
                    bl = inst.f[i] - frags[i] - 1
                    if bl <= 0:
                        continue
                    need = rem[i] - bl * (inst.C - inst.h[i])
                    if need > cap + 1e-9:
                        continue
                    amt = max(amt, need)
                    lo = rem[i] - amt
                    if 1e-9 < lo < inst.delta[i]:
                        amt = rem[i] - inst.delta[i]
                if amt < min(inst.delta[i], rem[i]) - 1e-9 or amt > cap + 1e-9:
                    continue
                place(inst, b, i, amt)
                rem[i] -= amt; frags[i] += 1
                placed = True
                break
            if not placed:
                b = Bin(Q, inst.C)
                bins.append(b)
                cap = capacity_for(inst, b, i)
                amt = min(rem[i], cap)
                if amt < rem[i] - 1e-9:
                    bl = inst.f[i] - frags[i] - 1
                    need = rem[i] - max(bl, 0) * (inst.C - inst.h[i])
                    amt = max(amt, min(need, cap))
                    lo = rem[i] - amt
                    if 1e-9 < lo < inst.delta[i]:
                        amt = rem[i] - inst.delta[i]
                if amt <= 1e-9:
                    return None, None
                place(inst, b, i, amt)
                rem[i] -= amt; frags[i] += 1
            if rem[i] <= 1e-7:
                rem[i] = 0.0
    return sum(1 for b in bins if b.occ), int(frags.sum())


def ffd_best_alpha(inst):
    best = (10**9, None)
    for a in np.linspace(0, 1, 11):
        r, fr = ffd_surrogate(inst, a)
        if r is not None and r < best[0]:
            best = (r, fr)
    return best


DEFAULT_ALPHAS = [(0.25, 0.35, 0.20, 0.20), (1, 0, 0, 0), (0, 1, 0, 0),
                  (0, 0, 1, 0), (0, 0, 0, 1), (0.5, 0.5, 0, 0),
                  (0.3, 0.3, 0.3, 0.1), (0.2, 0.5, 0.1, 0.2),
                  (0.4, 0.2, 0.2, 0.2), (0, 0.6, 0.2, 0.2),
                  (0.34, 0.33, 0.33, 0.0)]
DEFAULT_LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_THETAS = [0.0, 0.5, 1.0]


# ------------------------------------------------------------------ discovery
def looks_like_instance(path):
    try:
        with open(path) as fh:
            head = fh.read(400)
        if '"items"' not in head or '"capacity"' not in head:
            return False
        with open(path) as fh:
            d = json.load(fh)
        return isinstance(d.get('items'), list) and 'capacity' in d
    except Exception:
        return False


def discover(paths, recursive=True):
    found = []
    for p in paths:
        if os.path.isdir(p):
            pat = os.path.join(p, '**', '*.json') if recursive \
                else os.path.join(p, '*.json')
            found += glob.glob(pat, recursive=recursive)
        elif any(ch in p for ch in '*?['):
            found += glob.glob(p, recursive=recursive)
        elif os.path.isfile(p):
            found.append(p)
    seen, out = set(), []
    for f in sorted(found):
        rp = os.path.realpath(f)
        if rp not in seen and looks_like_instance(f):
            seen.add(rp)
            out.append(f)
    return out


# ------------------------------------------------------------------ parsing
def parse_floats(s):
    return [float(x) for x in str(s).replace(' ', '').split(',') if x != '']


def parse_alpha(spec):
    v = parse_floats(spec)
    if len(v) != 4:
        raise argparse.ArgumentTypeError(
            f"--alpha needs 4 comma-separated weights (a1,a2,a3,a4); got {spec!r}")
    return tuple(v)


# ------------------------------------------------------------------ one run
def run_instance(inst, alphas, lams, thetas, args):
    dg = inst.diagnose()
    lb_mc = (mclb_bound(inst, events=args.mclb_events, exact=args.mclb_exact,
                        milp_time=args.mclb_time)
             if getattr(args, 'mclb', False) else None)
    lb_best = dg['lb_clique'] if lb_mc is None else max(dg['lb_clique'], lb_mc)
    t0 = time.perf_counter()

    base = base_fr = None
    if not args.no_baseline:
        if args.ffd_alpha is None:
            base, base_fr = ffd_best_alpha(inst)
        else:
            base, base_fr = ffd_surrogate(inst, args.ffd_alpha)
    t_base = time.perf_counter() - t0

    t1 = time.perf_counter()
    best = dict(bins=None, frags=None, alpha=None, lam=None, theta=None)
    per_theta = {t: None for t in thetas}
    per_lam = {l: None for l in lams}
    runs = fails = viols = 0

    for a in alphas:
        R, phi, _ = rigidity(inst, alpha=a)
        G, _ = build_gamma(inst, phi, mu_block=args.mu_block,
                           free_lunch=args.free_lunch)
        for lam, th in itertools.product(lams, thetas):
            runs += 1
            u, fr, ok = pack(inst, R, G, theta=th, lam=lam,
                             anchor_mode=args.anchor, seed_split=args.seed_split,
                             lookahead=args.lookahead)
            if not ok:
                fails += 1
                continue
            if not args.no_verify:
                errs = verify(inst, pack.last_bins, pack.last_frags,
                              pack.last_amounts)
                if errs:
                    viols += 1
                    if args.strict:
                        raise RuntimeError(f"{inst.name}: {errs[:3]}")
                    continue
            if best['bins'] is None or u < best['bins']:
                best = dict(bins=u, frags=fr, alpha=a, lam=lam, theta=th)
            for key, val in ((per_theta, th), (per_lam, lam)):
                cur = key[val]
                key[val] = u if cur is None else min(cur, u)
    t_srph = time.perf_counter() - t1

    return dict(
        name=inst.name, n=inst.n, C=inst.C,
        d_target=inst.meta.get('params', {}).get('d_target'),
        eta=inst.meta.get('params', {}).get('eta'),
        lb=dg['lb_clique'], lb_mc=lb_mc, lb_best=lb_best,
        n_cliques=dg['n_cliques'],
        max_clique=dg['max_clique'], mean_breadth=dg['mean_breadth'],
        eff_conflicts=dg['eff_conflicts'], atomic_items=dg['atomic_items'],
        forced_split_items=dg['forced_split_items'],
        conflicting_but_disjoint=dg['conflicting_but_disjoint'],
        infeasible_items=len(dg['infeasible_items']),
        ffd_bins=base, ffd_splits=(None if base_fr is None else base_fr - inst.n),
        srph_bins=best['bins'],
        srph_splits=(None if best['frags'] is None else best['frags'] - inst.n),
        alpha=str(best['alpha']), lam=best['lam'], theta=best['theta'],
        runs=runs, failed=fails, violations=viols,
        t_baseline=round(t_base, 3), t_srph=round(t_srph, 3),
        per_theta={str(k): v for k, v in per_theta.items()},
        per_lam={str(k): v for k, v in per_lam.items()},
    )


# ------------------------------------------------------------------ reporting
def report(rows, args):
    ok = [r for r in rows if r['srph_bins'] is not None]
    if not ok:
        print("no solved instances", file=sys.stderr)
        return

    w = max(len(r['name']) for r in ok)
    hdr = (f"{'instance':<{w}} {'n':>4} {'LB':>4} {'SRPH':>5} {'spl':>5}")
    if not args.no_baseline:
        hdr += f" {'FFD':>5} {'spl':>5} {'d':>3}"
    hdr += f" {'gap%':>7} {'sec':>7}"
    print(hdr)
    print('-' * len(hdr))
    for r in ok:
        denom = r.get('lb_best') or r['lb']
        gap = (r['srph_bins'] - denom) / denom * 100 if denom else 0.0
        line = (f"{r['name']:<{w}} {r['n']:>4} {r['lb']:>4} "
                f"{r['srph_bins']:>5} {r['srph_splits']:>5}")
        if not args.no_baseline:
            dlt = r['ffd_bins'] - r['srph_bins']
            line += (f" {r['ffd_bins']:>5} {r['ffd_splits']:>5} "
                     f"{('+' if dlt > 0 else ''):>0}{dlt:>2}")
        line += f" {gap:>7.2f} {r['t_srph']:>7.2f}"
        print(line)

    print('\n=== summary ===')
    print(f"instances solved      {len(ok)}/{len(rows)}")
    print(f"mean gap over LB      {np.mean([(r['srph_bins']-r['lb'])/r['lb'] for r in ok])*100:.2f}%")
    mc = [r for r in ok if r.get('lb_mc')]
    if mc:
        print(f"mean gap over MC-LB   "
              f"{np.mean([(r['srph_bins']-r['lb_best'])/r['lb_best'] for r in mc])*100:.2f}%"
              f"   ({len(mc)} with valid MC-LB)")
        strict = sum(1 for r in mc if r['lb_mc'] > r['lb'])
        print(f"MC-LB > clique LB on  {strict}/{len(mc)} instances")
    print(f"mean splits (SRPH)    {np.mean([r['srph_splits'] for r in ok]):.1f}")
    if not args.no_baseline:
        W = sum(1 for r in ok if r['srph_bins'] < r['ffd_bins'])
        T = sum(1 for r in ok if r['srph_bins'] == r['ffd_bins'])
        L = sum(1 for r in ok if r['srph_bins'] > r['ffd_bins'])
        print(f"mean gap over LB (FFD){np.mean([(r['ffd_bins']-r['lb'])/r['lb'] for r in ok])*100:>7.2f}%")
        print(f"mean splits (FFD)     {np.mean([r['ffd_splits'] for r in ok]):.1f}")
        print(f"SRPH vs FFD(alpha)    {W} wins / {T} ties / {L} losses")
    tot_v = sum(r['violations'] for r in rows)
    tot_f = sum(r['failed'] for r in rows)
    print(f"verifier violations   {tot_v}"
          f"{'  <-- INVESTIGATE' if tot_v else ''}")
    print(f"infeasible packings   {tot_f}")

    dead = [k for k in ('forced_split_items', 'conflicting_but_disjoint')
            if all(r[k] == 0 for r in ok)]
    if dead:
        print(f"inert model terms     {', '.join(dead)} == 0 on every instance")

    if len(ok) > 1 and (len(args._thetas) > 1 or len(args._lams) > 1):
        print('\n=== reaches per-instance best ===')
        for label, key, vals in (('theta', 'per_theta', args._thetas),
                                 ('lambda', 'per_lam', args._lams)):
            if len(vals) < 2:
                continue
            cnt = {}
            for r in ok:
                m = min(v for v in r[key].values() if v is not None)
                for k, v in r[key].items():
                    if v == m:
                        cnt[k] = cnt.get(k, 0) + 1
            print(f"  {label:<7}" + "  ".join(
                f"{k}:{cnt.get(k,0)}/{len(ok)}" for k in sorted(cnt)))


# ------------------------------------------------------------------ main
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Batch runner for TBPP-C-LF instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('paths', nargs='*', default=None,
                   help="files, directories, or globs (default: ./instances, else ./)")
    p.add_argument('--no-recursive', action='store_true',
                   help="do not descend into subdirectories")

    g = p.add_argument_group('heuristic parameters')
    g.add_argument('--alpha', type=parse_alpha, action='append', metavar='a1,a2,a3,a4',
                   help="rigidity weights (size, degree, breadth, -slack); repeatable")
    g.add_argument('--lam', type=parse_floats, metavar='L[,L...]',
                   help="weight of global rigidity in the fill step")
    g.add_argument('--theta', type=parse_floats, metavar='T[,T...]',
                   help="1 = chain (anchor only), 0 = consensus (min over occupants)")
    g.add_argument('--tune', action='store_true',
                   help="sweep the full alpha x lam x theta grid")
    g.add_argument('--anchor', choices=('last', 'overlap'), default='last',
                   help="chain anchor: last inserted, or max clique overlap")
    g.add_argument('--seed-split', choices=('fill', 'even'), default='fill',
                   help="how a seed needing several bins is divided")
    g.add_argument('--lookahead', action='store_true',
                   help="one-step lookahead on the next chain link")
    g.add_argument('--mu-block', type=float, default=0.15,
                   help="collateral-blocking penalty weight in gamma")
    g.add_argument('--free-lunch', type=float, default=1.4,
                   help="gamma bonus for conflicting but temporally disjoint pairs")

    b = p.add_argument_group('baseline / validation')
    b.add_argument('--no-baseline', action='store_true', help="skip FFD(alpha)")
    b.add_argument('--ffd-alpha', type=float, default=None,
                   help="fix FFD alpha instead of taking the best of 11")
    b.add_argument('--no-verify', action='store_true',
                   help="skip the independent feasibility check (not advised)")
    b.add_argument('--strict', action='store_true',
                   help="abort on the first verifier violation")
    b.add_argument('--mclb', action='store_true',
                   help="compute MC-LB (tight conflict+capacity bound) and report gap")
    b.add_argument('--mclb-events', type=int, default=1,
                   help="MC-LB: candidate event times to evaluate")
    b.add_argument('--mclb-exact', action='store_true',
                   help="MC-LB: exact max-clique instead of greedy (slower, tighter)")
    b.add_argument('--mclb-time', type=float, default=30.0,
                   help="MC-LB: MILP time limit per event, seconds")

    o = p.add_argument_group('output')
    o.add_argument('--csv', metavar='FILE', help="write the per-instance table")
    o.add_argument('--json', metavar='FILE', help="write full records incl. sweeps")
    o.add_argument('--limit', type=int, default=None, help="cap instances run")
    o.add_argument('--quiet', action='store_true', help="summary only")

    args = p.parse_args(argv)

    if args.mclb and not _HAVE_MCLB:
        print("warning: --mclb requested but mclb.py (or scipy) not found; "
              "falling back to the clique lower bound.", file=sys.stderr)

    paths = args.paths or (['instances'] if os.path.isdir('instances') else ['.'])
    files = discover(paths, recursive=not args.no_recursive)
    if args.limit:
        files = files[:args.limit]
    if not files:
        p.error(f"no instance files found under {paths}")

    if args.tune:
        alphas = args.alpha or DEFAULT_ALPHAS
        lams = args.lam or DEFAULT_LAMS
        thetas = args.theta or DEFAULT_THETAS
    else:
        alphas = args.alpha or [DEFAULT_ALPHAS[0]]
        lams = args.lam or [0.5]
        thetas = args.theta or [1.0]
    args._lams, args._thetas = lams, thetas

    if not args.quiet:
        print(f"{len(files)} instance(s); "
              f"{len(alphas)}x{len(lams)}x{len(thetas)} = "
              f"{len(alphas)*len(lams)*len(thetas)} config(s) each\n")

    rows = []
    for fp in files:
        try:
            inst = load_instance(fp)
        except Exception as exc:
            print(f"skip {fp}: {exc}", file=sys.stderr)
            continue
        bad = inst.feasibility_check()
        if bad and not args.quiet:
            print(f"warning: {inst.name} has {len(bad)} items violating "
                  f"m_i <= f_i or delta_i <= C-h_i", file=sys.stderr)
        rows.append(run_instance(inst, alphas, lams, thetas, args))

    if not args.quiet:
        report(rows, args)
    else:
        ok = [r for r in rows if r['srph_bins'] is not None]
        print(f"solved {len(ok)}/{len(rows)}  mean gap "
              f"{np.mean([(r['srph_bins']-r['lb'])/r['lb'] for r in ok])*100:.2f}%")

    if args.csv:
        cols = [c for c in rows[0] if c not in ('per_theta', 'per_lam')]
        with open(args.csv, 'w', newline='') as fh:
            wr = csvmod.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.csv}")
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(rows, fh, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == '__main__':
    sys.exit(main())