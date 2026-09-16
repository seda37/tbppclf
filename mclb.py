#!/usr/bin/env python3
"""
MC-LB (conflict + capacity aware lower bound) as an importable module.
Adapted from tbpp_lb.py to accept already-loaded items instead of a path.

Public entry point:
    mclb_value(items, C, n_events=1, greedy=True, milp_time=30.0,
               clique_time=10.0, outside_conflicts=True) -> (int_lb, detail|None)

VALIDITY GUARD: a sub-MILP result is used as a lower bound ONLY if it solved to
proven optimality (status == 0). If it times out / is not optimal, that event is
skipped rather than trusted, because a primal incumbent would be an invalid
(too-high) lower bound. Returns the best valid event bound, or None if none.

Each `item` must expose: id, w, s, e, k, h, delta, conflicts (set of ids).
"""

import math
import time
import sys

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


# ------------------------------------------------------------
class _Ctx:
    """Lightweight instance context built from a list of items."""
    def __init__(self, items, C):
        self.C = float(C)
        self.items = {it.id: it for it in items}
        self.ids = sorted(self.items)
        self.adj = {i: set(self.items[i].conflicts) for i in self.ids}
        self.fmin = {i: math.ceil(self.items[i].w / (self.C - self.items[i].h))
                     for i in self.ids}
        self.events = sorted({self.items[i].s for i in self.ids})

    def alive(self, i, t):
        it = self.items[i]
        return it.s <= t < it.e

    def active(self, t):
        return [i for i in self.ids if self.alive(i, t)]

    def conflict(self, i, j):
        return j in self.adj[i]


# ------------------------------------------------------------
def max_clique(adj, nodes, time_limit=10.0):
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
    nodes = list(nodes)
    deg = {i: len(adj[i] & set(nodes)) for i in nodes}
    Q = []
    for i in sorted(nodes, key=lambda x: -deg[x]):
        if all(j in adj[i] for j in Q):
            Q.append(i)
    return Q


# ------------------------------------------------------------
class _Model:
    def __init__(self):
        self.cost, self.integ, self.lo, self.up = [], [], [], []
        self.rows = []

    def var(self, cost=0.0, integer=False, lo=0.0, up=np.inf):
        self.cost.append(cost); self.integ.append(1 if integer else 0)
        self.lo.append(lo); self.up.append(up)
        return len(self.cost) - 1

    def row(self, coefs, lo=-np.inf, up=np.inf):
        self.rows.append((dict(coefs), lo, up))

    def solve(self, time_limit=None):
        n = len(self.cost)
        A = np.zeros((len(self.rows), n))
        rl = np.empty(len(self.rows)); ru = np.empty(len(self.rows))
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


# ------------------------------------------------------------
def _mc_lb_at(ctx, t, Q, outside_conflicts=True, time_limit=30.0):
    C = ctx.C
    S = set(ctx.active(t))
    Q = [q for q in Q if q in S]
    if not Q:
        return None, "empty clique"
    O = [i for i in S if i not in Q]
    IT = ctx.items

    m = _Model()
    y, g = {}, {}
    bins = []
    for q in Q:
        nmax = IT[q].k + 1
        for r in range(nmax):
            b = len(bins); bins.append((q, r))
            y[b] = m.var(cost=1.0, integer=True, lo=0, up=1)
            g[b] = m.var(lo=0, up=C - IT[q].h)

    f, u = {}, {}
    for b, (q, r) in enumerate(bins):
        for o in O:
            if ctx.conflict(o, q):
                continue
            f[(o, b)] = m.var(lo=0, up=IT[o].w)
            u[(o, b)] = m.var(integer=True, lo=0, up=1)

    L = {o: m.var(lo=0, up=IT[o].w) for o in O}
    v = {o: m.var(integer=True, lo=0, up=1) for o in O}
    z = m.var(cost=1.0, integer=True, lo=0, up=len(O) + len(bins) + 5)

    for q in Q:
        idx = [b for b, (qq, r) in enumerate(bins) if qq == q]
        m.row({g[b]: 1.0 for b in idx}, IT[q].w, IT[q].w)
        for b in idx:
            m.row({g[b]: 1.0, y[b]: -(C - IT[q].h)}, up=0.0)
        for b1, b2 in zip(idx, idx[1:]):
            m.row({y[b1]: 1.0, y[b2]: -1.0}, lo=0.0)

    for b, (q, r) in enumerate(bins):
        co = {g[b]: 1.0, y[b]: IT[q].h - C}
        for o in O:
            if (o, b) in f:
                co[f[(o, b)]] = 1.0; co[u[(o, b)]] = IT[o].h
        m.row(co, up=0.0)

    for o in O:
        co = {L[o]: 1.0}
        for b in range(len(bins)):
            if (o, b) in f:
                co[f[(o, b)]] = 1.0
        m.row(co, IT[o].w, IT[o].w)

    for (o, b), fi in f.items():
        ui = u[(o, b)]
        m.row({fi: 1.0, ui: -min(IT[o].w, C - IT[o].h)}, up=0.0)
        m.row({fi: 1.0, ui: -IT[o].delta}, lo=0.0)
        m.row({ui: 1.0, y[b]: -1.0}, up=0.0)

    for o in O:
        us = {u[(o, b)]: 1.0 for b in range(len(bins)) if (o, b) in f}
        cap = C - IT[o].h
        budget = IT[o].k + 1
        co = dict(us); co[v[o]] = co.get(v[o], 0) + 1.0
        m.row(co, up=budget)
        co2 = {k: val * cap for k, val in us.items()}
        co2[L[o]] = co2.get(L[o], 0) + 1.0
        m.row(co2, up=budget * cap)
        m.row({L[o]: 1.0, v[o]: -IT[o].w}, up=0.0)

    if outside_conflicts:
        for b in range(len(bins)):
            for a_ in range(len(O)):
                for b_ in range(a_ + 1, len(O)):
                    o1, o2 = O[a_], O[b_]
                    if ctx.conflict(o1, o2) and (o1, b) in f and (o2, b) in f:
                        m.row({u[(o1, b)]: 1.0, u[(o2, b)]: 1.0}, up=1.0)

    co = {}
    for o in O:
        co[L[o]] = 1.0; co[v[o]] = IT[o].h
    co[z] = -C
    m.row(co, up=0.0)

    res = m.solve(time_limit=time_limit)
    # VALIDITY GUARD: only a proven-optimal solve is a valid lower bound.
    if res.x is None or getattr(res, "status", 1) != 0:
        return None, "not proven optimal (status=%s)" % getattr(res, "status", "?")
    lb = math.ceil(res.fun - 1e-6)
    return lb, dict(t=t, clique=len(Q), outside=len(O))


# ------------------------------------------------------------
def mclb_value(items, C, n_events=1, greedy=True, milp_time=30.0,
               clique_time=10.0, outside_conflicts=True):
    ctx = _Ctx(items, C)

    # rank events by overhead-weighing value (tightest capacity first)
    ranked = sorted(ctx.events,
                    key=lambda t: -sum(ctx.items[i].w + ctx.items[i].h * ctx.fmin[i]
                                       for i in ctx.active(t)))
    cands = ranked[:max(1, n_events)]

    # add the global-clique window
    if greedy:
        Qg = greedy_clique(ctx.adj, ctx.ids)
    else:
        Qg, _ = max_clique(ctx.adj, ctx.ids, time_limit=clique_time)
    if Qg:
        lo = max(ctx.items[i].s for i in Qg)
        hi = min(ctx.items[i].e for i in Qg)
        if lo < hi and lo not in cands:
            cands.append(lo)

    best, best_det = None, None
    for t in cands:
        S = ctx.active(t)
        if greedy:
            Q = greedy_clique(ctx.adj, S)
        else:
            Q, _ = max_clique(ctx.adj, S, time_limit=clique_time)
        val, det = _mc_lb_at(ctx, t, Q, outside_conflicts=outside_conflicts,
                             time_limit=milp_time)
        if val is not None and (best is None or val > best):
            best, best_det = val, det
    return best, best_det
