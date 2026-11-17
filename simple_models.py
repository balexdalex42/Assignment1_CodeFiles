"""Simple models for every task, evaluated on the same kind of validation
split as the full pipelines (CPU only, a few minutes in total).

Run:  python simple_models.py          (all tasks)
      python simple_models.py 1 5      (only tasks 1 and 5)
"""
import sys

import numpy as np

from recsys_lib import YEAR, CosineSim, Data, holdout, interaction_matrix, pair_weights, user_ptr

HOUR = 3600.0


def show(task, name, value, fmt="{:.4f}"):
    print(f"  task {task}  {name:45s} {fmt.format(value)}", flush=True)


# ---------------------------------------------------------------- ratings
def bias_model(u, i, r, nU, nI, lam_u, lam_i, iters=10):
    """r ~ mu + b_u + b_i, fitted by alternating closed-form ridge updates."""
    mu = r.mean()
    bu, bi = np.zeros(nU + 1), np.zeros(nI + 1)
    for _ in range(iters):
        bu = np.bincount(u, r - mu - bi[i], nU + 1) / (np.bincount(u, minlength=nU + 1) + lam_u)
        bi = np.bincount(i, r - mu - bu[u], nI + 1) / (np.bincount(i, minlength=nI + 1) + lam_i)
    return mu, bu, bi


def rating_task(task, path, use_time):
    d = Data(path)
    tr, va = holdout(d, 50000)
    u, i, r = d.u[tr], d.i[tr], d.r[tr]
    vu, vi, vr = d.u[va], d.i[va], d.r[va]
    mse = lambda p: float(((np.clip(p, 0, 5) - vr) ** 2).mean())
    mu = r.mean()
    show(task, "global mean", mse(np.full(len(va), mu)))
    for name, idx, vidx, n in (("user mean", u, vu, d.nU), ("item mean", i, vi, d.nI)):
        s, c = np.bincount(idx, r, n + 1), np.bincount(idx, minlength=n + 1)
        show(task, f"{name} (shrunk to global, lam=5)", mse(((s + 5 * mu) / (c + 5))[vidx]))
    best = None
    for lu in (1, 2, 5):
        for li in (1, 2, 5, 10):
            m, bu, bi = bias_model(u, i, r, d.nU, d.nI, lu, li)
            e = mse(m + bu[vu] + bi[vi])
            if best is None or e < best[0]:
                best = (e, lu, li, (m, bu, bi))
    show(task, f"bias model mu+b_u+b_i (lam_u={best[1]}, lam_i={best[2]})", best[0])
    if use_time:
        # bias model + global year-bin bias + item x year-bin bias on the residual
        m, bu, bi = best[3]
        t = np.clip(d.t, 0, 2e9)
        yb = ((t - t[tr].min()) // YEAR).astype(np.int64)
        res = r - m - bu[u] - bi[i]
        nb = yb.max() + 1
        g = np.bincount(yb[tr], res, nb) / (np.bincount(yb[tr], minlength=nb) + 50)
        res2 = res - g[yb[tr]]
        key_tr, key_va = i * nb + yb[tr], vi * nb + yb[va]
        ib = np.bincount(key_tr, res2, (d.nI + 1) * nb) / (np.bincount(key_tr, minlength=(d.nI + 1) * nb) + 20)
        show(task, "bias model + year bias + item x year bias", mse(m + bu[vu] + bi[vi] + g[yb[va]] + ib[key_va]))


# ---------------------------------------------------------- which-item tasks
def pair_acc(sp, sn):
    return float(((sp > sn) + 0.5 * (sp == sn)).mean())


def which_task(task, path, use_time):
    d = Data(path)
    tr, va = holdout(d, 50000)
    rng = np.random.default_rng(0)
    vu, pos, t = d.u[va], d.i[va], d.t[va]
    neg = d.i[tr][rng.integers(0, len(tr), len(va))]  # popularity-sampled, like the test
    X = interaction_matrix(d.u[tr], d.i[tr], d.nU, d.nI)
    pop = np.bincount(d.i[tr], minlength=d.nI + 1)
    show(task, "popularity", pair_acc(pop[pos], pop[neg]))
    inh = lambda c: np.asarray(X[vu, c]).ravel()
    show(task, "not-in-history rule (else coin flip)", pair_acc(-inh(pos), -inh(neg)))
    o = tr[np.argsort(d.u[tr], kind="stable")]
    ptr = user_ptr(d.u[o], d.nU + 1)
    lengths = ptr[vu + 1] - ptr[vu]
    W = pair_weights(None, ptr[vu], lengths, d.i[o], d.nI)
    sim = CosineSim(X)
    sp_, sn = sim.weighted_sum(W, pos), sim.weighted_sum(W, neg)
    show(task, "item-item cosine to user history (sum)", pair_acc(sp_, sn))
    show(task, "cosine sum + not-in-history rule", pair_acc(sp_ - 1e6 * inh(pos), sn - 1e6 * inh(neg)))
    if use_time:
        # how active is the candidate within +-1 year of t, relative to its popularity
        tt = np.clip(d.t[tr], 0, 2e9)
        o2 = np.lexsort((tt, d.i[tr]))
        key = d.i[tr][o2].astype(np.int64) * 4_000_000_000 + tt[o2].astype(np.int64)
        tq = np.clip(t, 0, 2e9).astype(np.int64)

        def dens(c):
            base = c.astype(np.int64) * 4_000_000_000
            lo = np.searchsorted(key, base + np.maximum(tq - int(YEAR), 0))
            hi = np.searchsorted(key, base + tq + int(YEAR))
            return np.log((hi - lo + 1) / (pop[c] + 2))
        dp, dn = dens(pos), dens(neg)
        show(task, "item activity near t (density)", pair_acc(dp, dn))
        # crude combination: rank-normalise and add
        z = lambda a, b: (a - np.concatenate([a, b]).mean()) / (np.concatenate([a, b]).std() + 1e-9)
        cp, cn = z(sp_, sn), z(sn, sp_)
        show(task, "cosine + density (z-scored sum) + history rule",
             pair_acc(cp + z(dp, dn) - 1e6 * inh(pos), z(sn, sp_) * 0 + cn + z(dn, dp) - 1e6 * inh(neg)))


# ------------------------------------------------------------------ task 5
def task5():
    d = Data("task_5_train.csv")
    ptr = user_ptr(d.u, d.nU + 1)
    last_rows = ptr[1:][:-1] - 1
    mask = np.ones(d.n, bool)
    mask[last_rows] = False
    tr = np.where(mask)[0]
    rng = np.random.default_rng(0)
    va = rng.choice(last_rows, 60000, replace=False)
    vu, pos = d.u[va], d.i[va]
    neg = d.i[tr][rng.integers(0, len(tr), len(va))]
    u, i = d.u[tr], d.i[tr]
    X = interaction_matrix(u, i, d.nU, d.nI)
    pop = np.bincount(i, minlength=d.nI + 1)
    show(5, "popularity", pair_acc(pop[pos], pop[neg]))
    tptr = user_ptr(u, d.nU + 1)
    last = i[tptr[vu + 1] - 1]
    same = u[1:] == u[:-1]
    import scipy.sparse as sp
    T = sp.csr_matrix((np.ones(same.sum()), (i[:-1][same], i[1:][same])), shape=(d.nI + 1, d.nI + 1))
    tp, tn = np.asarray(T[last, pos]).ravel(), np.asarray(T[last, neg]).ravel()
    show(5, "transition count last item -> candidate", pair_acc(tp, tn))
    sim = CosineSim(X)
    W1 = pair_weights(None, tptr[vu + 1] - 1, np.ones(len(vu), np.int64), i, d.nI)
    show(5, "cosine to last item", pair_acc(sim.weighted_sum(W1, pos), sim.weighted_sum(W1, neg)))
    W = pair_weights(None, tptr[vu], tptr[vu + 1] - tptr[vu], i, d.nI)
    cp, cn = sim.weighted_sum(W, pos), sim.weighted_sum(W, neg)
    show(5, "cosine to whole history (sum)", pair_acc(cp, cn))
    tt = np.clip(d.t[tr], 0, 2e9)
    recent = np.bincount(i[tt >= np.quantile(tt, 0.9)], minlength=d.nI + 1)
    rp, rn = np.log((recent[pos] + 1) / (pop[pos] + 10)), np.log((recent[neg] + 1) / (pop[neg] + 10))
    show(5, "recent popularity (last 10% of time) / popularity", pair_acc(rp, rn))
    z = lambda a, b: (a - np.concatenate([a, b]).mean()) / (np.concatenate([a, b]).std() + 1e-9)
    ip, ineg = np.asarray(X[vu, pos]).ravel(), np.asarray(X[vu, neg]).ravel()
    show(5, "log transitions + cosine + recency (z-sum) + rule",
         pair_acc(z(np.log1p(tp), np.log1p(tn)) + z(cp, cn) + z(rp, rn) - 1e6 * ip,
                  z(np.log1p(tn), np.log1p(tp)) + z(cn, cp) + z(rn, rp) - 1e6 * ineg))


# ------------------------------------------------------------------ task 6
def task6():
    d = Data("task_6_train.csv")
    ptr = user_ptr(d.u, d.nU)
    t = np.clip(d.t, 0, 2e9)
    ends = ptr[1:] - 1                       # held-out = last interaction
    users = np.where(np.diff(ptr) >= 4)[0]
    y = t[ends[users]] - t[ends[users] - 1]  # the gap to predict
    gaps = [np.diff(t[ptr[u]:ends[u]]) for u in users]
    mae = lambda p: float(np.abs(p - y).mean())
    allg = np.concatenate(gaps)
    show(6, "global median gap", mae(np.full(len(y), np.median(allg))), "{:,.0f}")
    show(6, "user median gap (course baseline)", mae(np.array([np.median(g) for g in gaps])), "{:,.0f}")
    lg = [np.log1p(g / HOUR) for g in gaps]
    glob = np.median(np.log1p(allg / HOUR))
    for k in (2, 5, 10):
        p = np.array([(len(l) * np.median(l) + k * glob) / (len(l) + k) for l in lg])
        show(6, f"user median of log-gaps shrunk to global (k={k})", mae(np.expm1(p) * HOUR), "{:,.0f}")
    for n in (3, 5, 10):
        show(6, f"median of the user's last {n} gaps",
             mae(np.array([np.median(g[-n:]) for g in gaps])), "{:,.0f}")
    for q in (0.3, 0.4, 0.5):
        show(6, f"user gap quantile q={q}", mae(np.array([np.quantile(g, q) for g in gaps])), "{:,.0f}")


if __name__ == "__main__":
    tasks = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5, 6]
    for k in tasks:
        print(f"== Task {k}", flush=True)
        if k == 1: rating_task(1, "tasks_1_2_train.csv", False)
        if k == 2: which_task(2, "tasks_1_2_train.csv", False)
        if k == 3: rating_task(3, "tasks_3_4_train.csv", True)
        if k == 4: which_task(4, "tasks_3_4_train.csv", True)
        if k == 5: task5()
        if k == 6: task6()
