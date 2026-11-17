# Task 1 (and 3): rating prediction.

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recsys_lib import (DEVICE, YEAR, ALSModel, CosineSim, Data, HistNet, concat_ranges, fit_rating,
                        holdout, interaction_matrix, log, predict_rating, read_csv, ridge_fit,
                        ridge_predict, set_seed, user_ptr, write_csv)

# rating models in the ensemble
MEMBERS = [
    ("bias",   "als",  dict(k=0,   lam_f=1,   lam_b=2)),
    ("als16",  "als",  dict(k=16,  lam_f=25,  lam_b=2)),
    ("als32",  "als",  dict(k=32,  lam_f=20,  lam_b=2)),
    ("als64",  "als",  dict(k=64,  lam_f=25,  lam_b=2)),
    ("adam64", "adam", dict(k=64,  lam_f=0.2, lam_b=0.01)),
]
ALS_ITERS = 20
ADAM_EPOCHS = 40
HIST_CFG = dict(lam=1e-3)  # history-attention model settings


class TimeEncoder:
    """Turns a timestamp into a time period and how far it is from the user's usual date."""

    def __init__(self, t, u, nU, n_bins=20):
        lo, hi = np.percentile(t, [0.5, 99.5])
        self.lo, self.hi = lo, hi
        tc = np.clip(t, lo, hi)
        self.edges = np.unique(np.quantile(tc, np.linspace(0, 1, n_bins + 1)[1:-1]))
        self.n_bins = len(self.edges) + 1
        s = np.bincount(u, tc, minlength=nU + 1)
        c = np.bincount(u, minlength=nU + 1)
        self.umean = np.where(c > 0, s / np.maximum(c, 1), np.median(tc))

    def __call__(self, t, u):
        tc = np.clip(t, self.lo, self.hi)
        b = np.searchsorted(self.edges, tc).astype(np.int64)
        days = (tc - self.umean[u]) / 86400.0
        dev = (np.sign(days) * np.abs(days) ** 0.4 / 10.0).astype(np.float32)
        return b, dev


def fit_biases(u, i, r, nU, nI, lam_u=2.0, lam_i=5.0, iters=10):
    """Fit rating = average + user bias + book bias."""
    mu = float(r.mean())
    bu, bi = np.zeros(nU + 1), np.zeros(nI + 1)
    for _ in range(iters):
        bu = np.bincount(u, r - mu - bi[i], nU + 1) / (np.bincount(u, minlength=nU + 1) + lam_u)
        bi = np.bincount(i, r - mu - bu[u], nI + 1) / (np.bincount(i, minlength=nI + 1) + lam_i)
    return mu, bu, bi


def _rating_stats(idx, r, n, mu):
    """How spread out the ratings are, and how often they are 5 or 1-2 stars."""
    c = np.bincount(idx, minlength=n)
    m = (np.bincount(idx, r, n) + 4 * mu) / (c + 4)
    var = (np.bincount(idx, r * r, n) + 4 * (mu ** 2 + 1)) / (c + 4) - m ** 2
    f5 = (np.bincount(idx, (r == 5).astype(float), n) + 1) / (c + 3)
    f12 = (np.bincount(idx, (r <= 2).astype(float), n) + 0.3) / (c + 3)
    return np.sqrt(np.maximum(var, 0)), f5, f12


def rating_features(d, idx, qu, qi, qt=None):
    """Hand-made features for each (user, book[, time]) query, from the training rows idx:
    how the user rated similar books, how they rated books read right before/after this one,
    how similar users rated this book, rating spread, and (with time) how the user was
    rating around that moment."""
    u, i, r = d.u[idx], d.i[idx], d.r[idx]
    mu, bu, bi = fit_biases(u, i, r, d.nU, d.nI)
    res = r - mu - bu[u] - bi[i]
    f = {}

    # each query user's past ratings
    o = np.argsort(u, kind="stable")
    ptr = user_ptr(u[o], d.nU + 1)
    L = ptr[qu + 1] - ptr[qu]
    pos = concat_ranges(ptr[qu], L)
    rep = np.repeat(np.arange(len(qu)), L)
    h_item, h_res, h_r = i[o][pos], res[o][pos], r[o][pos]
    shape = (len(qu), d.nI + 1)
    W_res = sp.csr_matrix((h_res, (rep, h_item)), shape=shape)
    W_one = sp.csr_matrix((np.ones(len(pos)), (rep, h_item)), shape=shape)

    # similar books the user rated
    X = interaction_matrix(u, i, d.nU, d.nI)
    isim = CosineSim(X)
    for p, k in ((1, 0.5), (2, 0.1), (3, 0.1)):
        num = isim.weighted_sum(W_res, qi, power=p)
        den = isim.weighted_sum(W_one, qi, power=p)
        f[f"nbr_res_p{p}"] = num / (den + k)
        if p == 1:
            f["nbr_mass"] = np.log1p(np.maximum(den, 0))
        if p == 2:
            den2 = den
    umean = np.bincount(u, r, d.nU + 1) / np.maximum(np.bincount(u, minlength=d.nU + 1), 1)
    W_cen = sp.csr_matrix((h_r - umean[qu][rep], (rep, h_item)), shape=shape)
    f["nbr_usercentred"] = isim.weighted_sum(W_cen, qi, power=2) / (den2 + 0.1)

    # the user's most similar past books
    nP = len(qu)
    s = isim.entry_sims(qi, rep, h_item)
    mx = np.zeros(nP)
    np.maximum.at(mx, rep, s)
    f["max_sim"] = mx
    o2 = np.lexsort((-s, rep))  # per pair, most similar first
    starts = np.r_[0, np.cumsum(L)[:-1]]
    rank = np.empty(len(rep), np.int64)
    rank[o2] = np.arange(len(rep)) - starts[rep[o2]]
    cen = h_r - umean[qu][rep]
    for k in (1, 5, 20):
        w = s * ((rank < k) & (s > 0))
        den = np.bincount(rep, w, nP) + 0.05
        f[f"top{k}_res"] = np.bincount(rep, w * h_res, nP) / den
        f[f"top{k}_usercentred"] = np.bincount(rep, w * cen, nP) / den
    w = s * s
    m1 = np.bincount(rep, w * h_res, nP) / (np.bincount(rep, w, nP) + 0.1)
    m2 = np.bincount(rep, w * h_res ** 2, nP) / (np.bincount(rep, w, nP) + 0.1)
    f["nbr_res_var"] = np.maximum(m2 - m1 ** 2, 0)

    # reading order: books people read just before/after this one
    tt = np.clip(d.t[idx], 0, 2e9)
    os_ = np.lexsort((tt, u))
    su, si = u[os_], i[os_]
    rows, cols, ww = [], [], []
    for g in (1, 2, 3):
        same = su[g:] == su[:-g]
        rows.append(si[:-g][same])
        cols.append(si[g:][same])
        ww.append(np.full(same.sum(), 1.0 / g, np.float32))
    Tm = sp.csr_matrix((np.concatenate(ww), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(d.nI + 1, d.nI + 1))
    Tm = (Tm + Tm.T).tocsr()
    deg = np.asarray(Tm.sum(1)).ravel()
    cnt = np.asarray(Tm[qi[rep], h_item]).ravel()
    snorm = cnt / np.sqrt((deg[qi[rep]] + 5) * (deg[h_item] + 5))
    for name, wv, k in (("cnt", cnt, 1.0), ("norm", snorm, 0.01)):
        den = np.bincount(rep, wv, nP)
        f[f"seq_res_{name}"] = np.bincount(rep, wv * h_res, nP) / (den + k)
        f[f"seq_cen_{name}"] = np.bincount(rep, wv * cen, nP) / (den + k)
        f[f"seq_mass_{name}"] = np.log1p(den)
    mx = np.zeros(nP)
    np.maximum.at(mx, rep, snorm)
    f["seq_max"] = mx
    o3 = np.lexsort((-snorm, rep))
    best = o3[starts[L > 0]]
    f["seq_top1_res"] = np.zeros(nP)
    f["seq_top1_res"][L > 0] = np.where(snorm[best] > 0, h_res[best], 0.0)

    # similar users who rated this book
    oi = np.argsort(i, kind="stable")
    iptr = user_ptr(i[oi], d.nI + 1)
    Li = iptr[qi + 1] - iptr[qi]
    posi = concat_ranges(iptr[qi], Li)
    repi = np.repeat(np.arange(len(qi)), Li)
    shape_u = (len(qi), d.nU + 1)
    Wu_res = sp.csr_matrix((res[oi][posi], (repi, u[oi][posi])), shape=shape_u)
    Wu_one = sp.csr_matrix((np.ones(len(posi)), (repi, u[oi][posi])), shape=shape_u)
    usim = CosineSim(X.T.tocsr())
    for p in (1, 2):
        f[f"user_nbr_res_p{p}"] = (usim.weighted_sum(Wu_res, qu, power=p) /
                                   (usim.weighted_sum(Wu_one, qu, power=p) + 0.5))

    # rating-distribution statistics
    us, u5, u12 = _rating_stats(u, r, d.nU + 1, mu)
    is_, i5, i12 = _rating_stats(i, r, d.nI + 1, mu)
    f.update(user_std=us[qu], user_frac5=u5[qu], user_frac12=u12[qu],
             item_std=is_[qi], item_frac5=i5[qi], item_frac12=i12[qi])

    # how the user was rating around that time
    if qt is not None:
        t = np.clip(d.t[idx], 0, 2e9)
        tq = np.clip(qt, 0, 2e9)
        dt = np.abs(t[o][pos] - tq[rep])
        for tau in (1, 30, 365):
            w = np.exp(-dt / (tau * 86400))
            f[f"user_res_{tau}d"] = (np.bincount(rep, w * h_res, len(qu)) /
                                    (np.bincount(rep, w, len(qu)) + 2))
        same = (dt < 86400).astype(float)
        f["user_res_sameday"] = (np.bincount(rep, same * h_res, len(qu)) /
                                 (np.bincount(rep, same, len(qu)) + 1))
        yr, yq = (t // YEAR).astype(np.int64), (tq // YEAR).astype(np.int64)
        y0, nb = yr.min(), 80
        key = i * nb + np.clip(yr - y0, 0, nb - 1)
        n = (d.nI + 1) * nb
        f["item_year_res"] = ((np.bincount(key, res, n) / (np.bincount(key, minlength=n) + 10))
                              [qi * nb + np.clip(yq - y0, 0, nb - 1)])
        for tau in (30, 365):
            w = np.exp(-dt / (tau * 86400)).astype(np.float32)
            Wt = sp.csr_matrix((h_res * w, (rep, h_item)), shape=shape)
            Wt1 = sp.csr_matrix((w, (rep, h_item)), shape=shape)
            f[f"nbr_time_{tau}d"] = (isim.weighted_sum(Wt, qi, power=2) /
                                    (isim.weighted_sum(Wt1, qi, power=2) + 0.1))
    return np.column_stack(list(f.values())), list(f)


def user_item_stats(d, idx):
    """How many ratings each user / book has (and the user's share of 0 ratings)."""
    nU, nI = d.nU + 1, d.nI + 1
    uc = np.bincount(d.u[idx], minlength=nU)
    ic = np.bincount(d.i[idx], minlength=nI)
    uz = np.bincount(d.u[idx], (d.r[idx] == 0).astype(float), minlength=nU)
    return dict(lu=np.log1p(uc), li=np.log1p(ic), zu=(uz + 0.2) / (uc + 10))


def stat_features(st, u, i):
    return np.column_stack([st["lu"][u], st["li"][i], st["zu"][u]])


# blender (Added by Claude)
class Blender:
    """Linear blend of everything, plus a small neural network that corrects it."""

    def __init__(self, hidden=32, epochs=15, seeds=3, wd=1e-3, drop=0.1):
        self.hidden, self.epochs, self.seeds, self.wd, self.drop = hidden, epochs, seeds, wd, drop

    def _t(self, x):
        return torch.tensor(x, dtype=torch.float32, device=DEVICE)

    def fit(self, X, y):
        self.w = ridge_fit(X, y)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xt, yt = self._t((X - self.mu) / self.sd), self._t(y)
        base = self._t(ridge_predict(self.w, X))
        self.nets = []
        for seed in range(self.seeds):
            set_seed(seed)
            net = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.ReLU(), nn.Dropout(self.drop),
                                nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                                nn.Linear(self.hidden, 1)).to(DEVICE)
            opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=self.wd)
            for _ in range(self.epochs):
                net.train()
                perm = torch.randperm(len(Xt), device=DEVICE)
                for s in range(0, len(Xt), 256):
                    k = perm[s:s + 256]
                    loss = ((base[k] + net(Xt[k])[:, 0] - yt[k]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            self.nets.append(net.eval())
        return self

    def predict(self, X, ridge_only=False):
        p = ridge_predict(self.w, X)
        if not ridge_only:
            with torch.no_grad():
                Xt = self._t((X - self.mu) / self.sd)
                p = p + np.mean([n(Xt)[:, 0].cpu().numpy() for n in self.nets], 0)
        return np.clip(p, 0, 5)


# member helper functions
def train_member(kind, kw, d, idx, n_iter, te=None, val=None):
    u, i, r = d.u[idx], d.i[idx], d.r[idx]
    b, dev = te(d.t[idx], u) if te else (None, None)
    if kind == "als":
        m = ALSModel(d.nU, d.nI, kw["k"], kw["lam_f"], kw["lam_b"],
                     n_bins=te.n_bins if te else 0)
        m.fit(u, i, r, iters=n_iter, b=b, dev=dev, val=val, verbose=False)
        return m, m.best_iter if val else n_iter, m.best_mse if val else None
    m, ep, mse = fit_rating(u, i, r, d.nU, d.nI, kw["k"], kw["lam_f"], kw["lam_b"],
                            epochs=n_iter, val=val, verbose=False,
                            n_bins=te.n_bins if te else 0, b=b, dev=dev)
    return m, ep, mse


def member_predict(kind, m, u, i, b=None, dev=None):
    if kind == "als":
        return m.predict(u, i, b, dev)
    return predict_rating(m, u, i, b, dev)


#  train / predict
def blend_matrix(d, tr, va, use_time, n_iters, chunk=60000):
    """Train every model on the training part and return their predictions + features
    for the validation rows (the blender's inputs)."""
    te = TimeEncoder(d.t[tr], d.u[tr], d.nU) if use_time else None
    vb, vdev = te(d.t[va], d.u[va]) if te else (None, None)
    val = dict(u=d.u[va], i=d.i[va], r=d.r[va], b=vb, dev=vdev)
    preds = []
    for name, kind, kw in MEMBERS:
        m, it, mse = train_member(kind, kw, d, tr, ALS_ITERS if kind == "als" else ADAM_EPOCHS,
                                  te, val)
        n_iters.setdefault(name, []).append(it)
        preds.append(member_predict(kind, m, val["u"], val["i"], vb, vdev))
        log(f"  member {name:7s} MSE {mse:.4f} (best iter/epoch {it})")
        del m
    t_tr, t_va = (d.t[tr], d.t[va]) if use_time else (None, None)
    h = HistNet(**HIST_CFG).fit(d.u[tr], d.i[tr], d.r[tr], t_tr, d.nU, d.nI,
                                val=dict(val, t=t_va), use_time=use_time)
    n_iters.setdefault("hist", []).append(h.best_epoch)
    hist_pred = h.predict(val["u"], val["i"], t_va)
    log(f"  member history-attention MSE {h.best_mse:.4f} (best epoch {h.best_epoch})")
    del h
    log(f"  computing features for {len(va)} rows")
    F = np.vstack([rating_features(d, tr, val["u"][s:s + chunk], val["i"][s:s + chunk],
                                   d.t[va][s:s + chunk] if use_time else None)[0]
                   for s in range(0, len(va), chunk)])
    return np.column_stack(preds + [stat_features(user_item_stats(d, tr), val["u"], val["i"]), F, hist_pred])


def train(train_file, use_time=False, n_val=150000, seed=0, save=None):
    """Tune on a validation split, fit the blender, then retrain everything on all data."""
    set_seed(seed)
    d = Data(train_file)
    tr, va = holdout(d, n_val, seed)
    log(f"train {len(tr)}  validation {len(va)}  (time features: {use_time})")
    n_iters = {}
    Xv, y = blend_matrix(d, tr, va, use_time, n_iters), d.r[va]
    if save:  # optional: save the blender inputs
        np.savez(save, X=Xv, y=y)

    # fair score: fit the blender on half the validation set, test on the other half
    half = np.random.default_rng(seed).permutation(len(y)) < len(y) // 2
    bl = Blender().fit(Xv[half], y[half])
    mse = lambda p: ((p - y[~half]) ** 2).mean()
    best_single = min(mse(Xv[~half, j]) for j in range(len(MEMBERS)))
    log(f"best single member: {best_single:.4f} | ridge blend: {mse(bl.predict(Xv[~half], True)):.4f}"
        f" | ridge + neural-net blend: {mse(bl.predict(Xv[~half])):.4f}   (held-out half)")
    blender = Blender().fit(Xv, y)

    # retrain everything on all the data
    allidx = np.arange(d.n)
    te_full = TimeEncoder(d.t, d.u, d.nU) if use_time else None
    models = []
    for name, kind, kw in MEMBERS:
        it = max(1, int(round(np.mean(n_iters[name]))))
        log(f"retraining {name} on all data for {it} iterations")
        m, _, _ = train_member(kind, kw, d, allidx, it, te_full, None)
        models.append((kind, m))
    it = max(1, int(round(np.mean(n_iters["hist"]))))
    log(f"retraining history-attention on all data for {it} epochs")
    hist = HistNet(**dict(HIST_CFG, epochs=it)).fit(d.u, d.i, d.r, d.t if use_time else None,
                                                    d.nU, d.nI, use_time=use_time)
    return dict(d=d, models=models, hist=hist, blender=blender, stats=user_item_stats(d, allidx),
                te=te_full, use_time=use_time)


def predict(bundle, test_file, out_file, use_time=False):
    header, rows = read_csv(test_file)
    d = bundle["d"]
    u = d.enc_u([r[0] for r in rows])
    i = d.enc_i([r[1] for r in rows])
    b = dev = t = None
    if use_time:
        t = np.array([float(r[header.index("timestamp")]) for r in rows])
        b, dev = bundle["te"](t, u)
    preds = [member_predict(kind, m, u, i, b, dev) for kind, m in bundle["models"]]
    F, _ = rating_features(d, np.arange(d.n), u, i, t)
    X = np.column_stack(preds + [stat_features(bundle["stats"], u, i), F, bundle["hist"].predict(u, i, t)])
    p = bundle["blender"].predict(X)
    n_in = len(header) - 1  # last column is the one to fill in
    write_csv(out_file, header, [r[:n_in] + [f"{x:.5f}"] for r, x in zip(rows, p)])


if __name__ == "__main__":
    bundle = train("tasks_1_2_train.csv")
    predict(bundle, "task_1_test.csv", "task_1_predictions.csv")
