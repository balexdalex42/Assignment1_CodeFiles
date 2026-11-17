#Task 2 (and 4): "which of item1 / item2 is the real interaction".

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recsys_lib import (DEVICE, YEAR, ALSModel, CosineSim, Data, PairwiseLR, bpr_scores, concat_ranges, entry_products,
                        fit_bpr, holdout, interaction_matrix, log, pair_weights,
                        read_csv, set_seed, user_ptr, write_csv)

# BPR settings for task 2 / task 4
BPR_MEMBERS = {
    False: [dict(k=64, lam=1e-3, lr=2e-3), dict(k=128, lam=1e-2, lr=1e-3)],    # task 2
    True:  [dict(k=128, lam=1e-2, lr=1e-3), dict(k=256, lam=1e-2, lr=1e-3)],   # task 4
}
BPR_EPOCHS = 50
ALS_CFG = dict(k=32, lam_f=20, lam_b=2, iters=8)
# time-aware BPR settings (task 4)
CTX_CFG = dict(K=5, dim=128, lam=1e-2, n_bins=40, epochs=40, patience=4)
GAP_EDGES = np.array([3600, 86400, 7 * 86400, 30 * 86400, 182 * 86400, 730 * 86400])


class CtxBPRNet(nn.Module):
    """Describes the user by their own vector plus the books they read around that moment
    (closer books count more), and adds how popular the book was in that period."""

    def __init__(self, nU, nI, n_ctx, dim, n_bins):
        super().__init__()
        self.pad, self.n_bins = nI + 1, n_bins
        self.P = nn.Embedding(nU + 1, dim)
        self.Q = nn.Embedding(nI + 2, dim, padding_idx=nI + 1)   # context-side vectors
        self.E = nn.Embedding(nI + 2, dim)                        # candidate-side vectors
        self.b = nn.Embedding(nI + 2, 1)
        self.bt = nn.Embedding((nI + 2) * n_bins, 1)
        self.w = nn.Parameter(torch.zeros(n_ctx))
        self.g = nn.Parameter(torch.zeros(len(GAP_EDGES) + 1))
        for e in (self.P, self.Q, self.E):
            nn.init.normal_(e.weight, 0, 0.05)
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.bt.weight)

    def user_vec(self, uu, cc, gg):
        m = (cc != self.pad).float() * torch.exp(self.g)[gg]
        wq = self.Q(cc) * (torch.softmax(self.w, 0) * len(self.w))[None, :, None] * m[..., None]
        return self.P(uu) + wq.sum(1) / ((m.sum(1, keepdim=True) + 1e-4).sqrt() + 1)

    def score(self, z, c, bb):
        return (z * self.E(c)).sum(-1) + self.b(c)[..., 0] + self.bt(c * self.n_bins + bb)[..., 0]


class CtxBPR:
    """Trains the time-aware BPR on every interaction against popular random books.
    mode="next" (task 5) only uses the books read before the prediction point."""

    def __init__(self, ctx, val=None, epochs=None, seed=0, mode="around"):
        cfg = CTX_CFG
        self.c, self.K, self.mode = ctx, cfg["K"], mode
        d = ctx.d
        self.pad = d.nI + 1
        lo, hi = np.percentile(ctx.ht, [0.5, 99.5])
        self.edges = np.quantile(np.clip(ctx.ht, lo, hi), np.linspace(0, 1, cfg["n_bins"] + 1)[1:-1])
        n = len(ctx.hi)
        j = np.arange(n)
        if mode == "next":
            tq = ctx.ht[np.maximum(j - 1, 0)]
            C, G = self.context(ctx.hu, j - 1, None, tq)
        else:
            tq = ctx.ht
            C, G = self.context(ctx.hu, j - 1, j + 1, tq)
        T = lambda a: torch.tensor(a, device=DEVICE)
        hu, hi_, Ct, Gt, Bt = T(ctx.hu), T(ctx.hi), T(C), T(G), T(np.searchsorted(self.edges, tq))
        set_seed(seed)
        self.net = CtxBPRNet(d.nU, d.nI, C.shape[1], cfg["dim"], cfg["n_bins"]).to(DEVICE)
        opt = torch.optim.Adam(self.net.parameters(), 2e-3)
        n_ep = epochs or cfg["epochs"]
        best, best_state, self.best_epoch = -1, None, n_ep
        for ep in range(n_ep):
            self.net.train()
            perm = torch.randperm(n, device=DEVICE)
            for s in range(0, n, 4096):
                k = perm[s:s + 4096]
                uu, cc, pp, gg, bb = hu[k], Ct[k], hi_[k], Gt[k], Bt[k]
                nn_ = hi_[torch.randint(0, n, (len(k),), device=DEVICE)]  # popularity negatives
                cc = torch.where(torch.rand(cc.shape, device=DEVICE) < 0.2, torch.full_like(cc, self.pad), cc)
                z = self.net.user_vec(uu, cc, gg)
                x = self.net.score(z, pp, bb) - self.net.score(z, nn_, bb)
                reg = cfg["lam"] * (z.pow(2).sum(1) + self.net.E(pp).pow(2).sum(1)
                                    + self.net.E(nn_).pow(2).sum(1)).mean()
                loss = -F.logsigmoid(x).mean() + reg
                opt.zero_grad()
                loss.backward()
                opt.step()
            if val is not None:
                acc = (self.scores(val["u"], val["pos"], val.get("t")) >
                       self.scores(val["u"], val["neg"], val.get("t"))).mean()
                if acc > best:
                    best, self.best_epoch = acc, ep + 1
                    best_state = {k_: v.clone() for k_, v in self.net.state_dict().items()}
                elif ep + 1 - self.best_epoch >= cfg["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
            log(f"  context BPR: val acc {best:.4f} at epoch {self.best_epoch}")
        self.net.eval()

    def context(self, users, p_before, p_after, tq):
        c = self.c
        lo, hi = c.hptr[users], c.hptr[users + 1]
        n = len(c.hi)
        cols, gaps = [], []
        after = [] if p_after is None else [(p_after + k, p_after + k < hi) for k in range(self.K)]
        for q, ok in [(p_before - k, p_before - k >= lo) for k in range(self.K)] + after:
            qc = np.clip(q, 0, n - 1)
            cols.append(np.where(ok, c.hi[qc], self.pad))
            gaps.append(np.searchsorted(GAP_EDGES, np.abs(c.ht[qc] - tq)))
        return np.stack(cols, 1), np.stack(gaps, 1)

    def scores(self, u, cand, t=None):
        if self.mode == "next":  # context = the user's last K books
            p = self.c.hptr[u + 1]
            tq = self.c.ht[np.clip(p - 1, 0, len(self.c.ht) - 1)]
            C, G = self.context(u, p - 1, None, tq)
        else:
            tq = np.clip(t, 0, 2e9)
            p = np.searchsorted(self.c.hkey, u.astype(np.int64) * 4_000_000_000 + tq.astype(np.int64))
            C, G = self.context(u, p - 1, p, tq)
        T = lambda a: torch.tensor(a, device=DEVICE)
        out = []
        with torch.no_grad():
            for s in range(0, len(u), 50000):
                sl = slice(s, s + 50000)
                z = self.net.user_vec(T(u[sl]), T(C[sl]), T(G[sl]))
                out.append(self.net.score(z, T(cand[sl]), T(np.searchsorted(self.edges, tq[sl]))).cpu().numpy())
        return np.concatenate(out)


class Context:
    """Everything the features need, fitted on the training rows idx."""

    def __init__(self, d, idx, use_time, val=None, n_epochs=None):
        self.d, self.use_time = d, use_time
        u, i = d.u[idx], d.i[idx]
        self.X = interaction_matrix(u, i, d.nU, d.nI)
        self.sim = CosineSim(self.X)
        self.Xc = self.X.tocsc()
        self.XcT = self.X.T.tocsr()
        self.pop = np.bincount(i, minlength=d.nI + 1)
        # history sorted by user, then time
        o = idx[np.lexsort((d.t[idx], d.u[idx]))]
        self.hu, self.hi, self.ht = d.u[o], d.i[o], np.clip(d.t[o], 0, 2e9)
        self.hptr = user_ptr(self.hu, d.nU + 1)
        # each book's reads over time
        o2 = idx[np.lexsort((np.clip(d.t[idx], 0, 2e9), d.i[idx]))]
        self.it_key = d.i[o2].astype(np.int64) * 4_000_000_000 + np.clip(d.t[o2], 0, 2e9).astype(np.int64)
        self.iptr = user_ptr(d.i[o2], d.nI + 1)
        self.item_first = np.full(d.nI + 1, np.nan)
        self.item_last = np.full(d.nI + 1, np.nan)
        has = self.iptr[1:] > self.iptr[:-1]
        self.item_first[has] = (self.it_key[self.iptr[:-1][has]] % 4_000_000_000)
        self.item_last[has] = (self.it_key[self.iptr[1:][has] - 1] % 4_000_000_000)

        # how often readers go from one book straight to another
        same = self.hu[1:] == self.hu[:-1]
        a_, b_ = self.hi[:-1][same], self.hi[1:][same]
        T = sp.csr_matrix((np.ones(len(a_), np.float32), (a_, b_)), shape=(d.nI + 1, d.nI + 1))
        self.trans = (T + T.T).tocsr()                      # either direction
        self.trans_deg = np.asarray(self.trans.sum(1)).ravel()
        self.hkey = self.hu.astype(np.int64) * 4_000_000_000 + self.ht.astype(np.int64)
        if use_time:
            # books read within 5 positions of each other (closer = more weight)
            rows, cols, w = [], [], []
            for g in range(1, 6):
                same = self.hu[g:] == self.hu[:-g]
                rows.append(self.hi[:-g][same])
                cols.append(self.hi[g:][same])
                w.append(np.full(same.sum(), 1.0 / g, np.float32))
            Tw = sp.csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(d.nI + 1, d.nI + 1))
            self.win = (Tw + Tw.T).tocsr()
            self.win_deg = np.asarray(self.win.sum(1)).ravel()

        self.bpr, self.epochs = [], {}
        for j, cfg in enumerate(BPR_MEMBERS[use_time]):
            ep = BPR_EPOCHS if n_epochs is None else n_epochs[j]
            m, best_ep, acc = fit_bpr(u, i, d.nU, d.nI, cfg["k"], cfg["lam"], epochs=ep, lr=cfg["lr"],
                                      neg_pool=i, val=val, verbose=False, seed=j, patience=6)
            if val is not None:
                log(f"  BPR k={cfg['k']}: val acc {acc:.4f} at epoch {best_ep}")
            self.epochs[j] = best_ep
            self.bpr.append(m)
        self.als = ALSModel(d.nU, d.nI, ALS_CFG["k"], ALS_CFG["lam_f"], ALS_CFG["lam_b"])
        self.als.fit(u, i, d.r[idx], iters=ALS_CFG["iters"], verbose=False)
        if use_time:
            self.ctx_bpr = CtxBPR(self, val=val, epochs=None if n_epochs is None else n_epochs["ctx"])
            self.epochs["ctx"] = self.ctx_bpr.best_epoch

    def features(self, u, c, t=None):
        d = self.d
        n = np.maximum(self.hptr[u + 1] - self.hptr[u], 1)
        lengths = self.hptr[u + 1] - self.hptr[u]
        W = pair_weights(None, self.hptr[u], lengths, self.hi, d.nI)
        cos_sum = self.sim.weighted_sum(W, c)
        in_hist = np.asarray(self.X[u, c]).ravel()
        f = {f"bpr{j}": bpr_scores(m, u, c) for j, m in enumerate(self.bpr)}
        f.update(als=self.als.predict(u, c), cos_sum=cos_sum, cos_mean=cos_sum / n,
                 in_hist=in_hist, logpop=np.log1p(self.pop[c]))
        # the user's most similar past books
        nP = len(u)
        rep = np.repeat(np.arange(nP), lengths)
        h_item = self.hi[concat_ranges(self.hptr[u], lengths)]
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        cs = self.sim.entry_sims(c, rep, h_item)
        co = entry_products(self.XcT, self.Xc, c, rep, h_item)
        jac = co / np.maximum(self.pop[h_item] + self.pop[c[rep]] - co, 1)
        for name, v in (("cos", cs), ("jac", jac)):
            mx = np.zeros(nP)
            np.maximum.at(mx, rep, v)
            o = np.lexsort((-v, rep))
            rank = np.empty(len(rep), np.int64)
            rank[o] = np.arange(len(rep)) - starts[rep[o]]
            f[f"max_{name}"] = mx
            f[f"top5_{name}"] = np.bincount(rep, v * (rank < 5), nP) / np.minimum(n, 5)
        mc = np.zeros(nP)
        np.maximum.at(mc, rep, co)
        f["max_co"] = np.log1p(mc)
        f["n_overlap"] = np.log1p(np.bincount(rep, (co > 0).astype(float), nP))
        f["frac_overlap"] = np.bincount(rep, (co > 0).astype(float), nP) / n
        if self.use_time:
            tq = np.clip(t, 0, 2e9).astype(np.int64)
            base = c.astype(np.int64) * 4_000_000_000
            # how active the book was around that date
            for name, w in (("7d", 7 * 86400), ("30d", 30 * 86400), ("6m", 0.5 * YEAR), ("2y", 2 * YEAR)):
                lo = np.searchsorted(self.it_key, base + np.maximum(tq - int(w), 0))
                hi = np.searchsorted(self.it_key, base + tq + int(w))
                f[f"dens_{name}"] = np.log((hi - lo + 1) / (self.pop[c] + 2))
            # how much of the book's reading happened before that date
            before = np.searchsorted(self.it_key, base + tq) - self.iptr[c]
            f["item_cdf"] = (before + 0.5) / (self.pop[c] + 1)
            first = np.nan_to_num(self.item_first[c], nan=2e9)
            last = np.nan_to_num(self.item_last[c], nan=0)
            f["before_first"] = np.log1p(np.maximum(first - tq, 0) / 86400)
            f["after_last"] = np.log1p(np.maximum(tq - last, 0) / 86400)
            # similarity to what the user read around then
            pos = concat_ranges(self.hptr[u], lengths)
            trep = np.repeat(tq, lengths)
            for name, tau in (("30d", 30 * 86400), ("6m", 0.5 * YEAR)):
                wts = np.exp(-np.abs(self.ht[pos] - trep) / tau).astype(np.float32)
                Wt = pair_weights(None, self.hptr[u], lengths, self.hi, d.nI, weights=wts)
                f[f"cos_time_{name}"] = self.sim.weighted_sum(Wt, c)
            # the books the user read just before / just after t
            p = np.searchsorted(self.hkey, u.astype(np.int64) * 4_000_000_000 + tq)
            for name, q, ok in (("prev", p - 1, p - 1 >= self.hptr[u]), ("next", p, p < self.hptr[u + 1])):
                nb = np.where(ok, self.hi[np.clip(q, 0, len(self.hi) - 1)], d.nI)  # nI = none
                cnt = np.asarray(self.trans[nb, c]).ravel()
                f[f"trans_{name}"] = np.log1p(cnt)
                f[f"trans_{name}_norm"] = cnt / (self.trans_deg[nb] + 5)
                W1 = sp.csr_matrix((ok.astype(np.float32), (np.arange(len(u)), nb)), shape=(len(u), d.nI + 1))
                f[f"cos_{name}"] = self.sim.weighted_sum(W1, c)
            # how often the nearby books are read together with the candidate
            lo_u, hi_u = self.hptr[u], self.hptr[u + 1]
            for K in (3, 10):
                tot, totn = np.zeros(len(u)), np.zeros(len(u))
                for k in range(1, K + 1):
                    for q, ok in ((p - k, p - k >= lo_u), (p + k - 1, p + k - 1 < hi_u)):
                        nb = np.where(ok, self.hi[np.clip(q, 0, len(self.hi) - 1)], d.nI)
                        cnt = np.asarray(self.win[nb, c]).ravel()
                        tot += np.log1p(cnt) / k
                        totn += cnt / (self.win_deg[nb] + 5) / k
                f[f"win_trans{K}"], f[f"win_trans{K}_norm"] = tot, totn
            f["ctx_bpr"] = self.ctx_bpr.scores(u, c, t)
        self.names = list(f)
        return np.column_stack([f[k] for k in self.names]).astype(np.float64)


class AntiMLP(nn.Module):
    """Compares the two candidates; swapping them flips the answer."""

    def __init__(self, d, h=64, drop=0.1):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(2 * d + 1, h), nn.ReLU(), nn.Dropout(drop),
                               nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, a, b, c):
        return (self.g(torch.cat([a, b, c], 1)) - self.g(torch.cat([b, a, c], 1)))[:, 0]


class PairwiseMLP:
    """Small neural network that picks between two candidates (average of a few seeds)."""

    def __init__(self, hidden=64, epochs=20, seeds=3, wd=1e-4):
        self.hidden, self.epochs, self.seeds, self.wd = hidden, epochs, seeds, wd

    def _t(self, x):
        return torch.tensor(x, dtype=torch.float32, device=DEVICE)

    def fit(self, F1, F2, H, y):
        both = np.vstack([F1, F2])
        self.mu, self.sd = both.mean(0), both.std(0) + 1e-9
        a, b = self._t((F1 - self.mu) / self.sd), self._t((F2 - self.mu) / self.sd)
        c, yy = self._t(H[:, None]), self._t(y)
        self.models = []
        for seed in range(self.seeds):
            set_seed(seed)
            m = AntiMLP(F1.shape[1], self.hidden).to(DEVICE)
            opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=self.wd)
            for _ in range(self.epochs):
                m.train()
                perm = torch.randperm(len(a), device=DEVICE)
                for s in range(0, len(a), 512):
                    i = perm[s:s + 512]
                    loss = F.binary_cross_entropy_with_logits(m(a[i], b[i], c[i]), yy[i])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            self.models.append(m.eval())
        return self

    def decision(self, F1, F2, H):
        a, b = self._t((F1 - self.mu) / self.sd), self._t((F2 - self.mu) / self.sd)
        c = self._t(H[:, None])
        with torch.no_grad():
            return np.mean([m(a, b, c).cpu().numpy() for m in self.models], 0)


def sample_val_pairs(d, tr, va, seed):
    rng = np.random.default_rng(seed)
    neg = d.i[tr][rng.integers(0, len(tr), len(va))]
    return d.u[va], d.i[va], neg, d.t[va]


def train(train_file, use_time=False, n_val=50000, seed=0, n_neg=1):
    """Tune on a validation split, fit the combiner, then retrain on all data.
    n_neg: wrong books per real one (more training pairs for the combiner)."""
    set_seed(seed)
    d = Data(train_file)
    tr, va = holdout(d, n_val, seed)
    u, pos, neg, t = sample_val_pairs(d, tr, va, seed)
    log(f"train {len(tr)}  validation positives {len(va)} x {n_neg} negatives  (time features: {use_time})")
    ctx = Context(d, tr, use_time, val=dict(u=u, pos=pos, neg=neg, t=t))
    Fp = ctx.features(u, pos, t)
    Fns = [ctx.features(u, neg, t)] + [ctx.features(u, sample_val_pairs(d, tr, va, seed + 100 + k)[2], t)
                                       for k in range(1, n_neg)]
    names = ctx.names
    for j, nme in enumerate(names):
        acc = ((Fp[:, j] > Fns[0][:, j]) + 0.5 * (Fp[:, j] == Fns[0][:, j])).mean()
        log(f"  single feature {nme:12s} val acc {acc:.4f}")
    group = np.tile(np.arange(len(u)), n_neg)    # which positive each pair belongs to
    Fp, Fn = Fp[group], np.vstack(Fns)
    # random item1/item2 order, as in the test file
    swap = np.random.default_rng(seed + 1).random(len(group)) < 0.5
    F1 = np.where(swap[:, None], Fn, Fp)
    F2 = np.where(swap[:, None], Fp, Fn)
    y = (~swap).astype(float)  # 1 if item1 is the positive
    H = np.log1p(ctx.hptr[u + 1] - ctx.hptr[u])[group]  # user history length
    # split by real book, so the same one is never on both sides
    half = (np.random.default_rng(seed + 2).permutation(len(u)) < len(u) // 2)[group]
    lr = PairwiseLR().fit(F1[half] - F2[half], y[half])
    acc = ((lr.decision(F1[~half] - F2[~half]) > 0) == (y[~half] == 1)).mean()
    log(f"logistic-regression combiner val accuracy (held-out half): {acc:.4f}")
    mlp = PairwiseMLP().fit(F1[half], F2[half], H[half], y[half])
    acc = ((mlp.decision(F1[~half], F2[~half], H[~half]) > 0) == (y[~half] == 1)).mean()
    log(f"ENSEMBLE (MLP combiner) val accuracy (held-out half): {acc:.4f}")
    mlp = PairwiseMLP().fit(F1, F2, H, y)
    epochs = ctx.epochs
    del ctx
    log("retraining on all training data")
    full = Context(d, np.arange(d.n), use_time, n_epochs=epochs)
    return dict(ctx=full, combiner=mlp)


def predict(bundle, test_file, out_file, use_time=False):
    header, rows = read_csv(test_file)
    ctx = bundle["ctx"]
    d = ctx.d
    col = {h: k for k, h in enumerate(header)}
    u = d.enc_u([r[col["user"]] for r in rows])
    c1 = d.enc_i([r[col["item1"]] for r in rows])
    c2 = d.enc_i([r[col["item2"]] for r in rows])
    t = np.array([float(r[col["timestamp"]]) for r in rows]) if use_time else None
    H = np.log1p(ctx.hptr[u + 1] - ctx.hptr[u])
    s = bundle["combiner"].decision(ctx.features(u, c1, t), ctx.features(u, c2, t), H)
    n_in = len(header) - 1
    out = [r[:n_in] + [r[col["item1"]] if x >= 0 else r[col["item2"]]] for r, x in zip(rows, s)]
    write_csv(out_file, header, out)


if __name__ == "__main__":
    bundle = train("tasks_1_2_train.csv")
    predict(bundle, "task_2_test.csv", "task_2_predictions.csv")
