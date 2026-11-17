"""Task 5: which of two books does the user read next?

Validation hides each user's last book. Clues: the user's taste (BPR), books often read
right after their last few books, similarity to recent and all past reading, popularity
(overall and recent), and the time-aware model from task 2/4. A small neural network
(task2.PairwiseMLP) picks between the two books.

Run:  python task5.py
"""
import numpy as np
import scipy.sparse as sp

from recsys_lib import (CosineSim, Data, PairwiseLR, bpr_scores, concat_ranges, entry_products, fit_bpr,
                        interaction_matrix, log, pair_weights, read_csv, set_seed, user_ptr, write_csv)
from task2 import CtxBPR, PairwiseMLP

BPR_CFG = dict(k=64, lam=1e-3)
FPMC_CFG = dict(k=64, lam=1e-3)
EPOCHS = 30


class SeqContext:
    def __init__(self, d, idx, val=None, epochs=None):
        """Everything the features need, fitted on the training rows idx (sorted by user and time)."""
        self.d = d
        nI = d.nI
        u, i, t = d.u[idx], d.i[idx], d.t[idx]
        self.u_rows, self.i_rows = u, i
        self.ptr = user_ptr(u, d.nU + 1)
        self.X = interaction_matrix(u, i, d.nU, d.nI)
        self.sim = CosineSim(self.X)
        self.Xc, self.XcT = self.X.tocsc(), self.X.T.tocsr()
        # used by the time-aware model
        self.hu, self.hi, self.ht, self.hptr = u, i, np.clip(t, 0, 2e9), self.ptr
        self.pop = np.bincount(i, minlength=nI + 1)
        # how often readers go from one book straight to another
        same = u[1:] == u[:-1]
        a, b = i[:-1][same], i[1:][same]
        self.T = sp.csr_matrix((np.ones(len(a), np.float32), (a, b)), shape=(nI + 1, nI + 1))
        self.outdeg = np.asarray(self.T.sum(1)).ravel()
        # recent popularity
        tc = np.clip(t, 0, 2e9)
        cut = np.quantile(tc, 0.9)
        self.recent = np.bincount(i[tc >= cut], minlength=nI + 1)
        last_k = np.concatenate([np.arange(max(s, e - 3), e) for s, e in
                                 zip(self.ptr[:-1], self.ptr[1:]) if e > s])
        self.endpop = np.bincount(i[last_k], minlength=nI + 1)
        self.last = np.full(d.nU + 1, nI)
        has = self.ptr[1:] > self.ptr[:-1]
        self.last[:-1][has[:-1]] = i[self.ptr[1:][:-1][has[:-1]] - 1]

        self.epochs = {}
        prev_val = None if val is None else dict(val, prev=self.last[val["u"]])
        ep = EPOCHS if epochs is None else epochs["bpr"]
        self.bpr, self.epochs["bpr"], acc = fit_bpr(u, i, d.nU, d.nI, BPR_CFG["k"], BPR_CFG["lam"],
                                                    epochs=ep, neg_pool=i, val=val, verbose=False)
        if val is not None:
            log(f"  BPR val acc {acc:.4f} (epoch {self.epochs['bpr']})")
        ep = EPOCHS if epochs is None else epochs["fpmc"]
        self.fpmc, self.epochs["fpmc"], acc = fit_bpr(
            u[1:][same], i[1:][same], d.nU, d.nI, FPMC_CFG["k"], FPMC_CFG["lam"], epochs=ep,
            neg_pool=i, prev=i[:-1][same], val=prev_val, verbose=False, seed=1)
        if val is not None:
            log(f"  FPMC val acc {acc:.4f} (epoch {self.epochs['fpmc']})")
        self.ctx_bpr = CtxBPR(self, val=val, epochs=None if epochs is None else epochs["ctx"], mode="next")
        self.epochs["ctx"] = self.ctx_bpr.best_epoch

    def _hist_weights(self, u, last_n=None, decay=None):
        s, e = self.ptr[u], self.ptr[u + 1]
        if last_n is not None:
            s = np.maximum(s, e - last_n)
        lengths = e - s
        w = None
        if decay is not None:  # recent books count more
            from recsys_lib import concat_ranges
            pos = concat_ranges(s, lengths)
            dist = np.repeat(e, lengths) - 1 - pos
            w = (decay ** dist).astype(np.float32)
        return pair_weights(None, s, lengths, self.i_rows, self.d.nI, w), np.maximum(lengths, 1)

    def features(self, u, c):
        last = self.last[u]
        f = {"bpr": bpr_scores(self.bpr, u, c), "fpmc": bpr_scores(self.fpmc, u, c, last)}
        tl = np.asarray(self.T[last, c]).ravel()
        f["trans_last_log"] = np.log1p(tl)
        f["trans_last_norm"] = tl / (self.outdeg[last] + 5)
        W3, _ = self._hist_weights(u, last_n=3, decay=0.5)
        Tn = (sp.diags(1 / (self.outdeg + 5)) @ self.T).tocsc()  # row-normalised transitions
        f["trans_last3"] = np.asarray(W3.multiply(Tn[:, c].T.tocsr()).sum(1)).ravel()
        W1, _ = self._hist_weights(u, last_n=1)
        f["cos_last"] = self.sim.weighted_sum(W1, c)
        W5, _ = self._hist_weights(u, last_n=5, decay=0.8)
        f["cos_last5"] = self.sim.weighted_sum(W5, c)
        Wall, n = self._hist_weights(u)
        cs = self.sim.weighted_sum(Wall, c)
        f["cos_sum"], f["cos_mean"] = cs, cs / n
        f["in_hist"] = np.asarray(self.X[u, c]).ravel()
        f["logpop"] = np.log1p(self.pop[c])
        f["recent_pop"] = np.log((self.recent[c] + 1) / (self.pop[c] + 10))
        f["end_pop"] = np.log((self.endpop[c] + 1) / (self.pop[c] + 10))
        f["ctx_bpr"] = self.ctx_bpr.scores(u, c)
        # the user's most similar past books (all / last 10)
        nP = len(u)
        e = self.ptr[u + 1]
        for name, s0 in (("all", self.ptr[u]), ("last10", np.maximum(self.ptr[u], e - 10))):
            L = e - s0
            rep = np.repeat(np.arange(nP), L)
            h_item = self.i_rows[concat_ranges(s0, L)]
            starts = np.r_[0, np.cumsum(L)[:-1]]
            cs = self.sim.entry_sims(c, rep, h_item)
            vals = [("cos", cs)]
            if name == "all":
                co = entry_products(self.XcT, self.Xc, c, rep, h_item)
                vals.append(("jac", co / np.maximum(self.pop[h_item] + self.pop[c[rep]] - co, 1)))
            for vn, v in vals:
                mx = np.zeros(nP)
                np.maximum.at(mx, rep, v)
                o = np.lexsort((-v, rep))
                rank = np.empty(len(rep), np.int64)
                rank[o] = np.arange(len(rep)) - starts[rep[o]]
                f[f"max_{vn}_{name}"] = mx
                f[f"top5_{vn}_{name}"] = np.bincount(rep, v * (rank < 5), nP) / np.clip(L, 1, 5)
        self.names = list(f)
        return np.column_stack([f[k] for k in self.names]).astype(np.float64)


def train(train_file="task_5_train.csv", n_val=60000, seed=0, n_neg=3, skip_zero=False):
    """Tune on each user's last book, fit the combiner, then retrain on all data.
    Rating-0 rows are kept (they are still reads)."""
    set_seed(seed)
    d = Data(train_file, skip_zero=skip_zero)
    # rows are grouped by user & time-sorted: hold out each user's last row
    ptr = user_ptr(d.u, d.nU + 1)  # works because rows are grouped by user
    assert (np.diff(d.u) != 0).sum() == d.nU - 1, "expected rows grouped by user"
    last_rows = ptr[1:][:-1] - 1
    mask = np.ones(d.n, bool)
    mask[last_rows] = False
    tr = np.where(mask)[0]
    rng = np.random.default_rng(seed)
    va = rng.choice(last_rows, min(n_val, len(last_rows)), replace=False)
    va = va[np.isin(d.i[va], np.unique(d.i[tr]))]  # keep pairs with a known positive
    u, pos = d.u[va], d.i[va]
    neg = d.i[tr][rng.integers(0, len(tr), len(va))]
    log(f"train rows {len(tr)}  validation pairs {len(va)}")
    ctx = SeqContext(d, tr, val=dict(u=u, pos=pos, neg=neg))
    Fp = ctx.features(u, pos)
    Fns = [ctx.features(u, neg)] + [ctx.features(u, d.i[tr][rng.integers(0, len(tr), len(va))])
                                    for _ in range(1, n_neg)]
    for j, nme in enumerate(ctx.names):
        acc = ((Fp[:, j] > Fns[0][:, j]) + 0.5 * (Fp[:, j] == Fns[0][:, j])).mean()
        log(f"  single feature {nme:16s} val acc {acc:.4f}")
    group = np.tile(np.arange(len(u)), n_neg)
    Fp, Fn = Fp[group], np.vstack(Fns)
    swap = rng.random(len(group)) < 0.5
    F1, F2 = np.where(swap[:, None], Fn, Fp), np.where(swap[:, None], Fp, Fn)
    y = (~swap).astype(float)
    H = np.log1p(ctx.ptr[u + 1] - ctx.ptr[u])[group]
    half = (rng.permutation(len(u)) < len(u) // 2)[group]   # split by positive
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
    return dict(ctx=SeqContext(d, np.arange(d.n), epochs=epochs), combiner=mlp)


def predict(bundle, test_file="task_5_test.csv", out_file="task_5_predictions.csv"):
    header, rows = read_csv(test_file)
    ctx = bundle["ctx"]
    d = ctx.d
    col = {h: k for k, h in enumerate(header)}
    u = d.enc_u([r[col["user"]] for r in rows])
    c1 = d.enc_i([r[col["item1"]] for r in rows])
    c2 = d.enc_i([r[col["item2"]] for r in rows])
    H = np.log1p(ctx.ptr[u + 1] - ctx.ptr[u])
    s = bundle["combiner"].decision(ctx.features(u, c1), ctx.features(u, c2), H)
    n_in = len(header) - 1
    out = [r[:n_in] + [r[col["item1"]] if x >= 0 else r[col["item2"]]] for r, x in zip(rows, s)]
    write_csv(out_file, header, out)


if __name__ == "__main__":
    predict(train())
