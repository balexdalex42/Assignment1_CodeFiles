"""Shared code used by every task: loading data, the rating and ranking models,
book-similarity helpers, and the models that combine everything (GPU when available)."""
import csv
import math
import random
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YEAR = 365.25 * 86400


def set_seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


#I/O stuf
def read_csv(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r if row]
    return header, rows


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    log(f"wrote {len(rows)} rows to {path}")



class Data:
    #encoding the data into arrays

    def __init__(self, path, skip_zero=True):
        _, rows = read_csv(path)
        if skip_zero:
            rows = [r for r in rows if r[3] != "0"]
        self.uid, self.iid = {}, {}
        for r in rows:
            self.uid.setdefault(r[0], len(self.uid))
            self.iid.setdefault(r[1], len(self.iid))
        self.nU, self.nI = len(self.uid), len(self.iid)
        self.u = np.fromiter((self.uid[r[0]] for r in rows), np.int64, len(rows))
        self.i = np.fromiter((self.iid[r[1]] for r in rows), np.int64, len(rows))
        self.t = np.fromiter((int(r[2]) for r in rows), np.float64, len(rows))
        self.r = np.fromiter((float(r[3]) for r in rows), np.float32, len(rows))
        self.n = len(rows)

    def enc_u(self, keys):
        return np.array([self.uid.get(k, self.nU) for k in keys], np.int64)

    def enc_i(self, keys):
        return np.array([self.iid.get(k, self.nI) for k in keys], np.int64)


def holdout(data, n_val, seed=0):
    """Hide random interactions for validation (only ones whose user and book still appear in training)."""
    rng = np.random.default_rng(seed)
    cand = rng.choice(data.n, size=int(n_val * 1.1), replace=False)
    mask = np.ones(data.n, bool)
    mask[cand] = False
    ucnt = np.bincount(data.u[mask], minlength=data.nU + 1)
    icnt = np.bincount(data.i[mask], minlength=data.nI + 1)
    cand = cand[(ucnt[data.u[cand]] > 0) & (icnt[data.i[cand]] > 0)][:n_val]
    mask = np.ones(data.n, bool)
    mask[cand] = False
    return np.where(mask)[0], cand


def user_ptr(u, n_users):
    """Where each user's rows start and end (rows sorted by user)."""
    return np.concatenate([[0], np.cumsum(np.bincount(u, minlength=n_users))])


def concat_ranges(starts, lengths):
    starts = np.asarray(starts, np.int64)
    lengths = np.asarray(lengths, np.int64)
    offs = np.repeat(starts - np.cumsum(lengths) + lengths, lengths)
    return offs + np.arange(lengths.sum())

# Latent factor rating model
class RatingModel(nn.Module):
    """rating = average + user bias + book bias + user/book taste match (+ optional time effects)"""

    def __init__(self, nU, nI, k, mu, n_bins=0):
        super().__init__()
        self.mu, self.k, self.n_bins, self.nI = mu, k, n_bins, nI
        self.bu = nn.Embedding(nU + 1, 1)
        self.bi = nn.Embedding(nI + 1, 1)
        nn.init.zeros_(self.bu.weight)
        nn.init.zeros_(self.bi.weight)
        if k:
            self.P = nn.Embedding(nU + 1, k)
            self.Q = nn.Embedding(nI + 1, k)
            nn.init.normal_(self.P.weight, std=0.01)
            nn.init.normal_(self.Q.weight, std=0.01)
            with torch.no_grad():  # "unknown" rows stay at zero
                self.P.weight[nU].zero_()
                self.Q.weight[nI].zero_()
        if n_bins:
            self.bb = nn.Embedding(n_bins, 1)
            self.bib = nn.Embedding((nI + 1) * n_bins, 1)
            self.au = nn.Embedding(nU + 1, 1)
            for e in (self.bb, self.bib, self.au):
                nn.init.zeros_(e.weight)

    def forward(self, u, i, b=None, dev=None):
        bu, bi = self.bu(u)[:, 0], self.bi(i)[:, 0]
        out = self.mu + bu + bi
        breg = bu ** 2 + bi ** 2
        freg = torch.zeros_like(out)
        if self.k:
            p, q = self.P(u), self.Q(i)
            out = out + (p * q).sum(1)
            freg = (p ** 2).sum(1) + (q ** 2).sum(1)
        if self.n_bins:
            bib = self.bib(i * self.n_bins + b)[:, 0]
            au = self.au(u)[:, 0]
            out = out + self.bb(b)[:, 0] + bib + au * dev
            breg = breg + bib ** 2 + au ** 2
        return out, breg, freg


def _tensors(*arrs):
    out = []
    for a in arrs:
        if a is None:
            out.append(None)
        elif np.issubdtype(np.asarray(a).dtype, np.integer):
            out.append(torch.as_tensor(np.asarray(a), dtype=torch.long, device=DEVICE))
        else:
            out.append(torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEVICE))
    return out


def predict_rating(model, u, i, b=None, dev=None, bs=65536):
    model.eval()
    u, i, b, dev = _tensors(u, i, b, dev)
    outs = []
    with torch.no_grad():
        for s in range(0, len(u), bs):
            sl = slice(s, s + bs)
            o, _, _ = model(u[sl], i[sl], None if b is None else b[sl],
                            None if dev is None else dev[sl])
            outs.append(o)
    return torch.cat(outs).clamp(0, 5).cpu().numpy()


def fit_rating(u, i, r, nU, nI, k, lam_f, lam_b, epochs=40, lr=3e-3, bs=4096,
               val=None, n_bins=0, b=None, dev=None, patience=3, seed=0, verbose=True):
    """Train the rating model with Adam, stopping when validation stops improving."""
    set_seed(seed)
    model = RatingModel(nU, nI, k, float(r.mean()), n_bins).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    U, I, R, B, D = _tensors(u, i, r, b, dev)
    n = len(U)
    best, best_ep, best_state, bad = np.inf, epochs, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            pred, breg, freg = model(U[idx], I[idx], None if B is None else B[idx],
                                     None if D is None else D[idx])
            loss = ((pred - R[idx]) ** 2).mean() + lam_f * freg.mean() + lam_b * breg.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val is not None:
            p = predict_rating(model, val["u"], val["i"], val.get("b"), val.get("dev"))
            mse = float(((p - val["r"]) ** 2).mean())
            if verbose:
                log(f"  k={k} lam_f={lam_f} lam_b={lam_b} epoch {ep}: val MSE {mse:.4f}")
            if mse < best - 1e-5:
                best, best_ep, bad = mse, ep, 0
                best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, best


def _ridge_side(idx, X, y, n, lam_vec):
    # one small regularised least-squares solve per user (or per book)
    d = X.shape[1]
    chunk = max(2000, int(4e7 / (d * d)))
    A = torch.zeros(n, d, d, device=DEVICE)
    bvec = torch.zeros(n, d, device=DEVICE)
    for s in range(0, len(idx), chunk):
        x = X[s:s + chunk]
        A.index_add_(0, idx[s:s + chunk], x[:, :, None] * x[:, None, :])
        bvec.index_add_(0, idx[s:s + chunk], x * y[s:s + chunk, None])
    A += torch.diag(lam_vec)
    return torch.linalg.solve(A, bvec)


class ALSModel:
    """Same rating model, fitted by alternating least squares: solve all users, then all
    books (then the time terms), and repeat."""

    def __init__(self, nU, nI, k, lam_f, lam_b, n_bins=0, lam_t=(10.0, 25.0, 50.0)):
        self.nU, self.nI, self.k, self.n_bins = nU, nI, k, n_bins
        self.lam_f, self.lam_b, self.lam_t = lam_f, lam_b, lam_t

    def predict(self, u, i, b=None, dev=None):
        u, i, b, dev = _tensors(u, i, b, dev)
        out = self.mu + self.bu[u] + self.bi[i] + (self.P[u] * self.Q[i]).sum(1)
        if self.n_bins:
            out = out + self.bb[b] + self.bib[i * self.n_bins + b] + self.au[u] * dev
        return out.clamp(0, 5).cpu().numpy()

    def fit(self, u, i, r, iters=10, b=None, dev=None, val=None, verbose=True, seed=0):
        set_seed(seed)
        U, I, R, B, D = _tensors(u, i, r, b, dev)
        nU, nI, k = self.nU + 1, self.nI + 1, self.k
        self.mu = float(R.mean())
        self.bu = torch.zeros(nU, device=DEVICE)
        self.bi = torch.zeros(nI, device=DEVICE)
        self.P = torch.randn(nU, k, device=DEVICE) * 0.01
        self.Q = torch.randn(nI, k, device=DEVICE) * 0.01
        self.P[-1] = 0
        self.Q[-1] = 0
        tb = torch.zeros(len(U), device=DEVICE)
        if self.n_bins:
            self.bb = torch.zeros(self.n_bins, device=DEVICE)
            self.bib = torch.zeros(nI * self.n_bins, device=DEVICE)
            self.au = torch.zeros(nU, device=DEVICE)
            IB = I * self.n_bins + B
        lam_u = torch.tensor([self.lam_f] * k + [self.lam_b], device=DEVICE)
        ones = torch.ones(len(U), 1, device=DEVICE)
        hist, best, best_it, best_state = [], np.inf, iters, None

        def scatter_mean(idx, res, n, lam):
            s = torch.zeros(n, device=DEVICE).index_add_(0, idx, res)
            c = torch.zeros(n, device=DEVICE).index_add_(0, idx, torch.ones_like(res))
            return s / (c + lam)

        for it in range(1, iters + 1):
            # update users
            y = R - self.mu - self.bi[I] - tb
            w = _ridge_side(U, torch.cat([self.Q[I], ones], 1), y, nU, lam_u)
            self.P, self.bu = w[:, :k].contiguous(), w[:, k].contiguous()
            # update books
            y = R - self.mu - self.bu[U] - tb
            w = _ridge_side(I, torch.cat([self.P[U], ones], 1), y, nI, lam_u)
            self.Q, self.bi = w[:, :k].contiguous(), w[:, k].contiguous()
            if self.n_bins:
                base = self.mu + self.bu[U] + self.bi[I] + (self.P[U] * self.Q[I]).sum(1)
                lb, lib, la = self.lam_t
                res = R - base - self.bib[IB] - self.au[U] * D
                self.bb = scatter_mean(B, res, self.n_bins, lb)
                res = R - base - self.bb[B] - self.au[U] * D
                self.bib = scatter_mean(IB, res, nI * self.n_bins, lib)
                res = R - base - self.bb[B] - self.bib[IB]
                num = torch.zeros(nU, device=DEVICE).index_add_(0, U, res * D)
                den = torch.zeros(nU, device=DEVICE).index_add_(0, U, D * D)
                self.au = num / (den + la)
                tb = self.bb[B] + self.bib[IB] + self.au[U] * D
            if val is not None:
                p = self.predict(val["u"], val["i"], val.get("b"), val.get("dev"))
                mse = float(((p - val["r"]) ** 2).mean())
                hist.append(mse)
                if verbose:
                    log(f"  ALS k={k} lam_f={self.lam_f} lam_b={self.lam_b} iter {it}: val MSE {mse:.4f}")
                if mse < best - 1e-5:
                    best, best_it = mse, it
        self.best_iter, self.best_mse = best_it, best
        return self


# ----------------------------------------------------------------------------
# BPR ranking model (optionally also using the previous book)
# ----------------------------------------------------------------------------
class BPRModel(nn.Module):
    """score = book bias + user/book match (+ previous book/this book match)"""

    def __init__(self, nU, nI, k, seq=False):
        super().__init__()
        self.seq = seq
        self.U = nn.Embedding(nU + 1, k)
        self.V = nn.Embedding(nI + 1, k)
        self.b = nn.Embedding(nI + 1, 1)
        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.V.weight, std=0.01)
        nn.init.zeros_(self.b.weight)
        if seq:
            self.W = nn.Embedding(nI + 1, k)
            self.Z = nn.Embedding(nI + 1, k)
            nn.init.normal_(self.W.weight, std=0.01)
            nn.init.normal_(self.Z.weight, std=0.01)
        with torch.no_grad():  # unknown user / item rows
            self.U.weight[nU].zero_()
            self.V.weight[nI].zero_()
            if seq:
                self.W.weight[nI].zero_()
                self.Z.weight[nI].zero_()

    def score(self, u, c, prev=None):
        uu, vc = self.U(u), self.V(c)
        s = self.b(c)[:, 0] + (uu * vc).sum(1)
        reg = (uu ** 2).sum(1) + (vc ** 2).sum(1)
        if self.seq:
            w, z = self.W(prev), self.Z(c)
            s = s + (w * z).sum(1)
            reg = reg + (w ** 2).sum(1) + (z ** 2).sum(1)
        return s, reg


def bpr_scores(model, u, c, prev=None, bs=65536):
    model.eval()
    u, c, prev = _tensors(u, c, prev)
    outs = []
    with torch.no_grad():
        for s in range(0, len(u), bs):
            sl = slice(s, s + bs)
            o, _ = model.score(u[sl], c[sl], None if prev is None else prev[sl])
            outs.append(o)
    return torch.cat(outs).cpu().numpy()


def fit_bpr(u, i, nU, nI, k, lam, epochs=30, lr=2e-3, bs=4096, neg_pool=None,
            prev=None, val=None, patience=3, seed=0, verbose=True):
    """Train BPR: score books the user read above random books (drawn by popularity, like the test)."""
    set_seed(seed)
    seq = prev is not None
    model = BPRModel(nU, nI, k, seq).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    U, I, PV = _tensors(u, i, prev)
    pool = torch.as_tensor(neg_pool if neg_pool is not None else i, dtype=torch.long, device=DEVICE)
    n = len(U)
    best, best_ep, best_state, bad = -1, epochs, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            j = pool[torch.randint(len(pool), (len(idx),), device=DEVICE)]
            pv = None if PV is None else PV[idx]
            sp_, rp = model.score(U[idx], I[idx], pv)
            sn, rn = model.score(U[idx], j, pv)
            loss = -F.logsigmoid(sp_ - sn).mean() + lam * (rp + rn).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val is not None:
            a = bpr_scores(model, val["u"], val["pos"], val.get("prev"))
            bneg = bpr_scores(model, val["u"], val["neg"], val.get("prev"))
            acc = float(((a > bneg) + 0.5 * (a == bneg)).mean())
            if verbose:
                log(f"  BPR k={k} lam={lam} seq={seq} epoch {ep}: val acc {acc:.4f}")
            if acc > best + 1e-4:
                best, best_ep, bad = acc, ep, 0
                best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, best


# ----------------------------------------------------------------------------
# Book similarity
# ----------------------------------------------------------------------------
def interaction_matrix(u, i, nU, nI):
    X = sp.csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(nU + 1, nI + 1))
    X.data[:] = 1.0  # binary
    return X


class CosineSim:
    """How similar two books are, based on how many readers they share."""

    def __init__(self, X):
        deg = np.asarray(X.sum(0)).ravel()
        inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1)), 0.0).astype(np.float32)
        self.Xn = (X @ sp.diags(inv)).tocsc()
        self.XnT = self.Xn.T.tocsr()

    def weighted_sum(self, W, cands, chunk=1500, power=1.0):
        """For each (user, candidate) pair: weighted sum of the candidate's similarity to the
        user's past books (power > 1 favours the closest books)."""
        W = W.tocsr()
        cands = np.asarray(cands)
        out = np.zeros(len(cands), np.float64)
        uniq, inv = np.unique(cands, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(0, len(uniq) + chunk, chunk))
        for c0 in range(0, len(uniq), chunk):
            cols = uniq[c0:c0 + chunk]
            M = (self.XnT @ self.Xn[:, cols]).tocsc()
            if power != 1.0:
                M.data **= power
            k = c0 // chunk
            pidx = order[bounds[k]:bounds[k + 1]]
            if len(pidx) == 0:
                continue
            for b0 in range(0, len(pidx), 500):  # small blocks to save memory
                pb = pidx[b0:b0 + 500]
                Mp = M[:, inv[pb] - c0].T.tocsr()
                out[pb] = np.asarray(W[pb].multiply(Mp).sum(1)).ravel()
        # ignore the candidate's similarity to itself
        self_w = np.asarray(W[np.arange(len(cands)), cands]).ravel()
        return out - self_w

    def entry_sims(self, cands, rep, h_item, chunk=1000):
        """Similarity between each past book and the candidate it is compared to."""
        return entry_products(self.XnT, self.Xn, cands, rep, h_item, chunk)


def entry_products(AT, A, cands, rep, h_item, chunk=1000):
    """Shared readers (or similarity) between each past book and its candidate."""
    cands = np.asarray(cands)
    out = np.zeros(len(rep), np.float32)
    uniq, inv = np.unique(cands, return_inverse=True)
    inv_e = inv[rep]
    order = np.argsort(inv_e, kind="stable")
    bounds = np.searchsorted(inv_e[order], np.arange(0, len(uniq) + chunk, chunk))
    for c0 in range(0, len(uniq), chunk):
        e = order[bounds[c0 // chunk]:bounds[c0 // chunk + 1]]
        if len(e) == 0:
            continue
        M = (AT @ A[:, uniq[c0:c0 + chunk]]).tocsc()
        for b0 in range(0, len(e), 2_000_000):
            eb = e[b0:b0 + 2_000_000]
            out[eb] = np.asarray(M[h_item[eb], inv_e[eb] - c0]).ravel()
    out[h_item == cands[rep]] = 0
    return out


def pair_weights(pair_rows, starts, lengths, items, nI, weights=None):
    """Sparse matrix with one row per pair, holding that user's past books."""
    pos = concat_ranges(starts, lengths)
    rows = np.repeat(np.arange(len(starts)), lengths)
    w = np.ones(len(pos), np.float32) if weights is None else weights
    return sp.csr_matrix((w, (rows, items[pos])), shape=(len(starts), nI + 1))


# ----------------------------------------------------------------------------
# Combining models
# ----------------------------------------------------------------------------
class PairwiseLR:
    """Logistic regression on the difference between the two candidates' features
    (swapping the candidates flips the answer)."""

    def __init__(self, l2=1e-2):
        self.l2 = l2

    def fit(self, D, y):
        self.scale = D.std(0) + 1e-9
        X = np.vstack([D, -D]) / self.scale
        Y = np.concatenate([y, 1 - y]).astype(np.float64)

        def f(w):
            z = X @ w
            p = 1 / (1 + np.exp(-z))
            ll = np.logaddexp(0, z) - Y * z
            return ll.mean() + self.l2 * w @ w, X.T @ (p - Y) / len(Y) + 2 * self.l2 * w

        self.w = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def decision(self, D):
        return (D / self.scale) @ self.w


def ridge_fit(X, y, l2=1e-3):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + l2 * len(X) * np.eye(Xb.shape[1])
    A[-1, -1] -= l2 * len(X)  # don't penalise intercept
    return np.linalg.solve(A, Xb.T @ y)


def ridge_predict(w, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


# ----------------------------------------------------------------- history-attention rating model
# Predicts a rating from the user's other ratings, focusing on the ones most relevant to this
# book (and, with timestamps, the ones closest in time).
N_GAP = 41  # number of time-gap buckets


def build_hist(u, i, r, t, nU, nI, L, rng):
    """Each user's past ratings as a fixed-size table (a random subset if there are too many)."""
    o = np.lexsort((rng.random(len(u)), u))
    us = u[o]
    start = np.searchsorted(us, np.arange(nU + 1))
    rank = np.arange(len(o)) - start[us]
    k = rank < L
    Hi = np.full((nU + 1, L), nI + 1, np.int64)
    Hr = np.zeros((nU + 1, L), np.int64)
    Ht = np.zeros((nU + 1, L), np.float32)
    Hi[us[k], rank[k]] = i[o][k]
    Hr[us[k], rank[k]] = r[o][k].astype(np.int64)
    Ht[us[k], rank[k]] = (t[o][k] / 86400.0).astype(np.float32)
    slot = np.full(len(u), -1, np.int64)
    slot[o[k]] = rank[k]
    T = lambda x: torch.as_tensor(x, device=DEVICE)
    return T(Hi), T(Hr), T(Ht), slot


class Net(nn.Module):
    """Attention over the user's past ratings -> predicted rating."""

    def __init__(self, nU, nI, d, mu, use_time):
        super().__init__()
        self.d, self.use_time = d, use_time
        self.Q = nn.Embedding(nI + 1, d)
        self.K = nn.Embedding(nI + 2, d)
        self.V = nn.Embedding(nI + 2, d)
        self.R = nn.Embedding(6, d)
        self.G = nn.Embedding(N_GAP, d)
        self.Gb = nn.Embedding(N_GAP, 1)
        self.bu = nn.Embedding(nU + 1, 1)
        self.bi = nn.Embedding(nI + 1, 1)
        for e in (self.Q, self.K, self.V, self.R, self.G):
            nn.init.normal_(e.weight, 0, 0.05)
        for e in (self.Gb, self.bu, self.bi):
            nn.init.zeros_(e.weight)
        self.mu = mu
        self.mlp = nn.Sequential(nn.Linear(4 * d + 2, 256), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, u, i, hi, hr, hg, mask):
        q = self.Q(i)
        k = self.K(hi) + self.G(hg)
        v = self.V(hi) + self.R(hr)
        logit = torch.einsum("bld,bd->bl", k, q) / math.sqrt(self.d) + self.Gb(hg)[..., 0]
        logit = logit.masked_fill(mask, -1e9)
        a = torch.softmax(logit, 1)
        att = torch.einsum("bl,bld->bd", a, v)
        m = (~mask).float()
        n = m.sum(1, keepdim=True)
        mean = torch.einsum("bl,bld->bd", m, v) / (n + 1)
        rmean = ((hr.float() - 3.8) * m).sum(1, keepdim=True) / (n + 2)
        h = torch.cat([q, att, mean, q * att, torch.log1p(n), rmean], 1)
        out = self.mu + self.bu(u)[:, 0] + self.bi(i)[:, 0] + self.mlp(h)[:, 0]
        reg = (q.pow(2).sum(1).mean() + (k.pow(2).sum(2) * m).sum() / m.sum().clamp(min=1)
               + (v.pow(2).sum(2) * m).sum() / m.sum().clamp(min=1)
               + self.bu(u).pow(2).mean() + self.bi(i).pow(2).mean())
        return out, reg


class HistNet:
    """Trains Net and predicts with it."""

    def __init__(self, dim=64, L=256, lam=1e-4, epochs=20, patience=3, bs=512, lr=1e-3,
                 drop_hist=0.1, seed=0):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def _batch(self, H, u, i, t, slot, train):
        Hi, Hr, Ht = H
        hi, hr = Hi[u], Hr[u]
        mask = hi == self.nI + 1
        if slot is not None:
            s = torch.as_tensor(slot, device=DEVICE)
            ok = s >= 0
            mask[ok.nonzero()[:, 0], s[ok]] = True
        if train and self.drop_hist > 0:
            mask = mask | (torch.rand(mask.shape, device=DEVICE) < self.drop_hist)
        if self.use_time:
            dt = (Ht[u] - torch.as_tensor(t / 86400.0, device=DEVICE, dtype=torch.float32)[:, None]) * 24
            b = torch.clamp(torch.log2(dt.abs() + 1).long(), max=19)
            hg = torch.where(dt < 0, 19 - b, 20 + b)
        else:
            hg = torch.full_like(hi, 40)
        return hi, hr, hg, mask

    def _predict(self, H, u, i, t, bs=2048):
        self.net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(u), bs):
                ub = torch.as_tensor(u[s:s + bs], device=DEVICE)
                ib = torch.as_tensor(i[s:s + bs], device=DEVICE)
                hi, hr, hg, mask = self._batch(H, ub, ib, t[s:s + bs] if t is not None else None, None, False)
                out.append(self.net(ub, ib, hi, hr, hg, mask)[0].clamp(1, 5).cpu().numpy())
        return np.concatenate(out)

    def fit(self, u, i, r, t, nU, nI, val=None, use_time=False):
        """Train; stops early on the validation set if one is given."""
        set_seed(self.seed)
        self.nU, self.nI, self.use_time = nU, nI, use_time
        self.data = (u, i, r, t)
        rng = np.random.default_rng(self.seed)
        self.net = Net(nU, nI, self.dim, float(r.mean()), use_time).to(DEVICE)
        opt = torch.optim.Adam(self.net.parameters(), self.lr)
        best, best_state, bad, self.best_epoch = 1e9, None, 0, self.epochs
        Hval = build_hist(u, i, r, t if t is not None else np.zeros(len(u)), nU, nI, self.L,
                          np.random.default_rng(123)) if val else None
        tt = t if t is not None else np.zeros(len(u))
        for ep in range(self.epochs):
            self.net.train()
            Hi, Hr, Ht, slot = build_hist(u, i, r, tt, nU, nI, self.L, rng)
            perm = rng.permutation(len(u))
            tot = 0.0
            for s in range(0, len(u), self.bs):
                b = perm[s:s + self.bs]
                ub = torch.as_tensor(u[b], device=DEVICE)
                ib = torch.as_tensor(i[b], device=DEVICE)
                hi, hr, hg, mask = self._batch((Hi, Hr, Ht), ub, ib, tt[b], slot[b], True)
                p, reg = self.net(ub, ib, hi, hr, hg, mask)
                y = torch.as_tensor(r[b], device=DEVICE, dtype=torch.float32)
                loss = ((p - y) ** 2).mean()
                opt.zero_grad()
                (loss + self.lam * reg).backward()
                opt.step()
                tot += loss.item() * len(b)
            del Hi, Hr, Ht
            msg = f"    histnet epoch {ep + 1}: train {tot / len(u):.4f}"
            if val:
                p = self._predict(Hval[:3], val["u"], val["i"], val.get("t"))
                mse = float(((p - val["r"]) ** 2).mean())
                msg += f"  val {mse:.4f}"
                if mse < best - 1e-4:
                    best, bad, self.best_epoch = mse, 0, ep + 1
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
            log(msg)
            if val and bad >= self.patience:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.best_mse = best
        return self

    def predict(self, u, i, t=None, n_draws=3):
        """Average over a few random history subsets (matters only for very long histories)."""
        hu, hi_, hr_, ht_ = self.data
        tt = ht_ if ht_ is not None else np.zeros(len(hu))
        ps = []
        for k in range(n_draws):
            H = build_hist(hu, hi_, hr_, tt, self.nU, self.nI, self.L, np.random.default_rng(1000 + k))
            ps.append(self._predict(H[:3], u, i, t))
        return np.mean(ps, 0)
