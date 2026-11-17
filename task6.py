"""Task 6: how long until the user's next read? (scored by absolute error)

Absolute error is smallest when predicting the typical (median) wait, so a small neural
network predicts the median of log(wait). Clues: the user's usual gaps and recent activity,
the next book's popularity and when it was being read, and how long other readers took to
get to that book (especially from the user's last book).

Training hides each user's last 1-3 reads and predicts them from the rest.

Run:  python task6.py
"""
import numpy as np
import torch
import torch.nn as nn

from recsys_lib import DEVICE, Data, log, read_csv, set_seed, user_ptr, write_csv

HOUR = 3600.0
TMAX = 2e9


def slog(x):
    """signed log1p in hours"""
    return np.sign(x) * np.log1p(np.abs(x) / HOUR)


def build(d, ptr, cut, users, items=None):
    """Features (and targets) for `users`, with each user's last `cut` reads hidden."""
    starts, ends = ptr[:-1], ptr[1:] - cut          # history = [start, end)
    keep = np.ones(d.n, bool)
    lens = np.diff(ptr)
    for c in range(1, cut + 1):
        keep[(ptr[1:] - c)[lens >= c]] = False  # only users that have that many rows
    hist = np.where(keep)[0]
    t = np.clip(d.t, 0, TMAX)
    t_end = np.quantile(t[hist], 0.995)
    # item statistics from history rows only
    ih, th = d.i[hist], t[hist]
    o = np.lexsort((th, ih))
    ih, th = ih[o], th[o]
    iptr = user_ptr(ih, d.nI + 1)
    pop = np.diff(iptr)
    has = pop > 0
    first = np.full(d.nI + 1, np.nan); first[has] = th[iptr[:-1][has]]
    p10 = np.full(d.nI + 1, np.nan); p10[has] = th[iptr[:-1][has] + (pop[has] * 0.1).astype(int)]
    med = np.full(d.nI + 1, np.nan); med[has] = th[iptr[:-1][has] + pop[has] // 2]
    key = ih.astype(np.int64) * 4_000_000_000 + th.astype(np.int64)
    # typical pause before each book, and how often it was bulk-added
    hu_, hi_ = d.u[hist], d.i[hist]
    prev_ok = np.r_[False, hu_[1:] == hu_[:-1]]
    g_prev = np.r_[0, np.diff(t[hist])]
    lgp = np.log1p(np.maximum(g_prev, 0) / HOUR)
    cnt = np.bincount(hi_[prev_ok], minlength=d.nI + 1)
    item_gap = (np.bincount(hi_[prev_ok], lgp[prev_ok], d.nI + 1) + 5 * np.median(lgp[prev_ok])) / (cnt + 5)
    item_bulk = (np.bincount(hi_[prev_ok], (g_prev[prev_ok] < 60).astype(float), d.nI + 1) + 1) / (cnt + 5)
    # how long other readers took between two books read back to back
    pa, pb = hi_[:-1][prev_ok[1:]], hi_[1:][prev_ok[1:]]
    pkey = pa.astype(np.int64) * (d.nI + 2) + pb
    pk, pinv = np.unique(pkey, return_inverse=True)
    psum = np.bincount(pinv, lgp[1:][prev_ok[1:]], len(pk))
    pcnt = np.bincount(pinv, minlength=len(pk))

    rows, ys = [], []
    for u in users:
        s, e = starts[u], ends[u]
        ts = t[s:e]
        g = np.diff(ts)
        lg = np.log1p(g / HOUR)
        tl = ts[-1]
        rows.append([np.log(e - s), np.median(lg), lg.mean(), lg.std(), lg[-1],
                     np.median(lg[-3:]), np.median(lg[-10:]), (g < 60).mean(), (g < 86400).mean(),
                     np.log1p((tl - ts[0]) / HOUR), (tl - 1.3e9) / 3e8, slog(t_end - tl),
                     np.log1p((ts > tl - 30 * 86400).sum()),     # activity in the last 30 days
                     lg[-1] - np.median(lg)])                    # last gap vs the usual gap
        if cut >= 1:
            ys.append(np.log1p(max(t[e] - tl, 0) / HOUR))
    U = np.array(rows, np.float64)
    tl = np.array([t[ends[u] - 1] for u in users])
    it = d.i[ends[users]] if items is None else items
    known = (it < d.nI) & (pop[np.minimum(it, d.nI)] > 0)
    after = np.searchsorted(key, it.astype(np.int64) * 4_000_000_000 + tl.astype(np.int64))
    frac_after = np.where(known, (iptr[it + 1] - np.clip(after, iptr[it], iptr[it + 1])) /
                         np.maximum(pop[it], 1), 0.5)
    I = np.column_stack([known, np.log1p(pop[it]), frac_after, item_gap[it], item_bulk[it]] +
                        [np.where(known, slog(np.nan_to_num(a[it]) - tl), 0) for a in (first, p10, med)])
    # when other readers read this book after the user's last read
    a0 = np.clip(after, iptr[it], iptr[it + 1])
    n_after = iptr[it + 1] - a0
    has_after = known & (n_after > 0)
    cond = [np.where(has_after, slog(th[np.minimum(a0 + (n_after * q).astype(int), len(th) - 1)] - tl), 0)
            for q in (0.25, 0.5, 0.75)]
    win = [np.log1p(np.searchsorted(key, it.astype(np.int64) * 4_000_000_000 + (tl + w).astype(np.int64))
                    - a0).clip(0) for w in (30 * 86400, 365 * 86400)]
    last_it = th[np.maximum(iptr[it + 1] - 1, 0)]
    # how long others took going from the user's last book to this one
    q = d.i[ends[users] - 1].astype(np.int64) * (d.nI + 2) + it
    j = np.clip(np.searchsorted(pk, q), 0, len(pk) - 1)
    hit = pk[j] == q
    c_ = np.where(hit, pcnt[j], 0)
    pair_gap = (np.where(hit, psum[j], 0) + 2 * item_gap[it]) / (c_ + 2)
    I = np.column_stack([I, has_after, np.log1p(n_after)] + cond + win +
                        [np.where(known, slog(last_it - tl), 0), np.log1p(c_), pair_gap])
    X = np.hstack([U, I])
    return X, (np.array(ys) if cut >= 1 else None)


class Net(nn.Module):
    def __init__(self, d_in, h=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                               nn.Linear(h, 1))

    def forward(self, x):
        return self.f(x)[:, 0]


def fit_net(X, y, epochs, Xv=None, yv_delta=None, seed=0, lr=1e-3, bs=1024):
    set_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    zcap = float(np.quantile(y, 0.99))  # cap predictions to avoid huge mistakes
    net = Net(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    Xt = torch.tensor(np.clip((X - mu) / sd, -5, 5), dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    hist = []
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        for s in range(0, len(Xt), bs):
            idx = perm[s:s + bs]
            loss = (net(Xt[idx]) - yt[idx]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if Xv is not None:
            mae = np.abs(to_delta((net, mu, sd, zcap), Xv) - yv_delta).mean()
            hist.append(mae)
            log(f"  epoch {ep}: val MAE {mae:,.0f}")
    return (net, mu, sd, zcap), hist


def to_delta(model, X):
    net, mu, sd, zcap = model
    net.eval()
    with torch.no_grad():
        z = net(torch.tensor(np.clip((X - mu) / sd, -5, 5), dtype=torch.float32,
                             device=DEVICE)).cpu().numpy()
    return np.expm1(np.clip(z, 0, zcap)) * HOUR


def train(train_file="task_6_train.csv", seed=0, cuts=(1, 2, 3), max_epochs=60, skip_zero=False):
    """Rating-0 rows are kept: they are still reads, so they matter for timing."""
    set_seed(seed)
    d = Data(train_file, skip_zero=skip_zero)
    assert (np.diff(d.u) != 0).sum() == d.nU - 1, "expected rows grouped by user"
    ptr = user_ptr(d.u, d.nU)
    lens = np.diff(ptr)
    rng = np.random.default_rng(seed)
    is_val = rng.random(d.nU) < 0.2
    Xs, ys = [], []
    for c in cuts:
        users = np.where((lens >= c + 3) & ~is_val)[0]
        X, y = build(d, ptr, c, users)
        Xs.append(X); ys.append(y)
        log(f"cut={c}: {len(users)} training examples")
    Xtr, ytr = np.vstack(Xs), np.concatenate(ys)
    vu = np.where(is_val & (lens >= 4))[0]
    Xv, yv = build(d, ptr, 1, vu)
    yv_delta = np.expm1(yv) * HOUR
    # simple baselines for comparison
    t = np.clip(d.t, 0, TMAX)
    med_user = np.array([np.median(np.diff(t[ptr[u]:ptr[u + 1] - 1])) for u in vu])
    log(f"baseline (user median gap) val MAE: {np.abs(med_user - yv_delta).mean():,.0f}")
    log(f"global median val MAE: {np.abs(np.median(yv_delta) - yv_delta).mean():,.0f}")
    _, hist = fit_net(Xtr, ytr, max_epochs, Xv, yv_delta, seed)
    best_ep = int(np.argmin(hist)) + 1
    log(f"best epoch {best_ep}: val MAE {min(hist):,.0f}")

    # Gaps grow toward the end of a history, and the test gap is one step past the training
    # data. Simulate that on the training data to pick a factor that scales predictions up.
    Xs, ys = [], []
    for c in [c + 1 for c in cuts]:
        X, y = build(d, ptr, c, np.where((lens >= c + 3) & ~is_val)[0])
        Xs.append(X); ys.append(y)
    shifted, _ = fit_net(np.vstack(Xs), np.concatenate(ys), best_ep, seed=seed)
    p = to_delta(shifted, Xv)
    grid = np.round(np.arange(1.0, 1.61, 0.05), 2)
    maes = [np.abs(s * p - yv_delta).mean() for s in grid]
    scale = float(grid[int(np.argmin(maes))])
    log(f"one-step-ahead simulation: MAE {maes[0]:,.0f} unscaled -> {min(maes):,.0f} "
        f"with predictions x{scale}")

    # final: train on every user (val users included)
    Xs, ys = [], []
    for c in cuts:
        X, y = build(d, ptr, c, np.where(lens >= c + 3)[0])
        Xs.append(X); ys.append(y)
    models = [fit_net(np.vstack(Xs), np.concatenate(ys), best_ep, seed=s)[0] for s in range(3)]
    # for users with too little history: the overall median gap
    fallback = float(np.median(np.expm1(np.concatenate(ys)) * HOUR))
    return dict(d=d, ptr=ptr, models=models, fallback=fallback * scale, scale=scale)


def predict(bundle, test_file="task_6_test.csv", out_file="task_6_predictions.csv"):
    header, rows = read_csv(test_file)
    d, ptr = bundle["d"], bundle["ptr"]
    users = d.enc_u([r[0] for r in rows])
    items = d.enc_i([r[1] for r in rows])
    lens = np.append(np.diff(ptr), 0)  # unknown users have no history
    ok = lens[users] >= 2
    log(f"{(~ok).sum()} test users have < 2 rated interactions -> fallback {bundle['fallback']:,.0f}s")
    p = np.full(len(rows), bundle["fallback"])
    X, _ = build(d, ptr, 0, users[ok], items[ok])
    # combine the seeds, then apply the scale factor
    p[ok] = np.median(np.stack([to_delta(m, X) for m in bundle["models"]]), 0) * bundle["scale"]
    n_in = len(header) - 1
    write_csv(out_file, header, [r[:n_in] + [f"{x:.1f}"] for r, x in zip(rows, p)])


if __name__ == "__main__":
    predict(train())
