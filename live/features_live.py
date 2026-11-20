"""Online (causal) version of the model features, computed with numpy for ALL link directions at once.
It reproduces window_features.py (2 s windows, EMA, neighbours) and the 5 s / 10 s windows of train_long.py
from the per-interval f_* values. check_features.py verifies it against the offline code."""
import numpy as np

W, ALPHA = 8, 0.3
LONG = (20, 40)
WIN = ['w_delay_level', 'w_delay_trend', 'w_delay_var', 'w_jitter_mean', 'w_queue_delay_est', 'w_util_mean',
       'w_corr_util_delay', 'w_loss_mean', 'w_probe_loss', 'w_corr_util_loss']
F_COLS = ['f_util', 'f_tx_mbps', 'f_tx_pps', 'f_rx_pps_far', 'f_tx_dropped', 'f_rx_dropped_far', 'f_probe_sent',
          'f_probe_recv', 'f_delay_mean_ms', 'f_delay_min_ms', 'f_delay_max_ms', 'f_delay_std_ms', 'f_delay_p50_ms',
          'f_probe_reordered', 'f_probe_lost', 'f_probe_loss_ratio', 'f_pathloss_wmean', 'f_pathloss_max',
          'f_path_pairs_active', 'f_path_pkts_2s']
_X = np.arange(W) - (W - 1) / 2


def _full(a, n):          # last n columns, or None if fewer than n intervals exist
    return a[:, -n:] if a.shape[1] >= n else None


def _mean_full(a, n):     # pandas rolling(n, min_periods=n).mean(): NaN if any NaN
    s = _full(a, n)
    return np.full(a.shape[0], np.nan) if s is None else s.mean(1)


def _corr(x, y):          # pandas rolling corr (ddof cancels); NaN -> 0 like .fillna(0)
    if x is None:
        return np.zeros(0)
    xm, ym = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    num = (xm * ym).sum(1)
    den = np.sqrt((xm ** 2).sum(1) * (ym ** 2).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = num / den
    r[~np.isfinite(r)] = 0.0
    return r


class FeatureEngine:
    def __init__(self, dirs, prop_delay_ms, capacity_mbps=10, queue_limit_bytes=150000):
        """dirs: list of (src, dst) per direction id, in dir_id order."""
        self.n = len(dirs)
        self.static = {'static_capacity_mbps': np.full(self.n, float(capacity_mbps)),
                       'static_prop_delay_ms': np.asarray(prop_delay_ms, float),
                       'static_queue_limit_bytes': np.full(self.n, float(queue_limit_bytes))}
        nb = np.zeros((self.n, self.n))
        for i, (a, b) in enumerate(dirs):
            for j, (c, d) in enumerate(dirs):
                if i != j and {a, b} & {c, d} and {a, b} != {c, d}:
                    nb[i, j] = 1
        self.nb = nb
        self.hist = {c: np.zeros((self.n, 0)) for c in F_COLS + ['p_delay_ms', 'x_portloss', 'x_excess_delay']}
        self.ema_d = np.full(self.n, np.nan)
        self.ema_l = np.full(self.n, np.nan)
        self.base = np.full(self.n, np.nan)
        self.k = 0

    def _push(self, name, v):
        self.hist[name] = np.concatenate([self.hist[name][:, -60:], v[:, None]], axis=1)

    def step(self, f):
        """f: dict f_* -> array(n) for the interval just finished. Returns dict feature -> array(n)."""
        H = self.hist
        prev_d = H['p_delay_ms'][:, -1] if H['p_delay_ms'].shape[1] else np.full(self.n, np.nan)
        for c in F_COLS:
            self._push(c, np.asarray(f[c], float))
        d = np.asarray(f['f_delay_mean_ms'], float).copy()
        # single missing delay filled from the previous interval (offline also checks the NEXT one; not possible online)
        miss = np.isnan(d)
        prev_raw_missing = np.isnan(H['f_delay_mean_ms'][:, -2]) if H['f_delay_mean_ms'].shape[1] >= 2 else np.ones(self.n, bool)
        fill = miss & ~prev_raw_missing
        d[fill] = prev_d[fill]
        self._push('p_delay_ms', d)
        loss = np.nan_to_num(np.asarray(f['f_pathloss_wmean'], float), nan=0.0)
        util = np.asarray(f['f_util'], float)
        # EMA (adjust=False); a missing value keeps the previous EMA
        e = np.where(np.isnan(self.ema_d), d, ALPHA * d + (1 - ALPHA) * self.ema_d)
        self.ema_d = np.where(np.isnan(d), self.ema_d, e)
        self.ema_l = np.where(np.isnan(self.ema_l), loss, ALPHA * loss + (1 - ALPHA) * self.ema_l)
        mn = np.asarray(f['f_delay_min_ms'], float)
        self.base = np.where(np.isnan(self.base), mn, np.fmin(self.base, mn))
        base_now = np.where(np.isnan(mn), np.nan, self.base)
        tx = np.asarray(f['f_tx_pps'], float)
        self._push('x_portloss', np.clip((tx - np.asarray(f['f_rx_pps_far'], float)) / np.clip(tx, 1, None), -0.5, 1))
        self._push('x_excess_delay', np.asarray(f['f_delay_mean_ms'], float) - self.static['static_prop_delay_ms'])
        out = {c: H[c][:, -1] for c in F_COLS}
        out.update(self.static)
        out['p_delay_ms'] = d
        out['p_delay_ema'] = self.ema_d.copy()
        out['p_loss_ema'] = self.ema_l.copy()
        D = H['p_delay_ms']; U = H['f_util']
        LOSS = np.nan_to_num(H['f_pathloss_wmean'], nan=0.0)
        out['w_delay_level'] = _mean_full(D, W)
        out['w_delay_d1'] = D[:, -1] - D[:, -2] if D.shape[1] >= 2 else np.full(self.n, np.nan)
        out['w_delay_d2'] = (D[:, -1] - 2 * D[:, -2] + D[:, -3]) if D.shape[1] >= 3 else np.full(self.n, np.nan)
        dw = _full(D, W)
        if dw is None:
            out['w_delay_trend'] = out['w_delay_var'] = np.full(self.n, np.nan)
        else:
            out['w_delay_trend'] = (dw - dw.mean(1, keepdims=True)) @ _X / (_X @ _X)
            out['w_delay_var'] = dw.var(1)
        out['w_jitter_mean'] = _mean_full(H['f_delay_std_ms'], W)
        out['w_queue_delay_est'] = d - base_now
        out['w_util_mean'] = _mean_full(U, W)
        uw = _full(U, W)
        out['w_util_max'] = np.full(self.n, np.nan) if uw is None else uw.max(1)
        cd = _corr(uw, dw) if uw is not None else np.full(self.n, 0.0)
        out['w_corr_util_delay'] = cd if len(cd) else np.zeros(self.n)
        out['w_loss_mean'] = _mean_full(LOSS, W)
        out['w_probe_loss'] = _mean_full(H['f_probe_loss_ratio'], W)
        lw, uu = LOSS[:, -W:], U[:, -W:]
        with np.errstate(invalid='ignore'):
            hi = np.where(uu >= 0.8, lw, np.nan); lo = np.where(uu < 0.5, lw, np.nan)
        with np.errstate(invalid='ignore', divide='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out['w_loss_when_util_high'] = np.nanmean(hi, 1)
                out['w_loss_when_util_low'] = np.nanmean(lo, 1)
        cl = _corr(uw, _full(LOSS, W)) if uw is not None else np.zeros(self.n)
        out['w_corr_util_loss'] = cl if len(cl) else np.zeros(self.n)
        # neighbours: mean over neighbouring directions, ignoring NaN
        for c in WIN:
            v = out[c]; ok = ~np.isnan(v)
            num = self.nb @ np.where(ok, v, 0.0); cnt = self.nb @ ok.astype(float)
            with np.errstate(invalid='ignore', divide='ignore'):
                out['nb_' + c] = np.where(cnt > 0, num / cnt, np.nan)
        # long windows (5 s, 10 s)
        for Lw in LONG:
            for src, name in [('f_pathloss_wmean', 'pathloss'), ('x_portloss', 'portloss'), ('f_delay_std_ms', 'delaystd'),
                              ('x_excess_delay', 'excessdelay'), ('f_probe_loss_ratio', 'probeloss')]:
                out[f'L{Lw}_{name}_mean'] = _mean_full(H[src], Lw)
            s = _full(H['f_delay_std_ms'], Lw)
            out[f'L{Lw}_delaystd_max'] = np.full(self.n, np.nan) if s is None or np.isnan(s).any() else s.max(1)
            if s is not None and np.isnan(s).any():
                out[f'L{Lw}_delaystd_max'] = np.where(np.isnan(s).any(1), np.nan, np.nanmax(s, 1))
            s = _full(H['f_pathloss_wmean'], Lw)
            out[f'L{Lw}_pathloss_std'] = np.full(self.n, np.nan) if s is None else s.std(1, ddof=1)
        out['w_complete'] = (~np.isnan(_full(D, W)).any(1) & ~np.isnan(uw).any(1)) if uw is not None else np.zeros(self.n, bool)
        self.k += 1
        return out
