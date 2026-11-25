from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal


# config
@dataclass
class Config:
    fs_target: int = 250
    mains_hz: tuple = (50.0, 60.0)
    notch_q: float = 30.0
    band_hz: tuple = (0.05, 35.0)
    band_order: int = 2
    median_ms: int = 100
    add_velocity: bool = True
    velocity_lowpass_hz: float = 10.0
    window_pre_s: float = 0.2
    window_post_s: float = 1.0
    flat_std_uv: float = 1.0
    max_abs_uv: float = 2000.0
    max_clipped_fraction: float = 0.01


@dataclass
class Recording:
    data: np.ndarray            # (2, n_samples), rows H then V
    fs: float
    subject: str
    events: list = field(default_factory=list)   # [(onset_s, label), ...]
    units_to_uv: float = 1.0


def causal_filter(sos, x):
    zi = signal.sosfilt_zi(sos)
    out = np.empty_like(x)
    for ch in range(x.shape[0]):
        out[ch], _ = signal.sosfilt(sos, x[ch], zi=zi * x[ch, 0])
    return out


def saturated_samples(raw, min_run=5):
    mask = np.zeros(raw.shape[1], dtype=bool)
    for ch in range(raw.shape[0]):
        for extreme in (raw[ch].min(), raw[ch].max()):
            at = np.r_[False, raw[ch] == extreme, False]
            edges = np.flatnonzero(np.diff(at.astype(int)))
            for start, end in zip(edges[0::2], edges[1::2]):
                if end - start >= min_run:
                    mask[start:end] = True
    return mask


def resample(rec: Recording, cfg: Config):
    x = rec.data.astype(float) * rec.units_to_uv
    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError(f"{rec.subject}: expected (2, n_samples), got {x.shape}")

    clipped = saturated_samples(x)

    ratio = Fraction(cfg.fs_target / rec.fs).limit_denominator(1000)
    if ratio != 1:
        x = signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=1)
        idx = np.minimum((np.arange(x.shape[1]) / ratio).astype(int), len(clipped) - 1)
        clipped = clipped[idx]
    return x, clipped


def notch_filter(x, cfg: Config):
    for f0 in cfg.mains_hz:
        if f0 < cfg.fs_target / 2:
            b, a = signal.iirnotch(f0, cfg.notch_q, fs=cfg.fs_target)
            x = causal_filter(signal.tf2sos(b, a), x)
    return x


def bandpass_filter(x, cfg: Config):
    sos = signal.butter(cfg.band_order, cfg.band_hz, btype="bandpass",
                        fs=cfg.fs_target, output="sos")
    return causal_filter(sos, x)


def median_filter(x, cfg: Config):
    n = int(round(cfg.median_ms / 1000 * cfg.fs_target))
    if n <= 1:
        return x
    padded = np.concatenate([np.repeat(x[:, :1], n - 1, axis=1), x], axis=1)
    return np.median(sliding_window_view(padded, n, axis=1), axis=-1)


def velocity_channels(x, cfg: Config):
    v = np.diff(x, axis=1, prepend=x[:, :1]) * cfg.fs_target
    sos = signal.butter(2, cfg.velocity_lowpass_hz, btype="lowpass",
                        fs=cfg.fs_target, output="sos")
    return causal_filter(sos, v)


def rejection_reason(window_pos, window_clipped, cfg: Config):
    if np.any(window_pos.std(axis=1) < cfg.flat_std_uv):
        return "flat"
    if np.any(np.abs(window_pos) > cfg.max_abs_uv):
        return "too_large"
    if window_clipped.mean() > cfg.max_clipped_fraction:
        return "clipped"
    return None


def segment(signals, clipped, events, cfg: Config):
    pre = int(round(cfg.window_pre_s * cfg.fs_target))
    post = int(round(cfg.window_post_s * cfg.fs_target))
    n_total = signals.shape[1]

    windows, labels = [], []
    dropped = {"out_of_bounds": 0, "flat": 0, "too_large": 0, "clipped": 0}

    for onset_s, label in events:
        centre = int(round(onset_s * cfg.fs_target))
        start, end = centre - pre, centre + post
        if start < 0 or end > n_total:
            dropped["out_of_bounds"] += 1
            continue
        w = signals[:, start:end]
        reason = rejection_reason(w[:2], clipped[start:end], cfg)
        if reason:
            dropped[reason] += 1
            continue
        windows.append(w)
        labels.append(label)
    return windows, labels, dropped


def subject_stats(continuous_list, clipped_list):
    good = [s[:, ~c] for s, c in zip(continuous_list, clipped_list)]
    allsig = np.concatenate(good, axis=1)
    mean = allsig.mean(axis=1, keepdims=True)
    std = allsig.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return mean, std


DEFAULT_FILTERS = [notch_filter, bandpass_filter, median_filter]


def clean_continuous(rec: Recording, cfg: Config, filters=None):
    x, clipped = resample(rec, cfg)
    for f in filters or DEFAULT_FILTERS:
        x = f(x, cfg)
    if cfg.add_velocity:
        x = np.concatenate([x, velocity_channels(x, cfg)], axis=0)
    return x, clipped


def preprocess(recordings, cfg: Config = None, filters=None):
    cfg = cfg or Config()
    cleaned = [(rec, *clean_continuous(rec, cfg, filters)) for rec in recordings]

    stats = {}
    for subj in sorted({rec.subject for rec in recordings}):
        sigs = [s for r, s, _ in cleaned if r.subject == subj]
        clips = [c for r, _, c in cleaned if r.subject == subj]
        stats[subj] = subject_stats(sigs, clips)

    X, y, subjects = [], [], []
    report = {}
    for rec, sig, clipped in cleaned:
        wins, labs, dropped = segment(sig, clipped, rec.events, cfg)
        mean, std = stats[rec.subject]
        X += [(w - mean) / std for w in wins]
        y += labs
        subjects += [rec.subject] * len(labs)

        r = report.setdefault(rec.subject, {"kept": 0, "dropped": {}})
        r["kept"] += len(labs)
        for k, v in dropped.items():
            r["dropped"][k] = r["dropped"].get(k, 0) + v

    channels = ["H", "V", "H_vel", "V_vel"] if cfg.add_velocity else ["H", "V"]
    n_samp = int(round(cfg.window_pre_s * cfg.fs_target)) + \
             int(round(cfg.window_post_s * cfg.fs_target))
    X = np.stack(X).astype(np.float32) if X else np.empty((0, len(channels), n_samp), np.float32)
    return {"X": X, "y": np.array(y), "subjects": np.array(subjects),
            "channels": channels, "report": report}


def plot_steps(rec: Recording, cfg: Config = None, seconds=(0, 20),
               path="iris_preprocessing_steps.png"):
    import matplotlib.pyplot as plt

    cfg = cfg or Config()
    raw, _ = resample(rec, cfg)
    notched = notch_filter(raw, cfg)
    banded = bandpass_filter(notched, cfg)
    med = median_filter(banded, cfg)
    vel = velocity_channels(med, cfg)

    t = np.arange(raw.shape[1]) / cfg.fs_target
    sel = (t >= seconds[0]) & (t < seconds[1])
    stages = [("Resampled raw", raw), ("+ Notch", notched),
              ("+ Band-pass", banded), ("+ Median", med),
              ("Velocity channels", vel)]

    fig, axes = plt.subplots(len(stages), 1, figsize=(12, 11), sharex=True)
    for ax, (name, s) in zip(axes, stages):
        ax.plot(t[sel], s[0, sel], lw=0.8, label="H")
        ax.plot(t[sel], s[1, sel], lw=0.8, label="V")
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_ylabel("uV/s" if "Velocity" in name else "uV")
        for onset, label in rec.events:
            if seconds[0] <= onset < seconds[1]:
                ax.axvline(onset, color="grey", lw=0.5, ls="--")
                if ax is axes[0]:
                    ax.text(onset, ax.get_ylim()[1], label, fontsize=7,
                            va="top", ha="left", color="grey")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"IRIS preprocessing, subject {rec.subject}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# synthetic demo data
def make_synthetic_recording(subject, fs=500, minutes=2, amplitude=1.0,
                             mains=50, seed=0):
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * fs)
    t = np.arange(n) / fs
    h = np.zeros(n)
    v = np.zeros(n)
    events = []

    onset = 2.0
    while onset < minutes * 60 - 3:
        kind = rng.choice(["left", "right", "up", "down", "blink"])
        i0 = int(onset * fs)
        hold = int(rng.uniform(0.4, 0.8) * fs)
        ramp = int(0.04 * fs)
        step = np.concatenate([np.linspace(0, 1, ramp), np.ones(hold),
                               np.linspace(1, 0, ramp)])
        size = 150 * amplitude * rng.uniform(0.8, 1.2)
        if kind == "left":
            h[i0:i0 + len(step)] += size * step
        elif kind == "right":
            h[i0:i0 + len(step)] -= size * step
        elif kind == "up":
            v[i0:i0 + len(step)] += size * step
        elif kind == "down":
            v[i0:i0 + len(step)] -= size * step
        else:
            pulse = signal.windows.hann(int(0.25 * fs))
            v[i0:i0 + len(pulse)] += 1.6 * size * pulse
            h[i0:i0 + len(pulse)] += 0.1 * size * pulse
        events.append((onset, str(kind)))
        onset += rng.uniform(1.8, 3.0)

    drift = np.cumsum(rng.normal(0, 0.4, (2, n)), axis=1)
    hum = 30 * np.sin(2 * np.pi * mains * t)
    noise = rng.normal(0, 5, (2, n))
    data = np.vstack([h, v]) + drift + hum + noise

    j = int(30 * fs)
    data[:, j:j + int(1.5 * fs)] = 5000.0

    return Recording(data=data, fs=fs, subject=subject, events=events)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    cfg = Config()
    recs = [
        make_synthetic_recording("s01", amplitude=1.0, mains=50, seed=1),
        make_synthetic_recording("s02", amplitude=0.4, mains=60, seed=2),
    ]

    out = preprocess(recs, cfg)
    print("X shape:", out["X"].shape)
    print("channels:", out["channels"])
    print("labels:", dict(zip(*np.unique(out["y"], return_counts=True))))
    for subj, r in out["report"].items():
        print(f"{subj}: kept {r['kept']}, dropped {r['dropped']}")
    print("saved plot:", plot_steps(recs[0], cfg, seconds=(0, 20)))
