"""Summary figure and numbers for Lesson 6 (same method as Lesson 5, fixed random seed).

Run from the repository root:  python scripts/make_summary_figure.py   (about 1 minute)
Writes results/summary_figure.png and results/numbers.json.
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

edges = pd.read_parquet(DATA / "pn_kc_edges_all.parquet")
ann = pd.read_csv(DATA / "neuron_annotations.tsv", sep="\t", low_memory=False)
upn = ann[(ann.cell_class == "ALPN") & (ann.cell_sub_class == "uniglomerular")].copy()
upn["glom"] = upn.cell_type.str.split("_").str[0]

def build(neuropil, thresh):
    sub = edges[(edges.neuropil == neuropil) & (edges.syn_count >= thresh)]
    sub = sub.merge(upn[["root_id", "glom"]], left_on="pre_pt_root_id", right_on="root_id")
    W = sub.groupby(["post_pt_root_id", "glom"]).syn_count.sum().unstack(fill_value=0)
    cls = (ann.set_index("root_id").cell_type.reindex(W.index)
              .str.extract(r"^(KCg|KCab|KCapbp|KCa'b')")[0].replace({"KCa'b'": "KCapbp"}).values)
    return W, (W > 0).astype(int), cls

def swap(rows, i, j, rng):
    a, b = rows[i], rows[j]
    common, only_a, only_b = a & b, a - b, b - a
    pool = list(only_a) + list(only_b); rng.shuffle(pool); k = len(only_a)
    rows[i] = common | set(pool[:k]); rows[j] = common | set(pool[k:])

def curveball(M, n_swaps, rng):
    rows = [set(np.flatnonzero(r)) for r in M]
    for _ in range(n_swaps):
        i, j = rng.choice(len(rows), 2, replace=False); swap(rows, i, j, rng)
    out = np.zeros_like(M)
    for i, r in enumerate(rows): out[i, list(r)] = 1
    return out

def curveball_by_class(M, classes, k, rng):
    out = M.copy()
    for c in pd.unique(classes):
        idx = np.flatnonzero(classes == c)
        if len(idx) > 1: out[idx] = curveball(M[idx], k * len(idx), rng)
    return out

def pair_z(B, cls, g1, g2, n_null=200):
    Bm = B.values; i1, i2 = list(B.columns).index(g1), list(B.columns).index(g2)
    obs = (Bm[:, i1] * Bm[:, i2]).sum(); null = []; Bs = Bm.copy()
    for _ in range(n_null):
        Bs = curveball_by_class(Bs, cls, 1, rng); null.append((Bs[:, i1] * Bs[:, i2]).sum())
    null = np.array(null)
    return dict(obs=int(obs), null_mean=round(float(null.mean()), 1),
                z=round(float((obs - null.mean()) / null.std()), 2),
                p=round(float(((null >= obs).sum() + 1) / (n_null + 1)), 4))

# ---------- A. replication + threshold
out = {"replication": [], "threshold_sweep_right": []}
PAIRS = [("DM1", "DM4"), ("DM1", "VM3"), ("DM1", "VA2"), ("DM4", "VM3")]
mats = {"right": build("MB_CA_R", 5), "left": build("MB_CA_L", 5)}
for side, (_, B_, cls_) in mats.items():
    for g1, g2 in PAIRS:
        out["replication"].append({"side": side, "pair": f"{g1}–{g2}", **pair_z(B_, cls_, g1, g2)})
for th in [1, 3, 5, 10]:
    _, B_, cls_ = build("MB_CA_R", th)
    out["threshold_sweep_right"].append({"thresh": th, "n_kc": len(B_),
        "mean_in_degree": round(float(B_.values.sum(1).mean()), 2), "z_DM1_DM4": pair_z(B_, cls_, "DM1", "DM4")["z"]})

# ---------- B. model
W_R, _, cls_R = mats["right"]
W_real = W_R.values.astype(float); glom = list(W_R.columns); n_glom = W_real.shape[1]
FOOD = [glom.index(g) for g in ["DM1", "DM4", "VM3", "VA2"]]

def kc_code(Wm, X, s=0.05):
    H = X @ Wm.T; return (H >= np.quantile(H, 1 - s, axis=1, keepdims=True)).astype(float)
def make_odors(n, food=False):
    p = np.full(n_glom, 0.2)
    if food: p[FOOD] = 0.9
    return (rng.random((n, n_glom)) < p) * rng.lognormal(0, 0.5, (n, n_glom))
def noisy(X, noise, reps):
    Xr = np.repeat(X, reps, axis=0); return np.clip(Xr + noise * rng.standard_normal(Xr.shape) * X.mean(), 0, None)
def valence_acc(enc, odors, labels, noise=1.0):
    clf = LogisticRegression(max_iter=3000).fit(enc(noisy(odors, noise, 10)), np.repeat(labels, 10))
    return clf.score(enc(noisy(odors, noise, 5)), np.repeat(labels, 5))
def shuffled_weights(Wv, cls):
    Bsh = curveball_by_class((Wv > 0).astype(int), cls, 5, rng); Wsh = np.zeros_like(Wv)
    for i in range(len(Wv)):
        w = Wv[i][Wv[i] > 0].copy(); rng.shuffle(w); Wsh[i, np.flatnonzero(Bsh[i])] = w
    return Wsh
def pr(R):
    Rc = R - R.mean(0); lam = np.clip(np.linalg.eigvalsh(Rc @ Rc.T), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())

W_shufs = [shuffled_weights(W_real, cls_R) for _ in range(5)]
Ps = [50, 100, 200, 400]
cap = {"P": Ps, "PN": [], "real": [], "shuf_mean": [], "shuf_sd": []}
for P in Ps:
    od = make_odors(P); lab = rng.integers(0, 2, P)
    cap["PN"].append(valence_acc(lambda X: X, od, lab))
    cap["real"].append(valence_acc(lambda X: kc_code(W_real, X), od, lab))
    s = [valence_acc(lambda X, Ws=Ws: kc_code(Ws, X), od, lab) for Ws in W_shufs]
    cap["shuf_mean"].append(float(np.mean(s))); cap["shuf_sd"].append(float(np.std(s)))
out["valence_capacity"] = {k: [round(v, 3) for v in vs] if k != "P" else vs for k, vs in cap.items()}

dim = {}
for food in [False, True]:
    od = make_odors(1000, food)
    sh = [pr(kc_code(Ws, od)) for Ws in W_shufs]
    dim["food" if food else "generic"] = {"PN": round(pr(od), 1), "real": round(pr(kc_code(W_real, od)), 1),
                                          "shuf_mean": round(float(np.mean(sh)), 1), "shuf_sd": round(float(np.std(sh)), 1)}
out["participation_ratio"] = dim
(OUT / "numbers.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

# ---------- figure
BLUE, AQUA, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#1baf7a", "#eb6834", "#8a8983", "#1f1f1d", "#6b6a64"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9), gridspec_kw={"width_ratios": [1.2, 1, 0.9]})

ax = axes[0]; rep = pd.DataFrame(out["replication"]); x = np.arange(len(PAIRS)); w = 0.38
for k, (side, col) in enumerate([("right", BLUE), ("left", AQUA)]):
    v = rep[rep.side == side].z.values
    ax.bar(x + (k - 0.5) * (w + 0.02), v, w, color=col, label=f"{side} hemisphere")
ax.axhline(3, color=GRAY, lw=0.8, ls="--"); ax.text(3.62, 3.1, "z = 3", color=MUTED, ha="left", va="bottom", fontsize=7); ax.set_xlim(-0.6, 4.0)
ax.set_xticks(x); ax.set_xticklabels([f"{a}–\n{b}" for a, b in PAIRS]); ax.set_ylabel("z vs class-matched null")
ax.set_title("A  Co-convergence replicates", loc="left", fontsize=8.5, color=INK)
ax.legend(frameon=False, fontsize=7, loc="upper right")

ax = axes[1]
ax.plot(Ps, cap["PN"], color=GRAY, lw=2, marker="s", ms=4, ls="--")
ax.plot(Ps, cap["real"], color=ORANGE, lw=2, marker="o", ms=4)
ax.errorbar(Ps, cap["shuf_mean"], yerr=cap["shuf_sd"], color=INK, lw=1.2, marker="^", ms=4, capsize=2)
ax.text(410, cap["PN"][-1], "PN (55-d)", color=MUTED, va="center", fontsize=7)
ax.text(410, cap["real"][-1] + 0.025, "KC, real W", color=ORANGE, va="center", fontsize=7)
ax.text(410, cap["shuf_mean"][-1] - 0.035, "KC, shuffled W", color=INK, va="center", fontsize=7)
ax.axvline(2 * n_glom, color=GRAY, lw=0.8, ls=":"); ax.text(2 * n_glom + 8, 0.93, "2N = 110", color=MUTED, fontsize=7)
ax.set_xticks(Ps); ax.set_xlim(20, 560)
ax.set_xlabel("odours with random valence (P)"); ax.set_ylabel("test accuracy")
ax.set_title("B  Expansion raises capacity", loc="left", fontsize=8.5, color=INK)

ax = axes[2]; x = np.arange(2)
real = [dim[k]["real"] for k in ["generic", "food"]]; sh = [dim[k]["shuf_mean"] for k in ["generic", "food"]]
ax.bar(x - 0.2, real, 0.38, color=ORANGE, label="real W")
ax.bar(x + 0.2, sh, 0.38, yerr=[dim[k]["shuf_sd"] for k in ["generic", "food"]], color=GRAY, label="shuffled W", capsize=2)
for i in range(2):
    ax.text(x[i] - 0.2, real[i] + 2, f"{real[i]:.0f}", ha="center", fontsize=7, color=INK)
    ax.text(x[i] + 0.2, sh[i] + 4, f"{sh[i]:.0f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(x); ax.set_xticklabels(["generic\nodours", "food-like\nodours"]); ax.set_ylabel("KC participation ratio")
ax.set_title("C  Real wiring: lower dimension", loc="left", fontsize=8.5, color=INK)
ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2); ax.set_ylim(0, max(sh) * 1.4)

fig.tight_layout(); fig.savefig(OUT / "summary_figure.png", dpi=200, facecolor="white")
print(json.dumps(out, indent=1, ensure_ascii=False))
