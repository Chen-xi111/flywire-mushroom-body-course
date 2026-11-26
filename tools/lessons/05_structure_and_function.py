from nbbuild import md, code, todo, answer

CELLS = [
md("""
# Lesson 5 · Back to the FlyWire question: is the structure real, and what is it for?

Lesson 4 showed that DM1–DM4 co-convergence in the right hemisphere is far above the null-model expectation. This lesson answers the two kinds of question you must answer before writing anything up.

**Part A · Robustness (does the result hold up?)**
- Is it still there in the other hemisphere? This is **replication**.
- Is it still there with a different `syn_count` threshold? This is **sensitivity analysis**.

**Part B · Coding theory: why expansion?**
- Why the mushroom body uses **expansion + sparsification**: **Cover's theorem**.

**Part C · Learning: how MBONs read out the KCs**
- How an MBON **learns** a **linear readout**, and why the choice of task matters.

**Part D · Hypothesis test**
- Is the structured wiring there to serve **food odours**?
- Tested with **dimensionality** (the **participation ratio**) and **learning capacity**.

8 TODOs in total. Run from top to bottom with **Shift + Enter**; the whole notebook takes about 2–3 minutes.
"""),
md("""
## Step 0: load the data and the Lesson 4 tools (provided)

`build(neuropil, thresh)` returns three things:
- `W`: the KC × glomerulus **synapse-count** matrix (weighted; used by the model in Part C);
- `B`: the **binary matrix** of the same shape (0/1; used for statistics, the `B` of Lesson 4);
- `cls`: the **class** of each KC: `KCg`, `KCab` or `KCapbp`.

`swap`, `curveball` and `curveball_by_class` are the functions you wrote in Lesson 4, copied unchanged.
"""),
code("""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

edges = pd.read_parquet("../data/pn_kc_edges_all.parquet")
ann = pd.read_csv("../data/neuron_annotations.tsv", sep="\\t", low_memory=False)
upn = ann[(ann.cell_class == "ALPN") & (ann.cell_sub_class == "uniglomerular")].copy()
upn["glom"] = upn.cell_type.str.split("_").str[0]

def build(neuropil, thresh):
    sub = edges[(edges.neuropil == neuropil) & (edges.syn_count >= thresh)]
    sub = sub.merge(upn[["root_id", "glom"]], left_on="pre_pt_root_id", right_on="root_id")
    W = sub.groupby(["post_pt_root_id", "glom"]).syn_count.sum().unstack(fill_value=0)
    B = (W > 0).astype(int)
    cls = (ann.set_index("root_id").cell_type.reindex(W.index)
              .str.extract(r"^(KCg|KCab|KCapbp|KCa'b')")[0]
              .replace({"KCa'b'": "KCapbp"}).values)
    return W, B, cls

def swap(rows, i, j, rng):
    a, b = rows[i], rows[j]
    common = a & b
    only_a, only_b = a - b, b - a
    pool = list(only_a) + list(only_b)
    rng.shuffle(pool)
    k = len(only_a)
    rows[i] = common | set(pool[:k])
    rows[j] = common | set(pool[k:])

def curveball(M, n_swaps, rng):
    rows = [set(np.flatnonzero(r)) for r in M]
    n = len(rows)
    for _ in range(n_swaps):
        i, j = rng.choice(n, 2, replace=False)
        swap(rows, i, j, rng)
    out = np.zeros_like(M)
    for i, r in enumerate(rows):
        out[i, list(r)] = 1
    return out

def curveball_by_class(M, classes, n_swaps_per_kc, rng):
    out = M.copy()
    for c in pd.unique(classes):
        idx = np.flatnonzero(classes == c)
        if len(idx) > 1:
            out[idx] = curveball(M[idx], n_swaps_per_kc * len(idx), rng)
    return out

rng = np.random.default_rng(0)
W_R, B_R, cls_R = build("MB_CA_R", 5)
print("right:", B_R.shape, " mean in-degree:", B_R.values.sum(axis=1).mean().round(2))
"""),
md("""
---
# Part A · Robustness

## Why replicate?

The DM1–DM4 pair (and the other pairs below) were picked **after looking at the right-hemisphere data**, by scanning all glomerulus pairs for the strongest effects. Using the same data both to choose a hypothesis and to test it is called **double dipping** (or **circular analysis**), and it makes p-values look better than they are.

The **left hemisphere** is an independent data set: a different set of PNs and KCs, proofread by different people. Testing **only the pairs fixed in advance** on the left side is a clean **held-out replication**:
- there is no **multiple comparisons** problem, because we test 4 pairs rather than re-scanning ~1,500;
- if the left side is also significant, the result is hard to explain as chance or as a right-side proofreading bias.

## TODO 1: build the left-hemisphere matrices

The left calyx neuropil is called `"MB_CA_L"`.

To do:
1. use `build` to get `W_L, B_L, cls_L`, still with threshold 5;
2. print the left-side shape and mean in-degree;
3. compare with the right side (printed above).

Think: are the numbers of KCs and the in-degrees on the two sides similar? If they were very different, what would that mean for the comparison?
"""),
todo("""
# TODO 1: write your code here
""", """
# TODO 1
W_L, B_L, cls_L = build("MB_CA_L", 5)
print("left :", B_L.shape, " mean in-degree:", B_L.values.sum(axis=1).mean().round(2))
"""),
md("""
## TODO 2: wrap the Lesson 4 test in a function

We will run the test for 2 hemispheres × 4 glomerulus pairs, and then for 4 thresholds. Copy-pasting the Lesson 4 code every time is error-prone, so wrap it in a function `pair_z`.

You wrote every line of it in Lesson 4:
- `obs`: Lesson 4 TODO 1;
- stratified shuffling: Lesson 4 TODO 6, with `curveball_by_class` and `n_swaps_per_kc = 1`;
- `z` and `p`: Lesson 4 TODO 5, with p = `(count + 1) / (n + 1)`.

To do:
1. fill in the `___`;
2. run `pair_z(B_R, cls_R, "DM1", "DM4")` on the right side; z should be close to your stratified result from Lesson 4 (it varies a little with the seed);
3. run the same pair on the left side.
"""),
todo("""
# TODO 2: fill in the ___
def pair_z(B, cls, g1, g2, n_null=100, rng=rng):
    Bm = B.values
    i1, i2 = list(B.columns).index(g1), list(B.columns).index(g2)
    obs = ___
    null = []
    Bs = Bm.copy()
    for _ in range(n_null):
        Bs = ___
        null.append((Bs[:, i1] * Bs[:, i2]).sum())
    null = np.array(null)
    z = ___
    p = ___
    return dict(obs=int(obs), null_mean=null.mean(), null_sd=null.std(), z=z, p=p)

print("right:", pair_z(B_R, cls_R, "DM1", "DM4"))
print("left :", pair_z(B_L, cls_L, "DM1", "DM4"))
""", """
# TODO 2
def pair_z(B, cls, g1, g2, n_null=100, rng=rng):
    Bm = B.values
    i1, i2 = list(B.columns).index(g1), list(B.columns).index(g2)
    obs = (Bm[:, i1] * Bm[:, i2]).sum()
    null = []
    Bs = Bm.copy()
    for _ in range(n_null):
        Bs = curveball_by_class(Bs, cls, 1, rng)
        null.append((Bs[:, i1] * Bs[:, i2]).sum())
    null = np.array(null)
    z = (obs - null.mean()) / null.std()
    p = (np.sum(null >= obs) + 1) / (n_null + 1)
    return dict(obs=int(obs), null_mean=null.mean(), null_sd=null.std(), z=z, p=p)

print("right:", pair_z(B_R, cls_R, "DM1", "DM4"))
print("left :", pair_z(B_L, cls_L, "DM1", "DM4"))
"""),
md("""
## TODO 3: four glomerulus pairs × two hemispheres in one table

The four strongest pairs in the right hemisphere were DM1–DM4, DM1–VM3, DM1–VA2 and DM4–VM3.

To do:
1. for each hemisphere and each pair, call `pair_z` and store the result, together with `side` and `pair`, as one row;
2. build a DataFrame `table` from all rows;
3. use `pivot` to make a small table with rows = pair, columns = side, values = z.

Hint: `{"side": side, "pair": f"{g1}-{g2}", **r}` unpacks the dict `r` into the new dict. `**` is called **dict unpacking**.

Think:
- Are all left-side z values > 3?
- Which pair is most consistent across sides, and which differs most?
- With only 4 tests on the left side, do you need a strict multiple-comparison correction?
"""),
todo("""
# TODO 3: write your code here
PAIRS = [("DM1", "DM4"), ("DM1", "VM3"), ("DM1", "VA2"), ("DM4", "VM3")]
""", """
# TODO 3
PAIRS = [("DM1", "DM4"), ("DM1", "VM3"), ("DM1", "VA2"), ("DM4", "VM3")]
rows = []
for side, (B_, cls_) in {"right": (B_R, cls_R), "left": (B_L, cls_L)}.items():
    for g1, g2 in PAIRS:
        r = pair_z(B_, cls_, g1, g2)
        rows.append({"side": side, "pair": f"{g1}-{g2}", **r})
table = pd.DataFrame(rows).round(2)
print(table)
table.pivot(index="pair", columns="side", values="z")
"""),
md("""
## TODO 4: sensitivity analysis: does the result survive a different threshold?

`syn_count >= 5` is a choice you justified with data in Lesson 3, but it is still a choice. Any reviewer will ask: **"what about 3, or 10?"**

To do: for the right side, use thresholds 1, 3, 5 and 10. For each, record:
- the number of KCs `n_kc`;
- `mean_in_degree`, the average number of glomeruli per KC (a single number: remember `.mean()`);
- `z_DM1_DM4`.

Then make a DataFrame `sweep`.

Think: which of these two conclusions is sensitive to the threshold, and which is **robust**?
- Conclusion 1: "each KC receives input from about 4 glomeruli on average";
- Conclusion 2: "DM1 and DM4 co-converge significantly".

How should each be worded in a write-up?
"""),
todo("""
# TODO 4: write your code here
""", """
# TODO 4
sweep = []
for th in [1, 3, 5, 10]:
    W_, B_, cls_ = build("MB_CA_R", th)
    r = pair_z(B_, cls_, "DM1", "DM4")
    sweep.append({"thresh": th, "n_kc": len(B_),
                  "mean_in_degree": B_.values.sum(axis=1).mean(), "z_DM1_DM4": r["z"]})
sweep = pd.DataFrame(sweep).round(2)
sweep
"""),
md("""
---
# Part B · Coding theory: why expansion?

## 1. Linear classifier

An **MBON** reading out the KCs can be approximated as:
- a **weighted sum** of the activity of all KCs;
- output "approach" if the sum is above a threshold, otherwise "avoid".

Geometrically this cuts the KC activity space with a **hyperplane**; the two sides of the plane are the two **valences** (good/bad). Such a classifier is a **linear classifier**.

## 2. Cover's theorem

Question: scatter P random points in N-dimensional space and give each a **random** 0/1 label. What is the probability that a single hyperplane separates the two classes perfectly?

Cover's (1965) answer:
- for P < 2N it almost always can;
- for P > 2N it almost never can.

**2N** is the **capacity** of a linear classifier.

Applied to the fly:
- PNs have only about 55 dimensions (55 glomeruli), so a linear readout of PNs could store the arbitrary valence of at most ~110 odours;
- a fly has to learn far more odour associations than that over its life.

## 3. Expansion + sparsification

The solution: first project the 55-dimensional PN signal into ~2,000-dimensional KC space (**expansion**).

Adding dimensions alone is not enough. If the 2,000 KCs were just linear combinations of 55 PNs, the representation would still lie in a 55-dimensional subspace. A **nonlinearity** is required:
- global inhibition from **APL** lets only the strongest ~5% of KCs fire;
- this is **k-winners-take-all**, a form of **sparsification**.

Only after this nonlinearity does the representation really occupy a high-dimensional space, and the capacity rises accordingly.

`kc_code` below implements this step:
1. compute each KC's input, `H = X @ W.T`;
2. keep only the strongest 5% as 1 and set the rest to 0.
"""),
code("""
def kc_code(Wmat, X, sparsity=0.05):
    \"\"\"Expansion + k-winners-take-all. X: odours x glomeruli; Wmat: KCs x glomeruli.
    Returns an odours x KCs matrix of 0/1.\"\"\"
    H = X @ Wmat.T
    thr = np.quantile(H, 1 - sparsity, axis=1, keepdims=True)
    return (H >= thr).astype(float)

def train_acc(X, y):
    \"\"\"Train a linear classifier on (X, y) and return its accuracy on the *training* set
    (this measures separability, not generalisation).\"\"\"
    clf = LogisticRegression(C=1e4, max_iter=5000).fit(X, y)
    return clf.score(X, y)
"""),
md("""
## TODO 5: verify Cover's theorem yourself

Setup:
- input dimension N = 20;
- a random projection `J` expands it to M = 2,000 dimensions, mimicking random PN→KC wiring;
- the number of points P takes the values 20, 40, 80, 160, with completely random labels.

To do: for each P,
1. generate `X`, a (P, N) standard normal matrix, with `rng.standard_normal`;
2. generate `y`, P random 0/1 labels, with `rng.integers(0, 2, P)`;
3. compute the accuracy in the original space, `acc_raw = train_acc(X, y)`, and after expansion, `acc_exp = train_acc(kc_code(J, X), y)`, and print both.

Predict first: once P exceeds 2N = 40, what happens to `acc_raw`? And to `acc_exp`?
"""),
todo("""
# TODO 5: write your code here
N, M = 20, 2000
J = rng.standard_normal((M, N))
""", """
# TODO 5
N, M = 20, 2000
J = rng.standard_normal((M, N))

for P in [20, 40, 80, 160]:
    X = rng.standard_normal((P, N))
    y = rng.integers(0, 2, P)
    acc_raw = train_acc(X, y)
    acc_exp = train_acc(kc_code(J, X), y)
    print(f"P={P:4d}   raw: {acc_raw:.2f}   expanded: {acc_exp:.2f}")
# Above P = 2N the raw accuracy drops below 1; the expanded code still separates every point.
"""),
md("""
---
# Part C · Learning: how MBONs read out the KCs

## 1. Learning happens at the KC→MBON synapse

When a fly smells odour A while being shocked, two things happen:
- **DANs** (PPL1 for punishment, PAM for reward) release **dopamine**;
- dopamine weakens the synapses from the **KCs that are active at that moment** onto a particular MBON.

This **plasticity** occurs only at synapses where "KC active" and "dopamine present" coincide, so it is a **three-factor learning rule**.

The next time the fly smells A, that MBON responds differently, and behaviour changes.

So: **the PN→KC wiring is essentially fixed, and learning happens in the KC→MBON readout.** In the model, an MBON is abstracted as a **trained** linear classifier, and **logistic regression** is such a classifier.

## 2. Why the task matters

A natural first attempt is **odour identification** with a **nearest-centroid** readout: tell apart 200 different odours. In that setting the KC code can look *worse* than the PN code. The reason is that the task is wrong for the question:
- in 55-dimensional PN space, random odours are already well separated from each other, so expansion has nothing to add, and the 5% 0/1 binarisation throws away intensity information.

The task the fly actually faces is **valence learning**:
- many odours, each randomly assigned a good or bad meaning, and the fly must learn the mapping;
- this is exactly the setting of Cover's theorem: P points with random labels.

## 3. Training and test sets

To measure more than memorisation, we train and test on **noisy odour samples**:
- during training the MBON sees 10 noisy samples of each odour;
- at test time it sees **new** noisy samples, mimicking that the same odour smells slightly different every time.

Test accuracy therefore measures **generalisation**.

The tools below are provided:
- `make_odors`: generates n random odours, i.e. glomerular activation vectors. With `food=True`, DM1, DM4, VM3 and VA2 are each active with probability 0.9 (used in Part D);
- `valence_acc`: test accuracy on the random-valence task;
- `shuffled_weights`: curveball within each class, then shuffle the weights within each KC, giving a "shuffled W";
- `W_shufs`: 3 shuffled versions of W.
"""),
code("""
W_real = W_R.values.astype(float)
glom = list(W_R.columns)
n_glom = W_real.shape[1]
FOOD = [glom.index(g) for g in ["DM1", "DM4", "VM3", "VA2"]]

def make_odors(n, food=False, p_active=0.2):
    p = np.full(n_glom, p_active)
    if food:
        p[FOOD] = 0.9
    return (rng.random((n, n_glom)) < p) * rng.lognormal(0, 0.5, (n, n_glom))

def noisy(X, noise, reps):
    Xr = np.repeat(X, reps, axis=0)
    return np.clip(Xr + noise * rng.standard_normal(Xr.shape) * X.mean(), 0, None)

def valence_acc(encode, odors, labels, noise=1.0, n_train=10, n_test=5):
    clf = LogisticRegression(max_iter=3000).fit(encode(noisy(odors, noise, n_train)), np.repeat(labels, n_train))
    return clf.score(encode(noisy(odors, noise, n_test)), np.repeat(labels, n_test))

def shuffled_weights(Wv, cls, rng):
    Bsh = curveball_by_class((Wv > 0).astype(int), cls, 5, rng)
    Wsh = np.zeros_like(Wv)
    for i in range(len(Wv)):
        w = Wv[i][Wv[i] > 0].copy()
        rng.shuffle(w)
        Wsh[i, np.flatnonzero(Bsh[i])] = w
    return Wsh

W_shufs = [shuffled_weights(W_real, cls_R, rng) for _ in range(3)]
print("ready; n_glom =", n_glom)
"""),
md("""
## TODO 6: valence learning: PN vs KC, real W vs shuffled W

To do: for P = 50, 100, 200, 400:
1. generate P generic odours, `odors = make_odors(P)`, and random labels, `labels = rng.integers(0, 2, P)`;
2. compute three accuracies:
   - `pn`: `valence_acc(lambda X: X, odors, labels)`;
   - `real`: with `kc_code(W_real, X)` as the encoder;
   - `shuf`: the mean over the 3 shuffled W in `W_shufs`;
3. store them in a DataFrame `cap` and plot accuracy vs P as three lines.

Write down your predictions before running:
- with 2N ≈ 110, what happens to the PN line at P = 200 and 400?
- which is better, real W or shuffled W?
"""),
todo("""
# TODO 6: write your code here
""", """
# TODO 6
cap = []
for P in [50, 100, 200, 400]:
    odors = make_odors(P)
    labels = rng.integers(0, 2, P)
    pn = valence_acc(lambda X: X, odors, labels)
    real = valence_acc(lambda X: kc_code(W_real, X), odors, labels)
    shuf = np.mean([valence_acc(lambda X, Ws=Ws: kc_code(Ws, X), odors, labels) for Ws in W_shufs])
    cap.append({"P": P, "PN": pn, "KC real W": real, "KC shuffled W": shuf})
cap = pd.DataFrame(cap).set_index("P").round(3)
print(cap)
cap.plot(marker="o", ylabel="test accuracy (random valence)", title="valence learning capacity");
"""),
md("""
**A Python detail:** inside the list comprehension, write `lambda X, Ws=Ws: kc_code(Ws, X)` rather than `lambda X: kc_code(Ws, X)`.

`Ws=Ws` "freezes" the current `Ws` into the lambda when it is defined. Without it, every lambda looks up `Ws` only **when it is called**, and may get the last value of the loop. This trap is called **late binding**.

Here each lambda is called immediately inside the comprehension, so both versions give the same result; but the habit will save you from bugs later.
"""),
md("""
---
# Part D · The "food odour" hypothesis

## Hypothesis → prediction → test

**Hypothesis:** DM1, DM4, VM3 and VA2 preferentially converge onto the same KCs in order to process **food odours** better. These glomeruli respond to vinegar, fruity and other fermentation odours, and are often co-activated.

To test it, the hypothesis must first be turned into a **prediction you can compute**. Two candidate measures:
1. **Learning capacity**: for valence learning on "food-like" odours, does real W do better than shuffled W?
2. **Dimensionality**: how many "effective dimensions" does the KC representation occupy? More dimensions means more separable patterns. Litwin-Kumar et al. (2017) used this measure to analyse the optimal number of KC inputs.

## Participation ratio

Let λ₁, λ₂, … be the **eigenvalues** of the **covariance matrix** of the KC representation R (odours × KCs). Then

  PR = (Σλᵢ)² / Σλᵢ²

Intuition:
- if the variance is spread evenly over d directions, all d eigenvalues are equal and PR = d;
- if all the variance lies in a single direction, PR = 1.

So PR is a continuous version of "effective dimension". It needs no labels; it only looks at the geometry of the representation.

`eigvals` is provided. It uses the **Gram matrix** trick: it finds the eigenvalues of a 1000 × 1000 matrix instead of the 2334 × 2334 covariance, which is much faster. The non-zero eigenvalues are the same up to a constant factor, and PR does not change under a constant factor.
"""),
code("""
def eigvals(R):
    Rc = R - R.mean(axis=0)
    return np.clip(np.linalg.eigvalsh(Rc @ Rc.T), 0, None)
"""),
md("""
## TODO 7: implement the participation ratio and check it against two known answers

To do:
1. complete `participation_ratio`; one line is enough;
2. run two **sanity checks**:
   - isotropic Gaussian data (1000 × 50): PR should be close to 50 (a little lower, because the sample is finite);
   - **rank-1** data, where every row is a multiple of the same vector: PR should be exactly 1.

When you write a new function, **check it against known answers before using it**. This is one of the most important habits in research.
"""),
todo("""
# TODO 7: complete the function
def participation_ratio(R):
    lam = eigvals(R)
    return ___

iso = rng.standard_normal((1000, 50))
rank1 = rng.standard_normal((1000, 1)) * rng.standard_normal((1, 50))
print("isotropic PR:", round(participation_ratio(iso), 1))
print("rank-1    PR:", round(participation_ratio(rank1), 2))
""", """
# TODO 7
def participation_ratio(R):
    lam = eigvals(R)
    return lam.sum() ** 2 / (lam ** 2).sum()

iso = rng.standard_normal((1000, 50))
rank1 = rng.standard_normal((1000, 1)) * rng.standard_normal((1, 50))
print("isotropic PR:", round(participation_ratio(iso), 1))
print("rank-1    PR:", round(participation_ratio(rank1), 2))
"""),
md("""
## TODO 8: test the food-odour hypothesis

To do: for both kinds of odour (`food=False`, generic; `food=True`, food-like):
1. **Dimensionality**:
   - generate 1000 odours, `od = make_odors(1000, food=food)`;
   - compute the PR of `PN` (just `od`), of `KC real` (`kc_code(W_real, od)`), and of `KC shuffled` (mean over the 3 shuffled W);
2. **Capacity**:
   - generate 200 odours and random labels;
   - compute `valence_acc` for `PN`, `KC real` and `KC shuffled` (mean over 3);
3. store one row per odour type, giving two tables, `dim_tab` and `acc_tab`.

Then answer:
- **real W vs shuffled W:** which has the higher PR? Which has the higher capacity?
- **food vs generic odours:** does the "advantage" or "disadvantage" of real W grow for food odours?
- do the results **support**, **contradict**, or **fail to decide** the hypothesis?
"""),
todo("""
# TODO 8: write your code here
""", """
# TODO 8
dim_rows, acc_rows = [], []
for food in [False, True]:
    name = "food" if food else "generic"
    od = make_odors(1000, food=food)
    dim_rows.append({"odors": name,
                     "PN": participation_ratio(od),
                     "KC real W": participation_ratio(kc_code(W_real, od)),
                     "KC shuffled W": np.mean([participation_ratio(kc_code(Ws, od)) for Ws in W_shufs])})
    odors, labels = make_odors(200, food=food), rng.integers(0, 2, 200)
    acc_rows.append({"odors": name,
                     "PN": valence_acc(lambda X: X, odors, labels),
                     "KC real W": valence_acc(lambda X: kc_code(W_real, X), odors, labels),
                     "KC shuffled W": np.mean([valence_acc(lambda X, Ws=Ws: kc_code(Ws, X), odors, labels)
                                               for Ws in W_shufs])})

dim_tab = pd.DataFrame(dim_rows).set_index("odors").round(1)
acc_tab = pd.DataFrame(acc_rows).set_index("odors").round(3)
print("participation ratio\\n", dim_tab, "\\n\\nvalence accuracy (P = 200)\\n", acc_tab)
"""),
md("""
## Interpretation: write your own first, then open the reference

**My interpretation (1–2 sentences each):**

1. Replication (TODO 1–3):
2. Sensitivity (TODO 4):
3. Cover's theorem (TODO 5):
4. Valence capacity (TODO 6):
5. Food hypothesis (TODO 8):

<details>
<summary><b>Reference results (open after writing yours)</b></summary>

The numbers vary a little with the random seed; look at the trends.

- **TODO 1:** left 2341 × 55, mean in-degree 4.36, very close to the right side (2334 × 55, 4.14), so the two sides can be compared directly.
- **TODO 2–3:** all four pairs are significant on the left side, with z ≈ 3.5–8, and none of the 100 shuffles reaches the observed value, so p < 0.01 (the smallest value 100 shuffles can give). DM1–DM4 is the strongest pair on both sides, z ≈ 7–8. **The result replicates in an independent hemisphere.**
- **TODO 4:**
  - the mean in-degree goes from about 5.1 to 3.2 as the threshold rises: **sensitive to the threshold**, so it can only be stated as "about 4 at ≥ 5 synapses";
  - the DM1–DM4 z stays above 5 throughout (≈ 5.3–7.9): **robust**.
- **TODO 5:** at P ≤ 40 both are 1.0; at P = 80 and 160 raw falls to about 0.65–0.8 while expanded stays at 1.0. **They split right at 2N**, which is Cover's theorem.
- **TODO 6:**
  - PN falls from about 0.96 to about 0.65, already dropping below 0.9 near 2N;
  - KC still has about 0.93 at P = 400. **The benefit of expansion appears, once the task is right**;
  - real W and shuffled W are almost identical; real W is at most ~2 percentage points lower.
- **TODO 7:** isotropic ≈ 45–50, rank-1 = 1.00.
- **TODO 8:**
  - PR: for generic odours, real W ≈ 108 vs shuffled W ≈ 123; for food odours ≈ 86 vs 100. **Real wiring has about 12–15% lower dimensionality**;
  - capacity: real W ≈ shuffled W, at most ~2 points lower, for both odour types; food odours are a little harder overall, because they are more similar to each other;
  - **in this toy model, structured wiring does not improve learning of food odours, and it lowers the dimensionality of the representation.** The hypothesis is not supported, but it is **not** ruled out either: the odours here are synthetic, and only one task was tested.

</details>

## What this means for a write-up

You now have a complete story line:
1. **The structure exists and is robust:** present on both sides, independent of the threshold, and not explained by KC class.
2. **The function of expansion:** in valence learning, the KC code far outperforms the PN code, as Cover's theorem predicts.
3. **The cost of the structure:** real wiring lowers the dimensionality of the KC representation by about 12–15%, with no visible loss of capacity on random odours.
4. **An open question:** if the structure has a dimensionality cost, what does it buy? Perhaps a preference for the statistics of **real** food odours (e.g. from the DoOR database), or the needs of **innate** pathways such as the lateral horn.

This becomes the Question and the "What I would do next" of the one-page note in Lesson 6.
"""),
md("""
## Self-test

1. What is **double dipping**? Why does the left-hemisphere test avoid it?
2. How should a result that is "sensitive to the threshold" and one that is "robust" each be worded in a write-up? Give one example of each from this lesson.
3. State **Cover's theorem** in one sentence. Why must expansion be followed by **sparsification** (a nonlinearity)?
4. Why does an odour-identification task with a nearest-centroid readout not show the benefit of expansion, while valence learning does?
5. What does a **participation ratio** of 1 mean, and what does a value of N mean?
6. What are the two biggest limitations of this lesson's food-odour test? Which data would you use next?
"""),
answer("""
**My answers:**

1.
2.
3.
4.
5.
6.
""", """
**Model answers:**

1. Using the same data to choose a hypothesis and to test it, which inflates significance. The pairs were chosen on the right side, and the left side is an independent data set, so testing only those pairs there is a fresh test.
2. Sensitive: report it with its condition, e.g. "KCs receive input from ~4 glomeruli at a threshold of ≥ 5 synapses (3.2–5.1 across thresholds of 1–10)". Robust: state it plainly and give the range, e.g. "DM1–DM4 co-convergence is robust to the threshold (z > 5 for thresholds 1–10)".
3. A linear classifier in N dimensions can separate almost any random labelling of up to ~2N points, and almost none beyond that. A linear expansion alone keeps the data in the original N-dimensional subspace; only a nonlinearity such as k-winners-take-all makes the representation genuinely high-dimensional.
4. Random odours are already far apart in 55-d PN space, so identifying them needs no extra dimensions, and binarisation loses information. Valence learning with random labels is the regime of Cover's theorem, where the number of patterns exceeds the PN capacity and expansion matters.
5. PR = 1: all the variance lies along one direction (effectively one-dimensional). PR = N: the variance is spread evenly over N directions.
6. (a) The odours are synthetic, so they say nothing yet about the statistics of real food odours; measured glomerular responses (e.g. the DoOR database) would fix this. (b) The model is a toy (one readout, one sparsity level, no APL dynamics, one hemisphere), so it should be repeated with other readouts and on the left side.
"""),
]
