from nbbuild import md, code, todo, answer

CELLS = [
md("""
# Lesson 1 · Neurons, synapses and neurotransmitters in the FlyWire annotations

**Concepts for this lesson** (read the background section of the course README first):
- a neuron is an electrical device: inputs are summed at the membrane, and an **action potential** is fired when a threshold is crossed;
- **synapses** pass the signal on chemically, and the **neurotransmitter** decides whether the effect is **excitatory** (in the fly brain mostly acetylcholine) or **inhibitory** (mostly GABA, and glutamate in many cases);
- in a connectome, the **synapse count** between two neurons is used as a rough proxy for connection strength.

**How to use this notebook**
- Choose a Python kernel that has the packages in `requirements.txt` (top right in VS Code / Jupyter: **Select Kernel**).
- Click a cell and press **Shift + Enter** to run it and move to the next one.
- Run the cells from top to bottom. Later cells use variables defined earlier (such as `ann`).
- Code cells can be run; markdown cells (like this one) are explanations.
"""),
md("""
## Step 0: load the data

- `neuron_annotations.tsv` is the "staff list" of the fly brain: **one row = one neuron**, about 140,000 rows. **One column = one attribute**, such as `cell_class`, `cell_type`, `top_nt` and `side`.
- It lives in `data/`. This notebook is in `notebooks/`, so the path is `../data/...`; `..` means "the folder one level up".
- `ann` is the name we give the table, short for *annotations*.
"""),
code("""
import pandas as pd

ann = pd.read_csv("../data/neuron_annotations.tsv", sep="\\t", low_memory=False)
print("loaded")
"""),
md("""
## Step 1: get to know the table

How many neurons (rows) are there? Which columns?
"""),
code("""
print(len(ann))
print(ann.columns.tolist())
"""),
md("Look at a few real rows, showing only the 6 most important columns. `df[[\"col1\", \"col2\"]]` selects several columns: note the double brackets."),
code("""
ann[["root_id", "super_class", "cell_class", "cell_type", "top_nt", "side"]].head(10)
"""),
md("""
## Step 2: the whole-brain `top_nt` distribution

`top_nt` is the neurotransmitter **predicted** from the electron-microscopy images by a machine-learning model; it is not an experimental measurement. `.value_counts()` counts how often each value appears.
"""),
code("""
ann.top_nt.value_counts()
"""),
md("""
## TODO 1: neurotransmitters of projection neurons (PNs)

PNs have `cell_class == "ALPN"` (Antennal Lobe Projection Neuron). They carry olfactory signals from the **glomeruli** of the antennal lobe to higher brain areas.

To do:
1. Use a boolean mask to keep the rows with `cell_class == "ALPN"` and store them in `pns`;
2. print the number of PNs, i.e. `len(pns)`;
3. look at the `top_nt` distribution of `pns`.

Bonus: use `pns.groupby("cell_sub_class").top_nt.value_counts()` to count uniglomerular and multiglomerular PNs separately.
"""),
todo("""
# TODO 1: write your code here
""", """
# TODO 1
pns = ann[ann.cell_class == "ALPN"]
print("number of PNs:", len(pns))
print(pns.top_nt.value_counts())

# bonus: uniglomerular PNs are mostly cholinergic (excitatory); many multiglomerular PNs are GABAergic (inhibitory)
pns.groupby("cell_sub_class").top_nt.value_counts()
"""),
md("""
## TODO 2: neurotransmitters of Kenyon cells (KCs)

**Before writing any code, write down your prediction as a comment in the cell below:** KCs receive excitatory input from PNs and pass the signal on downstream. Should they be excitatory or inhibitory? Which transmitter would they use?

Then follow TODO 1: change the class to `"Kenyon_Cell"` and call the variable `kcs`.

After running it, compare the result with your prediction. If they disagree, think about where the `top_nt` column comes from.
"""),
todo("""
# My prediction:

# TODO 2: write your code here
""", """
# My prediction: KCs are excitatory and cholinergic (acetylcholine).

# TODO 2
kcs = ann[ann.cell_class == "Kenyon_Cell"]
print("number of KCs:", len(kcs))
print(kcs.top_nt.value_counts())

# Almost every KC is predicted to be dopaminergic! KCs are known from experiments to be cholinergic,
# so this is a failure of the prediction model, not a biological finding:
# always check machine-learning annotations against known biology.
"""),
md("""
## TODO 3: the APL neuron

**APL** (Anterior Paired Lateral neuron): there is only one per hemisphere, and it is huge. KCs excite APL, and APL in turn inhibits all KCs (**global feedback inhibition**). As a result only about 5% of KCs are active at any moment (**sparse coding**).

To do:
1. This time filter on the `cell_type` column, with value `"APL"`;
2. show both the `top_nt` and `side` columns.

Think: does APL's predicted transmitter match its function?
"""),
todo("""
# TODO 3: write your code here
""", """
# TODO 3
apls = ann[ann.cell_type == "APL"]
print("number of APLs:", len(apls))
apls[["top_nt", "side"]]
# One APL per side, predicted GABAergic: consistent with its inhibitory role.
"""),
md("""
## Self-test

1. Why is the action potential called *all-or-none*? If every spike looks the same, how is information encoded?
2. Which neurotransmitters do you expect PNs and KCs to use in the fly central brain? Does the `top_nt` prediction agree? If not, which side do you trust, and why?
3. Using the synapse count as a proxy for synaptic weight: name two things that could go wrong.
"""),
answer("""
**My answers:**

1.
2.
3.
""", """
**Model answers:**

1. A spike is fired only once the membrane potential crosses threshold, and then it always has roughly the same size and shape, so its amplitude carries no information. Information is carried by **spike rate**, **spike timing**, and **which** neurons fire (a population code).
2. Both PNs (uniglomerular) and KCs should be cholinergic, i.e. excitatory. The prediction agrees for most uniglomerular PNs, but labels almost every KC as dopaminergic. Experimental evidence for KCs being cholinergic is strong, so trust the biology; `top_nt` is a model output with its own error modes.
3. (a) Synapse detection has false positives and false negatives, so small counts are noisy. (b) Even correct counts ignore synapse size, receptor type, location on the dendrite and plasticity, so two connections with equal counts can have very different strengths.
"""),
]
