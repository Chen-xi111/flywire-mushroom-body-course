from nbbuild import md, code, todo, answer

CELLS = [
md("""
# Lesson 3 · What kind of data is a connectome?

This lesson moves from the "staff list" (the annotation table) to the "wiring list" (the connection table), i.e. the connectome itself.

**Background: how an EM connectome is made**
1. A fly brain is fixed, stained and cut into thousands of ultra-thin sections, each imaged with an **electron microscope (EM)**.
2. The images are aligned into a 3D volume, and neural networks **segment** it into individual neurons.
3. Humans **proofread** the segmentation, fixing wrongly merged and split neurons.
4. Another network **detects synapses**, and each one is assigned to a presynaptic and a postsynaptic neuron. Counting them per neuron pair gives the **edge list**.

You will practise:
- reading what one row of an **edge list** means;
- tracing which KCs one PN connects to (**out-degree**, **synaptic weight**);
- looking at the **synapse count** distribution and why we use a **threshold**;
- turning an edge list into an **adjacency matrix** and computing its **sparsity**.

Run from top to bottom with **Shift + Enter**.
"""),
md("""
## Step 0: how big is the raw data? (provided, optional)

`proofread_connections_783.feather` is the full FlyWire connection table, about 850 MB. It is only needed for this one cell, and only if you downloaded it with `python scripts/download_data.py --full`. We do not load it into memory: `pyarrow.dataset` reads only the file header to show the structure and the number of rows.
"""),
code("""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.dataset as ds

full_path = Path("../data/proofread_connections_783.feather")
if full_path.exists():
    full = ds.dataset(full_path, format="feather")
    print(full.schema)
    print(f"{full.count_rows():,} rows")
else:
    print("Full table not downloaded (optional). It has ~16.8 million rows with columns:")
    print("pre_pt_root_id, post_pt_root_id, neuropil, syn_count")
"""),
md("""
Each row is one **edge**: the number of synapses from one presynaptic neuron to one postsynaptic neuron inside one **neuropil** (brain region). The whole brain has about 17 million edges.

To keep the exercises fast, all edges from **uniglomerular PNs (uPNs) to KCs** (all neuropils, both sides) have been extracted into a small file of about 130 KB, `pn_kc_edges_all.parquet`. Load it together with the annotation table.
"""),
code("""
edges = pd.read_parquet("../data/pn_kc_edges_all.parquet")
ann = pd.read_csv("../data/neuron_annotations.tsv", sep="\\t", low_memory=False)
print(edges.shape)
"""),
md("""
## TODO 1: read the edge list

To do:
1. look at the first 10 rows with `edges.head(10)`;
2. compute the total number of synapses with `edges.syn_count.sum()`;
3. use `.nunique()` to count the distinct presynaptic neurons (`pre_pt_root_id`) and postsynaptic neurons (`post_pt_root_id`).

A **root_id** is the number FlyWire gives each reconstructed neuron. It is **version-specific**: the same neuron can have a different root_id in a different release, because every proofreading edit to its shape gives it a new ID. The annotation and connection tables must therefore come from the same release, here **v783**.
"""),
todo("""
# TODO 1: write your code here
""", """
# TODO 1
print(edges.head(10))
print("total synapses:", edges.syn_count.sum())
print("distinct presynaptic neurons (PNs):", edges.pre_pt_root_id.nunique())
print("distinct postsynaptic neurons (KCs):", edges.post_pt_root_id.nunique())
"""),
md("""
## TODO 2: trace one PN

The cell below picks one **DA1** uPN on the right side (DA1 is the glomerulus that responds to the male pheromone cVA).

To do:
1. keep the rows of `edges` with `pre_pt_root_id == pn_id` as `pn_edges`;
2. sort by `syn_count`, largest first: `.sort_values("syn_count", ascending=False)`, and look at the top 10 rows;
3. print how many KCs it connects to (its **out-degree**) and its total number of synapses.

Bonus: the same KC can appear once per neuropil. Compare `pn_edges.post_pt_root_id.nunique()` with `len(pn_edges)` to see whether there are duplicates.
"""),
code("""
da1 = ann[(ann.cell_type == "DA1_lPN") & (ann.side == "right")]
pn_id = da1.root_id.iloc[0]
print("chosen PN:", pn_id)
"""),
todo("""
# TODO 2: write your code here
""", """
# TODO 2
pn_edges = edges[edges.pre_pt_root_id == pn_id]
print(pn_edges.sort_values("syn_count", ascending=False).head(10))
print("out-degree (distinct KCs):", pn_edges.post_pt_root_id.nunique())
print("total synapses:", pn_edges.syn_count.sum())
print("rows in pn_edges:", len(pn_edges))
"""),
md("""
## TODO 3: the synapse-count distribution and the threshold

FlyWire synapses were **detected automatically** by a neural network on the EM images, so some are **false positives**. An edge with only 1–2 synapses may be a detection error or a chance contact, so such edges are unreliable.

To do:
1. plot a histogram of `edges.syn_count`: `plt.hist(edges.syn_count, bins=range(0, 71))`, with a log-scale y axis;
2. compute the fraction of edges with `syn_count < 5`. Hint: `(edges.syn_count < 5).mean()`, since the mean of True/False values is the fraction of True;
3. compute the fraction of all **synapses** carried by those weak edges.

Think: if you drop edges with `syn_count < 5`, do you lose more edges or more synapses? What does that tell you?
"""),
todo("""
# TODO 3: write your code here
""", """
# TODO 3
plt.hist(edges.syn_count, bins=range(0, 71))
plt.yscale("log")
plt.xlabel("synapses per edge (syn_count)")
plt.ylabel("number of edges")
plt.axvline(5, color="red", linestyle="--")   # threshold = 5
plt.show()

weak = edges.syn_count < 5
print("fraction of edges with < 5 synapses:", round(weak.mean(), 3))
print("fraction of synapses in those edges:", round(edges.syn_count[weak].sum() / edges.syn_count.sum(), 3))
# Many edges but few synapses are lost: the threshold removes mostly noise and little signal.
"""),
md("""
## TODO 4: which neuropil are the edges in?

PNs and KCs should connect in the **calyx** (`MB_CA_L` / `MB_CA_R`), where KC dendrites form claws around PN boutons.

To do:
1. `edges.neuropil.value_counts()`;
2. compute the fraction of **synapses** that are in the calyx. Hint: `edges.neuropil.isin(["MB_CA_L", "MB_CA_R"])` gives a mask.

Think: what might the small fraction of edges outside the calyx be? (Hint: recall the limits of automatic detection from Lesson 1.)
"""),
todo("""
# TODO 4: write your code here
""", """
# TODO 4
print(edges.neuropil.value_counts())
in_calyx = edges.neuropil.isin(["MB_CA_L", "MB_CA_R"])
print("fraction of synapses in calyx:", round(edges.syn_count[in_calyx].sum() / edges.syn_count.sum(), 4))
print(edges.groupby(in_calyx).syn_count.mean())   # edges outside the calyx are much weaker on average
"""),
md("""
## TODO 5: from edge list to adjacency matrix

An **edge list** records only the connections that exist, one per row. An **adjacency matrix** is a table whose rows are presynaptic neurons and columns are postsynaptic neurons; each cell holds the synapse count, or 0 if there is no connection.

Both store the same **directed weighted graph** in different forms.

To do (right calyx only, `syn_count >= 5`):
1. keep `sub`: right calyx (`neuropil == "MB_CA_R"`) and `syn_count >= 5`;
2. pivot into a matrix:
   ```python
   A = sub.pivot_table(index="pre_pt_root_id", columns="post_pt_root_id",
                       values="syn_count", aggfunc="sum", fill_value=0)
   ```
3. print `A.shape`, then compute the **sparsity**, i.e. the fraction of non-zero cells: `(A > 0).values.mean()`;
4. take a look with `plt.imshow(A.values > 0, aspect="auto", interpolation="nearest")`.

Think: each KC connects to about 5 glomeruli on average. Roughly what fraction of non-zero cells would you expect? Does the result match?
"""),
todo("""
# TODO 5: write your code here
""", """
# TODO 5
sub = edges[(edges.neuropil == "MB_CA_R") & (edges.syn_count >= 5)]
A = sub.pivot_table(index="pre_pt_root_id", columns="post_pt_root_id",
                    values="syn_count", aggfunc="sum", fill_value=0)
print("A shape (PN x KC):", A.shape)
print("fraction non-zero:", round((A > 0).values.mean(), 4))
# Rows are individual PNs (~2-3 per glomerulus), so expect roughly (KC in-degree in PNs) / (number of PNs).

plt.imshow(A.values > 0, aspect="auto", interpolation="nearest")
plt.xlabel("KC"); plt.ylabel("PN")
plt.show()
"""),
md("""
## Self-test

1. How does a piece of fly brain become an edge list? Name the 4 steps in order.
2. Why can a root_id not be used across releases?
3. What are the benefit and the cost of a synapse threshold?
4. For the same graph, when is an edge list the better format, and when is an adjacency matrix?
"""),
answer("""
**My answers:**

1.
2.
3.
4.
""", """
**Model answers:**

1. Section and image with EM → align and automatically segment into neurons → proofread the segmentation → automatically detect synapses, assign them to neuron pairs and count them.
2. A root_id names a specific version of a segmented object. Any proofreading edit (merge or split) creates a new root_id, so an ID from one release may not exist, or may point to a different shape, in another.
3. Benefit: it removes most false-positive and incidental contacts, which are many edges but few synapses. Cost: some real weak connections are lost, and every derived number (e.g. inputs per KC) now depends on an arbitrary choice, so it needs a sensitivity analysis.
4. Edge lists are compact for sparse graphs and easy to filter and join (as with pandas). Adjacency matrices make linear algebra and whole-graph operations easy (degrees as row/column sums, matrix products, shuffles), at the cost of storing all the zeros.
"""),
]
