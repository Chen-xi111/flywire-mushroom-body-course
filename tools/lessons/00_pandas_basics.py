from nbbuild import md, code, todo, answer

CELLS = [
md("""
# Lesson 0 · pandas crash course: the 10 operations you will use most

Each operation is shown first on a **6-row toy table**, so the syntax is easy to see, and then used straight away on the real FlyWire annotation table `ann`.

Every section follows the same pattern: **explanation → worked example (already written, just run it) → TODO cell (you write it)**.

**Two core objects**
- **DataFrame**: a whole table with rows and columns, like one sheet in Excel.
- **Series**: a single column of a table, i.e. the one-dimensional version of a DataFrame.

> You can do this lesson first, or come back to it whenever a later lesson uses a pandas operation you have not seen.
"""),
md("## 0. Setup: a toy table and the FlyWire table"),
code("""
import pandas as pd

toy = pd.DataFrame({
    "name":       ["PN_a", "PN_b", "KC_1", "KC_2", "KC_3", "APL"],
    "cell_class": ["ALPN", "ALPN", "Kenyon_Cell", "Kenyon_Cell", "Kenyon_Cell", "APL"],
    "nt":         ["acetylcholine", "gaba", "acetylcholine", "acetylcholine", "acetylcholine", "gaba"],
    "side":       ["right", "left", "right", "right", "left", "right"],
    "n_syn":      [120, 80, 15, 22, 9, 3000],
})
ann = pd.read_csv("../data/neuron_annotations.tsv", sep="\\t", low_memory=False)
toy
"""),
md("""
## 1. Look at the table: `.head()`, `.shape`, `.columns`

- `df.head(n)`: the first n rows (5 by default);
- `df.shape`: (number of rows, number of columns). Note there are no parentheses: it is an **attribute**, not a function;
- `df.columns`: all column names.
"""),
code("""
print(toy.shape)
print(toy.columns.tolist())
toy.head(3)
"""),
md("**TODO 1**: print the shape of `ann` and show its first 3 rows."),
todo("""
# TODO 1: write your code here
""", """
# TODO 1
print(ann.shape)
ann.head(3)
"""),
md("""
## 2. Select columns: one pair of brackets `[]` for one column, two `[[ ]]` for several

- `toy["nt"]` returns a **Series** (one column);
- `toy[["name", "nt"]]` returns a **DataFrame** (several columns). The inner `[]` is an ordinary Python list.
- If a column name has no spaces you can also write `toy.nt`, which is the same as `toy["nt"]`.
"""),
code("""
print(toy["nt"])
toy[["name", "nt"]]
"""),
md("**TODO 2**: select the three columns `cell_type`, `top_nt` and `side` from `ann` and show the first 5 rows."),
todo("""
# TODO 2: write your code here
""", """
# TODO 2
ann[["cell_type", "top_nt", "side"]].head(5)
"""),
md("""
## 3. Filter rows with a boolean mask (the single most important operation)

Think of it in two steps:
1. `toy["side"] == "right"` produces a column of True/False values, called a **mask**;
2. `toy[mask]` keeps only the rows where the mask is True.

Usually both steps are written on one line: `toy[toy.side == "right"]`.
"""),
code("""
toy[toy.side == "right"]
"""),
md("**TODO 3**: keep the rows of `ann` with `cell_class == \"MBON\"`, store them as `mbons`, and print the number of rows."),
todo("""
# TODO 3: write your code here
""", """
# TODO 3
mbons = ann[ann.cell_class == "MBON"].copy()
print(len(mbons))
"""),
md("""
## 4. Several conditions: `&` (and), `|` (or), `~` (not). Wrap every condition in parentheses

```python
toy[(toy.side == "right") & (toy.nt == "gaba")]
```

Without the parentheses you get an error, because `&` binds more tightly than `==`.

**`.isin([...])`** tests whether a value equals any item in a list. It is shorter than chaining many `|`.
"""),
code("""
print(toy[(toy.side == "right") & (toy.nt == "acetylcholine")])
toy[toy.cell_class.isin(["ALPN", "APL"])]
"""),
md("**TODO 4**: from `ann`, keep the rows on the **right side** whose `cell_class` is `\"MBON\"` or `\"DAN\"`, and print the number of rows."),
todo("""
# TODO 4: write your code here
""", """
# TODO 4
print(len(ann[(ann.side == "right") & ann.cell_class.isin(["MBON", "DAN"])]))
"""),
md("""
## 5. Summarise a column: `.value_counts()`, `.nunique()`, `.describe()`

- `.value_counts()`: how many times each value appears, sorted from most to least common;
- `.nunique()`: how many distinct values there are;
- `.describe()`: a summary of a numeric column: count, mean, std, min, quartiles and max.
"""),
code("""
print(toy.nt.value_counts())
print(toy.cell_class.nunique())
toy.n_syn.describe()
"""),
md("**TODO 5**: how many different values does `super_class` take in `ann`? How many neurons are there in each?"),
todo("""
# TODO 5: write your code here
""", """
# TODO 5
print(ann.super_class.nunique())
ann.super_class.value_counts()
"""),
md("""
## 6. Create a new column

Assigning to a new column name creates that column. The operation is applied row by row:

```python
toy["strong"] = toy.n_syn > 50
```

> If you add a column to a table you filtered out of a bigger one, pandas may show a `SettingWithCopyWarning`. The fix is to add `.copy()` when filtering: `sub = df[mask].copy()`.
"""),
code("""
toy["strong"] = toy.n_syn > 50
toy["n_syn_x2"] = toy.n_syn * 2
toy
"""),
md("**TODO 6**: from `ann`, keep the Kenyon cells (`cell_class == \"Kenyon_Cell\"`) on the right side as `kcs` (remember `.copy()`). Add a column `is_gamma` equal to `kcs.cell_type == \"KCg-m\"`. Then count how many are True. Hint: `.sum()` counts True values."),
todo("""
# TODO 6: write your code here
""", """
# TODO 6
kcs = ann[(ann.cell_class == "Kenyon_Cell") & (ann.side == "right")].copy()
kcs["is_gamma"] = kcs.cell_type == "KCg-m"
print(kcs.is_gamma.sum())
"""),
md("""
## 7. Text columns: the `.str` methods

For a text column, add `.str` and then call a string method; pandas runs it on every row:
- `.str.startswith("KC")`: does the value start with a given string? Returns True/False;
- `.str.replace("ORN_", "")`: replace text;
- `.str.split("_").str[0]`: split on `_` and take piece number 0.

> Missing values (NaN) give NaN back from `.str` methods. Convert with `.astype(str)` first if that gets in the way.
"""),
code("""
print(toy.name.str.startswith("KC"))
print(toy.name.str.split("_").str[0])
"""),
md("**TODO 7**: for `kcs` from TODO 6, add a column `kc_class` equal to piece 0 of `cell_type` split on `-` (e.g. `KCg-m` → `KCg`), then count the values with `.value_counts()`."),
todo("""
# TODO 7: write your code here
""", """
# TODO 7
kcs["kc_class"] = kcs.cell_type.str.split("-").str[0]
kcs.kc_class.value_counts()
"""),
md("""
## 8. Group and summarise: `.groupby()`

Read it as: **group by column A, then compute some statistic of column B within each group.**

```python
toy.groupby("cell_class").n_syn.mean()        # mean synapse count per class
toy.groupby("cell_class").nt.value_counts()   # transmitter counts within each class
toy.groupby("cell_class").size()              # number of rows per group
```
"""),
code("""
print(toy.groupby("cell_class").n_syn.mean())
toy.groupby("cell_class").nt.value_counts()
"""),
md("**TODO 8**: using the rows of `ann` with `cell_class == \"ALPN\"`, group by `cell_sub_class` and count the `top_nt` values in each group. (This is the bonus question of Lesson 1.)"),
todo("""
# TODO 8: write your code here
""", """
# TODO 8
pns = ann[ann.cell_class == "ALPN"]
pns.groupby("cell_sub_class").top_nt.value_counts()
"""),
md("""
## 9. Sort: `.sort_values()`

```python
toy.sort_values("n_syn")                    # smallest first
toy.sort_values("n_syn", ascending=False)   # largest first
```
"""),
code("""
toy.sort_values("n_syn", ascending=False)
"""),
md("**TODO 9**: among the right-side uniglomerular PNs (`cell_class == \"ALPN\"` and `cell_sub_class == \"uniglomerular\"`), count the neurons of each `cell_type`, sort from largest to smallest and show the top 5. Hint: `groupby(...).size()` returns a Series that you can `.sort_values(ascending=False)`; or simply use `value_counts()`, which is already sorted."),
todo("""
# TODO 9: write your code here
""", """
# TODO 9
upns = ann[(ann.side == "right") & (ann.cell_class == "ALPN") & (ann.cell_sub_class == "uniglomerular")]
upns.groupby("cell_type").size().sort_values(ascending=False).head(5)
# equivalent: upns.cell_type.value_counts().head(5)
"""),
md("""
## 10. Join two tables: `.merge()`

A connection table only has a `pre` and a `post` neuron ID. To know what kind of cell each one is, you have to look it up in the annotation table. `merge` does this lookup automatically, like VLOOKUP in Excel.

```python
edges.merge(toy[["name", "cell_class"]], left_on="pre", right_on="name")
```

This means: match the `pre` column of `edges` against the `name` column of `toy`, and attach the matching row's `cell_class`.
"""),
code("""
edges = pd.DataFrame({"pre": ["PN_a", "PN_a", "PN_b"], "post": ["KC_1", "KC_2", "KC_3"], "syn_count": [7, 12, 5]})
edges.merge(toy[["name", "cell_class"]], left_on="pre", right_on="name")
"""),
md("""
**TODO 10**: merge `edges` with `toy` twice, to get the `cell_class` of the pre neuron and of the post neuron.

Hints:
- In the second merge both tables have a `cell_class` column, so pandas adds the suffixes `_x` and `_y` automatically.
- You can choose your own suffixes with `suffixes=("_pre", "_post")`.
- At the end keep only `pre`, `post`, `syn_count` and the two class columns.
"""),
todo("""
# TODO 10: write your code here
""", """
# TODO 10
lookup = toy[["name", "cell_class"]]
both = (edges
        .merge(lookup, left_on="pre", right_on="name")
        .merge(lookup, left_on="post", right_on="name", suffixes=("_pre", "_post")))
both[["pre", "post", "syn_count", "cell_class_pre", "cell_class_post"]]
"""),
md("""
## Cheat sheet

| What you want | How to write it |
|---|---|
| First rows | `df.head()` |
| One column / several | `df["a"]` / `df[["a", "b"]]` |
| Filter rows | `df[df.a == 1]` |
| Several conditions | `df[(cond1) & (cond2)]`; `\\|` is or, `~` is not |
| Value in a list | `df[df.a.isin([...])]` |
| Counting | `df.a.value_counts()`, `df.a.nunique()`, `len(df)` |
| Summary statistics | `df.a.describe()`, `.mean()`, `.sum()` |
| New column | `df["new"] = ...` (use `.copy()` on filtered tables) |
| Text | `df.a.str.startswith()`, `.str.replace()`, `.str.split("_").str[0]` |
| Group statistics | `df.groupby("a").b.mean()` / `.size()` / `.value_counts()` |
| Sort | `df.sort_values("a", ascending=False)` |
| Lookup join | `df1.merge(df2, left_on="x", right_on="y")` |

**Two common traps**
1. Forgetting parentheses with several conditions: `df[df.a == 1 & df.b == 2]` raises an error.
2. One pair of brackets or two: `df["a"]` is a Series; `df[["a"]]` is a DataFrame with one column.
"""),
]
