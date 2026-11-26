"""Download the FlyWire data used by the course into data/.

    python scripts/download_data.py          # annotations only (~32 MB), enough for every lesson
    python scripts/download_data.py --full   # also the full connection table (~850 MB)

With --full, the small PN->KC edge file (data/pn_kc_edges_all.parquet, shipped with the
repository) is rebuilt from the full table, so you can check that it was derived correctly.

Sources (both CC BY 4.0):
* Neuron annotations: Schlegel et al. 2024, flyconnectome/flywire_annotations (pinned commit).
* Connections: Dorkenwald et al. 2024, FlyWire v783, Zenodo record 10676866.
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

ANNOTATIONS = dict(
    name="neuron_annotations.tsv",
    url=("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
         "a83b2776d60d5764cef36b927f5f9679c16c47a2/supplemental_files/"
         "Supplemental_file1_neuron_annotations.tsv"),
    md5="fd4fcb205256ca2f80a2eca3a0cc5d87",
)
CONNECTIONS = dict(
    name="proofread_connections_783.feather",
    url="https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1",
    md5=None,  # size is checked instead
    size=852_022_274,
)


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(spec):
    dest = DATA / spec["name"]
    if dest.exists():
        print(f"{dest.name}: already present, skipping")
        return dest
    print(f"downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")

    def progress(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(spec["url"], tmp, reporthook=progress)
    print()
    if spec.get("md5") and md5sum(tmp) != spec["md5"]:
        tmp.unlink()
        sys.exit(f"{dest.name}: checksum mismatch, download removed")
    if spec.get("size") and tmp.stat().st_size != spec["size"]:
        tmp.unlink()
        sys.exit(f"{dest.name}: unexpected file size, download removed")
    tmp.rename(dest)
    return dest


def rebuild_edges():
    """Extract every uniglomerular-PN -> KC edge (all neuropils, both sides)."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as feather

    ann = pd.read_csv(DATA / ANNOTATIONS["name"], sep="\t", low_memory=False)
    pn = ann[(ann.cell_class == "ALPN") & (ann.cell_sub_class == "uniglomerular")]
    kc = ann[ann.cell_class == "Kenyon_Cell"]
    t = feather.read_table(DATA / CONNECTIONS["name"],
                           columns=["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count"])
    mask = pc.and_(pc.is_in(t["pre_pt_root_id"], value_set=pa.array(pn.root_id.values)),
                   pc.is_in(t["post_pt_root_id"], value_set=pa.array(kc.root_id.values)))
    edges = t.filter(mask).to_pandas()
    out = DATA / "pn_kc_edges_all.parquet"
    edges.to_parquet(out, index=False)
    print(f"rebuilt {out.name}: {len(edges):,} edges")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="also download the ~850 MB connection table")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    fetch(ANNOTATIONS)
    if args.full:
        fetch(CONNECTIONS)
        rebuild_edges()
    print("done")


if __name__ == "__main__":
    main()
