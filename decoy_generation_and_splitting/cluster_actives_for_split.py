"""
cluster_actives_for_split.py

Clusters the true MAO-B actives by structural similarity (Butina
algorithm, Tanimoto >= 0.70 on Morgan fingerprints), following the
diversity-sampling OTS approach used for the source protocol's PPARA
example (in place of full AVE, given this project's dataset scale --
see project log for rationale).

Step 1 of this script just reports the cluster-size distribution, so
a sensible test-active target count can be chosen based on real data
rather than guessed blind. Step 2 (actually assigning train/test
labels) is run separately once that target is decided -- see
assign_active_split.py.

Usage:
    python cluster_actives_for_split.py \\
        --actives_smi MAOB-merged-actives.smi \\
        --excluded_csv actives-excluded.csv \\
        --output cluster_assignments.csv
"""

import argparse
import warnings
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit's MorganGenerator deprecation notices

FP_RADIUS = 2
FP_NBITS = 2048
TANIMOTO_THRESHOLD = 0.70


def main():
    parser = argparse.ArgumentParser(description="Cluster actives by structural similarity for OTS train/test split.")
    parser.add_argument("--actives_smi", required=True, help="Path to the actives .smi file (SMILES MolID per line)")
    parser.add_argument("--excluded_csv", default=None, help="Optional: CSV of permanently-excluded MolIDs (e.g. from merge_recovered_docked.py) to skip")
    parser.add_argument("--output", required=True, help="Path for the cluster assignment CSV")
    args = parser.parse_args()

    excluded_ids = set()
    if args.excluded_csv:
        excl_df = pd.read_csv(args.excluded_csv)
        excluded_ids = set(excl_df["MolID"])
        print(f"Loaded {len(excluded_ids)} excluded MolID(s) to skip.")

    mols = []
    mol_ids = []
    with open(args.actives_smi) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            smiles = parts[0]
            mol_id = parts[1] if len(parts) > 1 else None

            if mol_id in excluded_ids:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"  WARNING: could not parse SMILES for {mol_id}, skipping.")
                continue

            mols.append(mol)
            mol_ids.append(mol_id)

    print(f"Loaded {len(mols)} valid actives for clustering "
          f"(after excluding {len(excluded_ids)} permanently-failed molecules).")

    print("Computing Morgan fingerprints ...")
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FP_NBITS) for m in mols]

    print(f"Running Butina clustering (Tanimoto >= {TANIMOTO_THRESHOLD}) ...")
    # Butina needs a distance matrix (1 - similarity) as a flattened lower triangle
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    clusters = Butina.ClusterData(dists, n, 1 - TANIMOTO_THRESHOLD, isDistData=True)

    print(f"Found {len(clusters)} clusters.")

    # Build a MolID -> cluster index mapping, and report the size distribution
    cluster_sizes = [len(c) for c in clusters]
    cluster_sizes_sorted = sorted(cluster_sizes)

    rows = []
    for cluster_idx, cluster_members in enumerate(clusters):
        for member_idx in cluster_members:
            rows.append({
                "MolID": mol_ids[member_idx],
                "ClusterID": cluster_idx,
                "ClusterSize": len(cluster_members),
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.output, index=False)

    n_singletons = sum(1 for s in cluster_sizes if s == 1)
    n_small = sum(1 for s in cluster_sizes if 1 < s <= 3)

    print(f"\n=== Cluster size distribution ===")
    print(f"Total clusters: {len(clusters)}")
    print(f"Singleton clusters (size 1): {n_singletons}")
    print(f"Small clusters (size 2-3): {n_small}")
    print(f"Largest cluster size: {max(cluster_sizes)}")
    print(f"Median cluster size: {pd.Series(cluster_sizes).median()}")
    print(f"\nFull cluster size distribution (smallest 20 shown):")
    print(cluster_sizes_sorted[:20])
    print(f"\nSaved cluster assignments to {args.output}")
    print(
        f"\nNext step: decide a target test-active count based on the "
        f"{n_singletons} available singleton-cluster actives (and "
        f"optionally the {n_small} small-cluster actives too), then run "
        f"assign_active_split.py."
    )


if __name__ == "__main__":
    main()
