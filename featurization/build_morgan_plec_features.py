"""
build_morgan_plec_features.py

Generates Morgan/ECFP4 fingerprints (ligand-only, no receptor/docked
pose needed) for every molecule in an existing PLEC feature file, and
concatenates them to produce a combined Morgan+PLEC feature file --
following the precedent in Caba et al. 2024 (PARP1), who found that
combining Morgan fingerprints with PLEC features improved
discriminatory power for most algorithms (RF, XGB, SVM classifiers;
RF, XGB, ANN, DNN regressors), with no change for their best-
performing combination (PLEC-only SVM regressor).

Morgan fingerprint parameters: radius=2 (matching "ECFP4", where 4 is
the diameter = 2*radius), 2048 bits -- matching the exact convention
already used elsewhere in this project (Section C's Butina clustering
step used the same radius=2, 2048-bit Morgan fingerprint for Tanimoto
similarity).

SMILES are pulled from two sources (actives/inactives don't share a
file with decoys):
    - MAOB-data-IC50.csv (MolID, SMILES, Activity, pIC50, ...) for
      actives/inactives
    - MAOB-decoys-with-ids.smi (SMILES MolID, one per line) for decoys

Usage:
    python build_morgan_plec_features.py \\
        --plec_features train_PLEC_features.csv \\
        --actives_inactives_data MAOB-data-IC50.csv \\
        --decoys_smi MAOB-decoys-with-ids.smi \\
        --output train_MFPLEC_features.csv
"""

import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048


def load_smiles_lookup(actives_inactives_csv, decoys_smi):
    """Returns {MolID: SMILES} combining both source files."""
    lookup = {}

    df = pd.read_csv(actives_inactives_csv)
    id_col = "MolID" if "MolID" in df.columns else "mol_name"
    for _, row in df.iterrows():
        lookup[row[id_col]] = row["SMILES"]
    print(f"Loaded {len(df)} actives/inactives SMILES from {actives_inactives_csv}.")

    n_decoys = 0
    with open(decoys_smi) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                smiles, mol_id = parts[0], parts[1]
                lookup[mol_id] = smiles
                n_decoys += 1
    print(f"Loaded {n_decoys} decoy SMILES from {decoys_smi}.")

    return lookup


def compute_morgan_fp(smiles):
    """Returns a list of 0/1 ints (length MORGAN_NBITS), or None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
    except Exception:
        return None
    return list(fp)


def main():
    parser = argparse.ArgumentParser(description="Generate Morgan fingerprints and combine with existing PLEC features.")
    parser.add_argument("--plec_features", required=True, help="Existing PLEC feature CSV (e.g. train_PLEC_features.csv)")
    parser.add_argument("--actives_inactives_data", required=True, help="MAOB-data-IC50.csv")
    parser.add_argument("--decoys_smi", required=True, help="MAOB-decoys-with-ids.smi")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    smiles_lookup = load_smiles_lookup(args.actives_inactives_data, args.decoys_smi)

    plec_df = pd.read_csv(args.plec_features)
    print(f"Loaded {len(plec_df)} molecules with PLEC features from {args.plec_features}.")

    plec_cols = [c for c in plec_df.columns if c.startswith("f") and c[1:].isdigit()]
    other_cols = [c for c in plec_df.columns if c not in plec_cols]

    morgan_rows = []
    n_missing_smiles = 0
    n_failed_fp = 0
    kept_indices = []

    for idx, row in plec_df.iterrows():
        mol_id = row["mol_name"]
        smiles = smiles_lookup.get(mol_id)
        if smiles is None:
            n_missing_smiles += 1
            continue
        fp = compute_morgan_fp(smiles)
        if fp is None:
            n_failed_fp += 1
            continue
        morgan_rows.append(fp)
        kept_indices.append(idx)

    if n_missing_smiles > 0:
        print(f"WARNING: {n_missing_smiles} molecule(s) had no SMILES found in either source file -- excluded.")
    if n_failed_fp > 0:
        print(f"WARNING: {n_failed_fp} molecule(s) failed Morgan fingerprint computation -- excluded.")

    morgan_df = pd.DataFrame(morgan_rows, columns=[f"mf{i}" for i in range(MORGAN_NBITS)])
    morgan_df.index = kept_indices

    kept_plec_df = plec_df.loc[kept_indices].reset_index(drop=True)
    morgan_df = morgan_df.reset_index(drop=True)

    # Combined column order: mol_name, PLEC features, Morgan features, then remaining columns (activity, pIC50, ...)
    remaining_other_cols = [c for c in other_cols if c != "mol_name"]
    combined_df = pd.concat([
        kept_plec_df[["mol_name"]],
        kept_plec_df[plec_cols],
        morgan_df,
        kept_plec_df[remaining_other_cols],
    ], axis=1)

    combined_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(combined_df)} molecules ({len(plec_cols)} PLEC + {MORGAN_NBITS} Morgan features) to {args.output}")


if __name__ == "__main__":
    main()
