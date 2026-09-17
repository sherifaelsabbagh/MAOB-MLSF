"""
label_and_potency_ic50.py

Unified Section B (v2) pipeline: builds a single, IC50-only actives/
inactives dataset for MAO-B, with both an Active/Inactive label AND a
pIC50 value for every molecule -- designed so classification and
regression models can be trained/evaluated on the EXACT SAME dataset
(see project log for full rationale; this replaces the earlier
two-stage approach of label_*_by_molecule.py + build_potency_dataset.py).

Pipeline:
    1. Parse PubChem raw export: IC50 rows only, qualifiers '=' or '>'
       only, values converted from uM to nM.
    2. Fetch SMILES for the surviving PubChem CIDs via PubChemPy
       (batched, with retry/backoff -- reused from retrieve_smiles_maob.py).
    3. Parse ChEMBL raw export (semicolon-delimited): IC50 rows only,
       nM units only, qualifiers '=' or '>' only. SMILES already present.
    4. Standardize every molecule's SMILES (RDKit: salt stripping,
       charge neutralization, tautomer canonicalization).
    5. Pool ALL measurements (both sources) per molecule by
       standardized SMILES. For each molecule:
         - Consistency check: std-dev of raw IC50 (nM) across all its
           measurements. If >= 2, DISCARD (too inconsistent to trust),
           following the approach in Caba et al. 2024 (PARP1 ML-SF paper).
         - Otherwise: any exact ('=') measurement <= 1000 nM -> Active,
           pIC50 = -log10(mean of exact measurements' value in M).
           All measurements are '>'-censored and > 1000 nM -> Inactive,
           pIC50 = 2.0 (fixed placeholder, matching Caba et al.'s
           convention for decoys/assumed-inactives).
           Any other combination -> DISCARD (ambiguous/contradictory).
    6. Assign sequential MolIDs (MAOB_ACT_####, MAOB_INACT_####) in
       order of appearance.

Outputs:
    MAOB-data-IC50.csv       (MolID, SMILES, Activity, pIC50,
                               N_measurements, Std_IC50_nM, SourceIDs)
    MAOB-actives-IC50.smi    (SMILES MolID, actives only)
    MAOB-inactives-IC50.smi  (SMILES MolID, inactives only)
    MAOB-discarded-IC50.csv  (audit log: which molecules were dropped
                               and why -- failed std-dev check vs.
                               ambiguous/contradictory)

Usage:
    python label_and_potency_ic50.py \\
        --pubchem_raw pubchem_protacxn_P27338_bioactivity_protein.csv \\
        --chembl_raw DOWNLOAD-QqCIYLteG0sCE23LJD8_cWLFGDntSuKaAxPtpW2v8RI_eq_.csv \\
        --outdir sectionB_v2
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

STD_DEV_THRESHOLD_NM = 2.0
ACTIVE_THRESHOLD_NM = 1000.0  # 1 uM
DECOY_PLACEHOLDER_PIC50 = 2.0

PUBCHEMPY_BATCH_SIZE = 100
PUBCHEMPY_SLEEP_BETWEEN_BATCHES = 0.5
PUBCHEMPY_MAX_RETRIES = 3


def parse_pubchem_ic50(pubchem_raw_path):
    print("[1/6] Parsing PubChem raw IC50 data ...")
    df = pd.read_csv(pubchem_raw_path)

    df = df[df["Activity_Type"] == "IC50"]
    df = df[df["Activity_Qualifier"].astype(str).str.strip().isin(["=", ">"])]
    df = df.dropna(subset=["Activity_Value", "Compound_CID"])

    df["value_nM"] = df["Activity_Value"] * 1000.0
    df["qualifier"] = df["Activity_Qualifier"].astype(str).str.strip()
    df["Compound_CID"] = df["Compound_CID"].astype(int)

    records = df[["Compound_CID", "value_nM", "qualifier"]].to_dict("records")
    n_unique_cids = df["Compound_CID"].nunique()
    print(f"  {len(records)} usable IC50 measurement(s) across {n_unique_cids} unique PubChem CID(s).")
    return records, sorted(df["Compound_CID"].unique().tolist())


def fetch_pubchem_smiles(cid_list):
    import pubchempy as pcp

    print(f"[2/6] Fetching SMILES for {len(cid_list)} PubChem CID(s) via PubChemPy ...")
    cid_to_smiles = {}
    n_batches = (len(cid_list) + PUBCHEMPY_BATCH_SIZE - 1) // PUBCHEMPY_BATCH_SIZE

    for i in range(0, len(cid_list), PUBCHEMPY_BATCH_SIZE):
        batch = cid_list[i:i + PUBCHEMPY_BATCH_SIZE]
        batch_num = i // PUBCHEMPY_BATCH_SIZE + 1

        for attempt in range(1, PUBCHEMPY_MAX_RETRIES + 1):
            try:
                compounds = pcp.get_compounds(batch, "cid")
                for c in compounds:
                    if c.canonical_smiles:
                        cid_to_smiles[c.cid] = c.canonical_smiles
                print(f"  Batch {batch_num}/{n_batches}: fetched {len(compounds)} compound(s).", flush=True)
                break
            except Exception as e:
                print(f"  Batch {batch_num}/{n_batches}: attempt {attempt}/{PUBCHEMPY_MAX_RETRIES} "
                      f"failed ({e}), retrying...", flush=True)
                time.sleep(2 ** attempt)
        else:
            print(f"  Batch {batch_num}/{n_batches}: FAILED after {PUBCHEMPY_MAX_RETRIES} attempts, skipping.", flush=True)

        time.sleep(PUBCHEMPY_SLEEP_BETWEEN_BATCHES)

    print(f"  Resolved SMILES for {len(cid_to_smiles)} / {len(cid_list)} CID(s).")
    return cid_to_smiles


def parse_chembl_ic50(chembl_raw_path):
    print("[3/6] Parsing ChEMBL raw IC50 data ...")

    with open(chembl_raw_path, encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","

    df = pd.read_csv(chembl_raw_path, sep=delimiter, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    df = df[df["Standard Type"] == "IC50"]
    df = df[df["Standard Units"] == "nM"]
    df["qualifier"] = df["Standard Relation"].astype(str).str.strip().str.strip("'")
    df = df[df["qualifier"].isin(["=", ">"])]
    df = df.dropna(subset=["Standard Value", "Molecule ChEMBL ID", "Smiles"])

    records = df[["Molecule ChEMBL ID", "Smiles", "Standard Value", "qualifier"]].rename(
        columns={"Molecule ChEMBL ID": "chembl_id", "Smiles": "smiles", "Standard Value": "value_nM"}
    ).to_dict("records")

    n_unique = df["Molecule ChEMBL ID"].nunique()
    print(f"  Detected delimiter: {delimiter!r}")
    print(f"  {len(records)} usable IC50 measurement(s) across {n_unique} unique ChEMBL ID(s).")
    return records


_normalizer = rdMolStandardize.Normalizer()
_frag_chooser = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()
_tautomer_enumerator = rdMolStandardize.TautomerEnumerator()


def standardize_smiles(smiles):
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _normalizer.normalize(mol)
        mol = _frag_chooser.choose(mol)
        mol = _uncharger.uncharge(mol)
        mol = _tautomer_enumerator.Canonicalize(mol)
    except Exception:
        return None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def pool_and_label(all_measurements):
    print("[5/6] Pooling measurements per molecule and applying labeling logic ...")

    df = pd.DataFrame(all_measurements)
    grouped = df.groupby("std_smiles")

    results = []
    discarded = []

    seen_order = []
    seen_set = set()
    for s in df["std_smiles"]:
        if s not in seen_set:
            seen_set.add(s)
            seen_order.append(s)

    for std_smiles in tqdm(seen_order, desc="Labeling molecules"):
        group = grouped.get_group(std_smiles)
        values = group["value_nM"].values
        std_dev = float(np.std(values)) if len(values) > 1 else 0.0
        source_ids = ";".join(sorted(set(group["source_id"])))

        if std_dev >= STD_DEV_THRESHOLD_NM:
            discarded.append({
                "std_smiles": std_smiles, "reason": "inconsistent_replicates",
                "std_dev_nM": std_dev, "n_measurements": len(group),
                "SourceIDs": source_ids,
            })
            continue

        exact_vals = group.loc[group["qualifier"] == "=", "value_nM"].values
        gt_vals = group.loc[group["qualifier"] == ">", "value_nM"].values

        is_active = len(exact_vals) > 0 and np.any(exact_vals <= ACTIVE_THRESHOLD_NM)

        # Confident inactive: EVERY measurement supports inactivity, where
        # support means either (a) an exact ('=') measurement > threshold
        # (a precise, unambiguous weak-potency result -- this was
        # incorrectly NOT accepted as inactive evidence in an earlier
        # version of this script, silently discarding such molecules as
        # "ambiguous" instead), or (b) a '>'-censored measurement whose
        # lower bound already exceeds the threshold.
        exact_supports_inactive = np.all(exact_vals > ACTIVE_THRESHOLD_NM) if len(exact_vals) > 0 else True
        gt_supports_inactive = np.all(gt_vals > ACTIVE_THRESHOLD_NM) if len(gt_vals) > 0 else True
        has_any_valid_inactive_evidence = len(exact_vals) > 0 or len(gt_vals) > 0
        is_confident_inactive = (
            (not is_active) and has_any_valid_inactive_evidence
            and exact_supports_inactive and gt_supports_inactive
        )

        if is_active:
            mean_ic50_M = np.mean(exact_vals) * 1e-9
            pic50 = -np.log10(mean_ic50_M)
            results.append({
                "std_smiles": std_smiles, "Activity": "Active", "pIC50": pic50,
                "N_measurements": len(group), "Std_IC50_nM": std_dev,
                "SourceIDs": source_ids,
            })
        elif is_confident_inactive:
            results.append({
                "std_smiles": std_smiles, "Activity": "Inactive", "pIC50": DECOY_PLACEHOLDER_PIC50,
                "N_measurements": len(group), "Std_IC50_nM": std_dev,
                "SourceIDs": source_ids,
            })
        else:
            discarded.append({
                "std_smiles": std_smiles, "reason": "ambiguous_or_contradictory",
                "std_dev_nM": std_dev, "n_measurements": len(group),
                "SourceIDs": source_ids,
            })

    results_df = pd.DataFrame(results)
    discarded_df = pd.DataFrame(discarded)

    print(f"  Active: {(results_df['Activity']=='Active').sum() if len(results_df) else 0}")
    print(f"  Inactive: {(results_df['Activity']=='Inactive').sum() if len(results_df) else 0}")
    print(f"  Discarded (inconsistent replicates): "
          f"{(discarded_df['reason']=='inconsistent_replicates').sum() if len(discarded_df) else 0}")
    print(f"  Discarded (ambiguous/contradictory): "
          f"{(discarded_df['reason']=='ambiguous_or_contradictory').sum() if len(discarded_df) else 0}")

    return results_df, discarded_df


def main():
    parser = argparse.ArgumentParser(description="Unified IC50-only labeling + potency reconstruction for MAO-B.")
    parser.add_argument("--pubchem_raw", required=True)
    parser.add_argument("--chembl_raw", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pubchem_records, pubchem_cids = parse_pubchem_ic50(args.pubchem_raw)
    cid_to_smiles = fetch_pubchem_smiles(pubchem_cids)
    chembl_records = parse_chembl_ic50(args.chembl_raw)

    print("[4/6] Standardizing SMILES for all measurements ...")
    all_measurements = []
    n_pubchem_unresolved = 0

    for rec in tqdm(pubchem_records, desc="Standardizing PubChem records"):
        smiles = cid_to_smiles.get(rec["Compound_CID"])
        if smiles is None:
            n_pubchem_unresolved += 1
            continue
        std_smiles = standardize_smiles(smiles)
        if std_smiles is None:
            continue
        all_measurements.append({
            "std_smiles": std_smiles, "value_nM": rec["value_nM"],
            "qualifier": rec["qualifier"], "source_id": f"PubChem:{rec['Compound_CID']}",
        })

    for rec in tqdm(chembl_records, desc="Standardizing ChEMBL records"):
        std_smiles = standardize_smiles(rec["smiles"])
        if std_smiles is None:
            continue
        all_measurements.append({
            "std_smiles": std_smiles, "value_nM": rec["value_nM"],
            "qualifier": rec["qualifier"], "source_id": f"ChEMBL:{rec['chembl_id']}",
        })

    if n_pubchem_unresolved > 0:
        print(f"  WARNING: {n_pubchem_unresolved} PubChem measurement(s) skipped "
              f"(CID could not be resolved to a SMILES).")
    print(f"  Total standardized measurements: {len(all_measurements)}")

    results_df, discarded_df = pool_and_label(all_measurements)

    print("[6/6] Assigning MolIDs and writing outputs ...")

    active_mask = results_df["Activity"] == "Active"
    inactive_mask = results_df["Activity"] == "Inactive"

    results_df = results_df.reset_index(drop=True)
    mol_ids = [None] * len(results_df)
    act_counter, inact_counter = 1, 1
    for i, is_active in enumerate(active_mask):
        if is_active:
            mol_ids[i] = f"MAOB_ACT_{act_counter:04d}"
            act_counter += 1
        else:
            mol_ids[i] = f"MAOB_INACT_{inact_counter:04d}"
            inact_counter += 1
    results_df["MolID"] = mol_ids

    data_out = results_df.rename(columns={"std_smiles": "SMILES"})[
        ["MolID", "SMILES", "Activity", "pIC50", "N_measurements", "Std_IC50_nM", "SourceIDs"]
    ]
    data_path = os.path.join(args.outdir, "MAOB-data-IC50.csv")
    data_out.to_csv(data_path, index=False)

    actives_smi_path = os.path.join(args.outdir, "MAOB-actives-IC50.smi")
    with open(actives_smi_path, "w") as f:
        for row in data_out[data_out["Activity"] == "Active"].itertuples():
            f.write(f"{row.SMILES} {row.MolID}\n")

    inactives_smi_path = os.path.join(args.outdir, "MAOB-inactives-IC50.smi")
    with open(inactives_smi_path, "w") as f:
        for row in data_out[data_out["Activity"] == "Inactive"].itertuples():
            f.write(f"{row.SMILES} {row.MolID}\n")

    discard_path = os.path.join(args.outdir, "MAOB-discarded-IC50.csv")
    discarded_df.to_csv(discard_path, index=False)

    print(f"\n=== Final summary ===")
    print(f"Active: {active_mask.sum()}")
    print(f"Inactive: {inactive_mask.sum()}")
    print(f"Discarded: {len(discarded_df)}")
    print(f"\nOutputs written to {args.outdir}/:")
    print(f"  {data_path}")
    print(f"  {actives_smi_path}")
    print(f"  {inactives_smi_path}")
    print(f"  {discard_path}")


if __name__ == "__main__":
    main()
