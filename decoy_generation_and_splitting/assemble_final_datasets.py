"""
assemble_final_datasets.py

Final assembly step for Section C -> Section D: produces the two
combined training/test files needed for target-specific ML SF
training (both classification AND regression), per Steps 100/116 of
the MLSF protocol:

    train.sdf + train.csv   (train actives + ALL decoys, combined --
                              matching Step 100: "the actives and
                              inactives are put together in each of
                              these files")
    test.sdf  + test.csv    (test actives + ALL true inactives, combined)

Train/test membership for ACTIVES comes from active_split_assignments.csv
(MolID, Split columns), produced by cluster_actives_for_split.py +
assign_active_split.py. Decoys go ENTIRELY to training and true
inactives go ENTIRELY to test, per the OTS/PPARA-style design used in
this project (see project log for rationale).

CSVs include 'mol_name', 'activity', AND 'pIC50' columns -- 'activity'
and 'pIC50' are NOT read from the docked SDF (Smina docking never
touches these), but instead JOINED BY MolID against external label
sources:
    - actives/inactives: MAOB-data-IC50.csv (real Activity/pIC50 values,
      computed in Section B v2; true inactives keep their REAL measured
      pIC50, not a placeholder -- confirmed against Caba et al. 2024's
      own stated methodology, which applies its pIC50=2 placeholder
      only to molecules lacking real potency data, e.g. decoys).
    - decoys: decoys-data.csv (from build_decoy_labels.py; fixed
      placeholder Activity="Inactive", pIC50=2.0, since decoys have no
      real measured potency at all).

Usage:
    python assemble_final_datasets.py \\
        --actives_docked actives-docked-final.sdf \\
        --active_split active_split_assignments.csv \\
        --inactives_docked inactives-docked-final.sdf \\
        --decoys_docked decoys-docked-final.sdf \\
        --actives_inactives_labels MAOB-data-IC50.csv \\
        --decoy_labels decoys-data.csv \\
        --outdir final_datasets
"""

import argparse
import os
import pandas as pd
from rdkit import Chem


def load_docked_by_id(sdf_path):
    """Load a docked SDF, returning {MolID: mol} using the '_Name' property."""
    result = {}
    n_no_name = 0
    for mol in Chem.SDMolSupplier(sdf_path):
        if mol is None:
            continue
        if mol.HasProp("_Name") and mol.GetProp("_Name"):
            result[mol.GetProp("_Name")] = mol
        else:
            n_no_name += 1
    if n_no_name > 0:
        print(f"  WARNING: {n_no_name} molecule(s) in {sdf_path} have no "
              f"usable _Name/ID property and were skipped.")
    return result


def load_labels_by_id(csv_path, id_col="MolID", activity_col="Activity", pic50_col="pIC50"):
    """
    Load a label CSV, returning {MolID: (activity, pIC50)}. Handles both
    MAOB-data-IC50.csv's column naming (MolID, Activity, pIC50) and
    decoys-data.csv's naming (mol_name, activity, pIC50) automatically.
    """
    df = pd.read_csv(csv_path)
    # auto-detect actual column names present, since the two label
    # sources use slightly different capitalization/naming conventions
    id_col_actual = id_col if id_col in df.columns else "mol_name"
    activity_col_actual = activity_col if activity_col in df.columns else "activity"
    pic50_col_actual = pic50_col if pic50_col in df.columns else "pIC50"

    return {
        row[id_col_actual]: (row[activity_col_actual], row[pic50_col_actual])
        for _, row in df.iterrows()
    }


def write_combined_set(components, labels_by_id, out_sdf_path, out_csv_path):
    """
    Write a combined training or test set to one SDF + one CSV.

    components: list of (mol_dict, mol_ids, expected_activity_label)
    tuples. expected_activity_label is used only as a sanity check
    against the label actually found in labels_by_id, not written
    directly -- the real activity/pIC50 values come from labels_by_id.
    """
    writer = Chem.SDWriter(out_sdf_path)
    csv_rows = []

    for mol_dict, mol_ids, expected_activity_label in components:
        n_missing_mol = 0
        n_missing_label = 0
        n_mismatched_label = 0
        for mol_id in mol_ids:
            mol = mol_dict.get(mol_id)
            if mol is None:
                n_missing_mol += 1
                continue

            label_entry = labels_by_id.get(mol_id)
            if label_entry is None:
                n_missing_label += 1
                continue
            activity, pic50 = label_entry

            if activity != expected_activity_label:
                n_mismatched_label += 1
                # still proceed using the label source's own value, which
                # is authoritative -- but flag the mismatch for review

            writer.write(mol)
            csv_rows.append({"mol_name": mol_id, "activity": activity, "pIC50": pic50})

        if n_missing_mol > 0:
            print(f"  WARNING: {n_missing_mol} MolID(s) (expected label={expected_activity_label}) "
                  f"not found in the docked SDF -- expected for known exclusions.")
        if n_missing_label > 0:
            print(f"  WARNING: {n_missing_label} MolID(s) (expected label={expected_activity_label}) "
                  f"found in the docked SDF but MISSING from the label source -- these "
                  f"were skipped, check for a mismatch between docking outputs and label files.")
        if n_mismatched_label > 0:
            print(f"  WARNING: {n_mismatched_label} MolID(s) had a DIFFERENT activity label "
                  f"in the label source than expected ({expected_activity_label}) -- "
                  f"the label source's value was used; investigate if this count is nonzero.")

    writer.close()

    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(out_csv_path, index=False)

    print(f"  Wrote {len(csv_rows)} molecules -> {out_sdf_path} / {out_csv_path}")
    return len(csv_rows)


def main():
    parser = argparse.ArgumentParser(description="Assemble the combined train/test files for Section D.")
    parser.add_argument("--actives_docked", required=True)
    parser.add_argument("--active_split", required=True, help="CSV with MolID, Split (train/test) columns")
    parser.add_argument("--inactives_docked", required=True)
    parser.add_argument("--decoys_docked", required=True)
    parser.add_argument("--actives_inactives_labels", required=True, help="MAOB-data-IC50.csv (MolID, Activity, pIC50)")
    parser.add_argument("--decoy_labels", required=True, help="decoys-data.csv (mol_name, activity, pIC50)")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading docked actives ...")
    actives_by_id = load_docked_by_id(args.actives_docked)
    print(f"  {len(actives_by_id)} docked actives loaded.")

    print("Loading docked inactives ...")
    inactives_by_id = load_docked_by_id(args.inactives_docked)
    print(f"  {len(inactives_by_id)} docked inactives loaded.")

    print("Loading docked decoys ...")
    decoys_by_id = load_docked_by_id(args.decoys_docked)
    print(f"  {len(decoys_by_id)} docked decoys loaded.")

    print("Loading actives/inactives labels (real Activity + pIC50) ...")
    ai_labels = load_labels_by_id(args.actives_inactives_labels)
    print(f"  {len(ai_labels)} labeled molecules loaded.")

    print("Loading decoy labels (placeholder Activity + pIC50) ...")
    decoy_labels = load_labels_by_id(args.decoy_labels)
    print(f"  {len(decoy_labels)} labeled decoys loaded.")

    # merge both label sources into one lookup (no ID collisions expected,
    # since actives/inactives and decoys use disjoint MolID prefixes)
    all_labels = {**ai_labels, **decoy_labels}

    split_df = pd.read_csv(args.active_split)
    train_active_ids = split_df.loc[split_df["Split"] == "train", "MolID"].tolist()
    test_active_ids = split_df.loc[split_df["Split"] == "test", "MolID"].tolist()
    print(f"\nActive split: {len(train_active_ids)} train / {len(test_active_ids)} test "
          f"(from {args.active_split})")

    all_decoy_ids = list(decoys_by_id.keys())
    all_inactive_ids = list(inactives_by_id.keys())

    print("\n=== Writing combined TRAIN set (train actives + all decoys) ===")
    n_train = write_combined_set(
        [
            (actives_by_id, train_active_ids, "Active"),
            (decoys_by_id, all_decoy_ids, "Inactive"),
        ],
        all_labels,
        os.path.join(args.outdir, "train.sdf"),
        os.path.join(args.outdir, "train.csv"),
    )

    print("\n=== Writing combined TEST set (test actives + all true inactives) ===")
    n_test = write_combined_set(
        [
            (actives_by_id, test_active_ids, "Active"),
            (inactives_by_id, all_inactive_ids, "Inactive"),
        ],
        all_labels,
        os.path.join(args.outdir, "test.sdf"),
        os.path.join(args.outdir, "test.csv"),
    )

    n_train_act = len(train_active_ids)
    n_test_act = len(test_active_ids)
    n_train_inact = len(all_decoy_ids)
    n_test_inact = len(all_inactive_ids)

    print(f"\n=== Final summary ===")
    print(f"Train: {n_train_act} actives + {n_train_inact} inactives (decoys) = {n_train} total written")
    print(f"Test:  {n_test_act} actives + {n_test_inact} inactives (true)    = {n_test} total written")
    print(f"Test active:inactive ratio = 1:{n_test_inact / n_test_act:.2f}")


if __name__ == "__main__":
    main()
