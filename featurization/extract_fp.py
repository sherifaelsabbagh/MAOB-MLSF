#!/usr/bin/env python
"""
extract_fp.py

Extracts protein-ligand interaction fingerprints (PLEC, IFP, or SIFP)
from a docked SDF file, matching each fingerprint back to its molecule
by ID (mol_name) -- NOT by list position -- to avoid silent
feature/label misalignment if any molecule fails feature extraction.

Fixes relative to the original script this was adapted from:
    1. CRITICAL: features are matched to the dataset CSV by MolID, not
       by concatenation order. In the original script, any molecule
       that failed feature extraction (returned None) was silently
       skipped from the features list WITHOUT skipping the
       corresponding row in the labels dataframe, causing every
       subsequent molecule's features to be paired with the WRONG
       label via positional pd.concat. This is fixed here by building
       an explicit {MolID: feature_vector} dict and joining on MolID.
    2. Reads SDF directly (oddt.toolkit.readfile("sdf", ...)), matching
       this project's actual docked-file format, rather than requiring
       a mol2 conversion step first.
    3. Reads a CSV dataset (mol_name, activity columns), matching
       train.csv/test.csv's actual format, rather than requiring TSV.
    4. Resumable: if interrupted, results already written to the
       output CSV are skipped on the next run rather than
       recomputed from scratch. Resume detection reads only the
       'mol_name' column of the existing output (via chunked reading),
       not the full file, to avoid re-introducing the memory problem
       described in fix 6 below.
    5. PLEC size confirmed as 4092, depth_ligand=1, depth_protein=5,
       matching the featurization scheme reported in Caba et al. 2024
       (PARP1-specific ML SFs paper) and its companion code exactly.
    6. MEMORY: writes results incrementally in small batches (default
       2000 molecules), rather than accumulating ALL feature rows in
       memory before writing once at the end. The original
       accumulate-then-write-once approach caused an out-of-memory
       crash on a 124,092-molecule training set even at 64GB of
       requested cluster memory -- generation itself completed
       successfully, but the final DataFrame construction/write step
       (holding the complete feature matrix, and on a resumed run,
       ALSO the complete prior output, simultaneously in memory)
       exceeded available memory. Batched incremental writing keeps
       memory usage roughly constant regardless of total dataset size.

Usage:
    python extract_fp.py \\
        --dataset train.csv \\
        --mols train.sdf \\
        --protein rec.pdb \\
        --protein_format pdb \\
        --fp PLEC \\
        --out train_PLEC_features.csv
"""

import argparse
import os
import sys
import pandas as pd
import oddt
from tqdm import tqdm
from oddt.fingerprints import PLEC, SimpleInteractionFingerprint, InteractionFingerprint

PLEC_SIZE = 4092  # matches Caba et al. 2024 (PARP1 paper) and its companion
                   # notebook exactly; previously set to 4096 (a rounded
                   # power-of-2 approximation) -- corrected after direct
                   # confirmation against both the paper's stated methodology
                   # and its actual released code.
PLEC_DEPTH_LIGAND = 1
PLEC_DEPTH_PROTEIN = 5
PLEC_DISTANCE_CUTOFF = 4.5


def get_mol_id(mol):
    """Extract a molecule's ID from its title/_Name property, however ODDT exposes it."""
    if hasattr(mol, "title") and mol.title:
        return mol.title
    return None


def extract_feature(mol, receptor, fp_type):
    """Compute one fingerprint for one molecule against the receptor. Returns None on failure."""
    try:
        if fp_type == "PLEC":
            return PLEC(
                mol, protein=receptor, size=PLEC_SIZE,
                depth_protein=PLEC_DEPTH_PROTEIN, depth_ligand=PLEC_DEPTH_LIGAND,
                distance_cutoff=PLEC_DISTANCE_CUTOFF, sparse=False,
            )
        elif fp_type == "SIFP":
            return SimpleInteractionFingerprint(mol, receptor)
        elif fp_type == "IFP":
            return InteractionFingerprint(mol, receptor)
    except Exception as e:
        print(f"  WARNING: feature extraction failed for a molecule: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract protein-ligand interaction features, safely matched by MolID.")
    parser.add_argument("--dataset", required=True, help="CSV file with mol_name, activity columns (e.g. train.csv)")
    parser.add_argument("--mols", required=True, help="Docked SDF file (e.g. train.sdf)")
    parser.add_argument("--protein", required=True, help="Receptor file")
    parser.add_argument("--protein_format", default="pdb", choices=["pdb", "mol2"], help="Receptor file format (default: pdb)")
    parser.add_argument("--fp", required=True, choices=["PLEC", "SIFP", "IFP"], help="Fingerprint type")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=2000,
                         help="Number of molecules to accumulate in memory before flushing "
                              "to disk (default: 2000). Lower this if memory is still tight; "
                              "raise it for slightly faster I/O on large-memory nodes. This is "
                              "what keeps memory usage roughly CONSTANT regardless of total "
                              "dataset size, unlike the previous version of this script, which "
                              "accumulated ALL rows in memory before writing once at the end -- "
                              "the cause of an out-of-memory crash on the full 124,092-molecule "
                              "training set even at 64GB of requested memory.")
    args = parser.parse_args()

    for f in [args.dataset, args.mols, args.protein]:
        if not os.path.exists(f):
            sys.exit(f"ERROR: file not found: {f}")

    df = pd.read_csv(args.dataset)
    if "mol_name" not in df.columns:
        sys.exit(f"ERROR: {args.dataset} must have a 'mol_name' column.")
    print(f"Loaded {len(df)} molecules from {args.dataset}.")

    label_lookup = df.set_index("mol_name")
    other_cols = [c for c in df.columns if c != "mol_name"]

    receptor = next(oddt.toolkit.readfile(args.protein_format, args.protein))
    receptor.protein = True

    # Resume support: only need the SET of already-computed IDs, not their
    # actual feature values -- avoids loading the (potentially very large)
    # existing output file's full contents into memory just to check
    # membership, which by itself could re-introduce the same memory
    # problem this rewrite is meant to fix.
    computed_ids = set()
    out_exists = os.path.exists(args.out)
    if out_exists:
        for chunk in pd.read_csv(args.out, usecols=["mol_name"], chunksize=50000):
            computed_ids.update(chunk["mol_name"])
        print(f"Found existing output with {len(computed_ids)} molecules already "
              f"processed; these will be skipped (resume mode).")

    ids_to_process = set(df["mol_name"]) - computed_ids
    print(f"{len(ids_to_process)} molecule(s) remaining to process.")

    if len(ids_to_process) == 0:
        print("Nothing to do -- all molecules already have features computed.")
        return

    n_failed = 0
    n_written = 0
    batch = []
    header_written = out_exists  # if resuming, the header already exists in args.out

    def flush_batch(batch):
        """Write one batch of feature rows to disk, joined to labels by mol_name, then clear it."""
        nonlocal header_written, n_written
        if not batch:
            return
        batch_df = pd.DataFrame(batch)
        batch_df = batch_df.merge(
            label_lookup[other_cols], left_on="mol_name", right_index=True, how="left"
        )
        batch_df.to_csv(args.out, mode="a", header=not header_written, index=False)
        header_written = True
        n_written += len(batch_df)
        batch.clear()

    mol_supplier = oddt.toolkit.readfile("sdf", args.mols)
    for mol in tqdm(mol_supplier, desc=f"Generating {args.fp} fingerprints"):
        mol_id = get_mol_id(mol)
        if mol_id is None or mol_id not in ids_to_process:
            continue  # not in our dataset, or already computed in a previous run

        feature = extract_feature(mol, receptor, args.fp)
        if feature is None:
            n_failed += 1
            continue

        row = {"mol_name": mol_id}
        row.update({f"f{i}": v for i, v in enumerate(feature)})
        batch.append(row)

        if len(batch) >= args.batch_size:
            flush_batch(batch)

    flush_batch(batch)  # flush whatever remains (final partial batch)

    if n_failed > 0:
        print(f"\nWARNING: {n_failed} molecule(s) failed {args.fp} feature "
              f"extraction and are EXCLUDED from the output (document this "
              f"as a further exclusion category, consistent with Section C's "
              f"docking-failure exclusions).")

    print(f"\n{args.fp} features saved to {args.out}")
    print(f"Newly written this run: {n_written}")
    print(f"Total molecules with features (including prior runs): {len(computed_ids) + n_written}")


if __name__ == "__main__":
    main()
