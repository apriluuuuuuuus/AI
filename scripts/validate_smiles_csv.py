import argparse
import csv
import sys


def main():
    # This validates the exact CSV contract expected by utils.dataset.read_smiles:
    # a header row containing a `smiles` column and RDKit-parseable values.
    parser = argparse.ArgumentParser(description="Validate a MOLEA pretrain SMILES CSV.")
    parser.add_argument("csv_path", help="CSV file with a smiles column.")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--min-rows", type=int, default=8)
    args = parser.parse_args()

    try:
        from rdkit import Chem
    except ImportError:
        print("ERROR: rdkit is not installed in the active environment.", file=sys.stderr)
        return 2

    valid = 0
    invalid = []
    with open(args.csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        # Fail early if the file is a raw .smi file or the column is named
        # differently, because pretrain.py will not be able to read it.
        if not reader.fieldnames or args.smiles_col not in reader.fieldnames:
            print(f"ERROR: missing required column '{args.smiles_col}'.", file=sys.stderr)
            return 1

        for row_idx, row in enumerate(reader, start=2):
            smiles = (row.get(args.smiles_col) or "").strip()
            # RDKit is the source of truth for whether MOLEA can featurize a row.
            if smiles and Chem.MolFromSmiles(smiles) is not None:
                valid += 1
            else:
                invalid.append((row_idx, smiles))

    if invalid:
        for row_idx, smiles in invalid[:20]:
            print(f"ERROR: invalid SMILES at row {row_idx}: {smiles!r}", file=sys.stderr)
        if len(invalid) > 20:
            print(f"ERROR: {len(invalid) - 20} more invalid rows omitted.", file=sys.stderr)
        return 1

    if valid < args.min_rows:
        print(f"ERROR: only {valid} valid rows; need at least {args.min_rows}.", file=sys.stderr)
        return 1

    print(f"OK: {valid} valid SMILES rows in {args.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
