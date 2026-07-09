import importlib


# Minimal imports required for the local MOLEA pretraining path. If one of these
# fails, the smoke run should stop before downloading data or training.
REQUIRED = [
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "rdkit",
    "pandas",
    "yaml",
    "tensorboard",
]


def main():
    # Import each package and print its version so the run log is self-describing.
    for name in REQUIRED:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"OK {name}: {version}")

    import torch
    from rdkit import Chem

    # CPU mode is expected in this local environment.
    print(f"OK torch.cuda.is_available: {torch.cuda.is_available()}")

    # RDKit parsing is the real data-format gate for pretraining CSV files.
    if Chem.MolFromSmiles("CCO") is None:
        raise RuntimeError("RDKit failed to parse a simple test SMILES.")
    print("OK RDKit SMILES parsing")


if __name__ == "__main__":
    main()
