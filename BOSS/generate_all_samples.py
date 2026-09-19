from pathlib import Path

from boss_discrete.bif_data import generate_samples_from_bif, load_bif_metadata


NETWORKS = [
    "alarm",
    "alarm3",
    "alarm10",
    "child3",
    "child5",
    "child10",
    "andes",
    "pigs",
]
SAMPLE_SIZES = [200, 300, 500, 1000, 2000, 5000]
BASE_SEED = 42


def main():
    bif_dir = Path("data/bif")
    sample_dir = Path("data/samples")
    for network_index, network in enumerate(NETWORKS):
        metadata = load_bif_metadata(bif_dir / f"{network}.bif")
        for size_index, size in enumerate(SAMPLE_SIZES):
            seed = BASE_SEED + network_index * 100 + size_index
            output = sample_dir / f"{network}_{size}.csv"
            generate_samples_from_bif(metadata, size, seed, output)
            print(f"Generated {output} with seed={seed}")


if __name__ == "__main__":
    main()

