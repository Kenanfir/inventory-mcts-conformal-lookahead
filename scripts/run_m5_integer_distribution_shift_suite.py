"""Run matched whole-unit demand-shift scenarios with and without ACI."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_m5_distribution_shift as shift


DEFAULT_SEEDS = (0, 1, 4, 7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched integer-order shock and ACI-ablation scenarios")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    for seed in args.seeds:
        for gamma, label in ((shift.m5eval.ACI_GAMMA, "full"), (0.0, "aci_disabled")):
            suffix = "" if seed == 0 else f"_seed{seed}"
            filename = f"m5_integer_distribution_shift_{label}{suffix}.json"
            output = PROJECT_ROOT / "research" / filename
            print(f"=== seed={seed}, {label}, whole-unit orders ===", flush=True)
            shift.run(
                sample_seed=seed,
                n_per_stratum=shift.N_PER_STRATUM,
                opening_inventory_mode="gaussian_target",
                planner_seed_base=0,
                aci_gamma=gamma,
                integer_order_units=True,
                output_path=output,
            )


if __name__ == "__main__":
    main()
