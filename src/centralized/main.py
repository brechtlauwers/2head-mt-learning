"""
Main entry point for the thesis experiments.
"""

import argparse
from centralized.experiments import (
    generate_results,
    study_mod_dominance,
    inference_view_drop,
    drop_missing_experiment,
    plot_gradient_similarity,
    plot_lambda_sensitivity,
    plot_gradsim_vs_missing
)


def main():
    parser = argparse.ArgumentParser(description="Incomplete Multi-View Learning Experiments")

    # Define the command line arguments
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["generate_results", "mod_dominance", "inference_drop",
                                 "drop_missing", "grad_sim", "grad_sim_vs_missing",
                                 "lambda_sensitivity"],
                        help="Which experiment to run.")
    parser.add_argument("--dataset", type=str, default="modelnet", 
                        choices=["modelnet", "youtube", "sensit", "har"],
                        help="Dataset to use.")
    parser.add_argument("--imputation_type", type=str, default="zero",
                        choices=["zero", "gaus", "cpm", "dcp"],
                        help="Imputation strategy to fill in missing values.")
    parser.add_argument("--fusion_type", type=str, default="mid_late",
                        choices=["early", "mid", "mid_late", "late"],
                        help="Fusion strategy to evaluate.")
    parser.add_argument("--missing_rate", type=float, default=0.5,
                        help="Missing rate for the views (e.g., 0.5 for 50%).")
    parser.add_argument("--seeds", type=int, nargs='+', default=[50],
                        help="Random seed for reproducibility (e.g., --seeds 50 121 893).")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs.")

    args = parser.parse_args()

    print(f"--- Starting Experiment: {args.experiment.upper()} ---")
    print(f"Dataset: {args.dataset} | Fusion: {args.fusion_type} | Seeds: {args.seeds}")

# Route to the correct function based on the argument
    if args.experiment == "generate_results":
        generate_results(
            dataset_name=args.dataset, 
            experiment_type=args.imputation_type, 
            seeds=args.seeds,
            fusion_type=args.fusion_type
        )
    elif args.experiment == "mod_dominance":
        study_mod_dominance(dataset_name=args.dataset, seeds=args.seeds, epochs=args.epochs)
    elif args.experiment == "inference_drop":
        inference_view_drop(dataset_name=args.dataset, seeds=args.seeds, epochs=args.epochs)
    elif args.experiment == "drop_missing":
        drop_missing_experiment(dataset_name=args.dataset, seeds=args.seeds, epochs=args.epochs)
    
    # Only pass first seed because these functions only run once
    elif args.experiment == "grad_sim":
        plot_gradient_similarity(dataset_name=args.dataset, missing_rate=args.missing_rate,seed=args.seeds[0], epochs=args.epochs)
    elif args.experiment == "grad_sim_vs_missing":
        plot_gradsim_vs_missing(dataset_name=args.dataset, imputation_type=args.imputation_type, seed=args.seeds[0], epochs=args.epochs)
    elif args.experiment == "lambda_sensitivity":
        plot_lambda_sensitivity(dataset_name=args.dataset, imputation_type=args.imputation_type, seed=args.seeds[0], epochs=args.epochs)
    else:
        print("Unknown experiment.")


if __name__ == "__main__":
    main()
