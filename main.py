
# import os
# from data.datasets import load_synthetic_data, load_real_data
# from src.models import get_models
# from src.experiment import run_experiment
# from src.visualize import plot_results

# os.makedirs("results", exist_ok=True)

# models = get_models()

# # Synthetic Dataset
# X_syn, y_syn, _ = load_synthetic_data(noise=0.3)
# syn_results = run_experiment(X_syn, y_syn, models)
# plot_results(
#     syn_results,
#     title="Stability vs RMSE (Synthetic Dataset)",
#     save_path="results/synthetic_results.png"
# )

# # Real Dataset
# X_real, y_real = load_real_data()
# real_results = run_experiment(X_real, y_real, models)
# plot_results(
#     real_results,
#     title="Stability vs RMSE (California Housing Dataset)",
#     save_path="results/real_results.png"
# )

# print("=== Synthetic Dataset Results ===")
# for k, v in syn_results.items():
#     print(k, v)

# print("\n=== Real Dataset Results ===")
# for k, v in real_results.items():
#     print(k, v)

import os
from data.datasets import load_synthetic_data, load_real_data
from src.models import get_models
from src.experiment import run_experiment
from src.visualize import plot_results

def print_results(title, results):
    print(f"\n=== {title} ===")
    for model, metrics in results.items():
        print(
            f"{model:15s} | RMSE: {metrics['rmse']:.4f} | Stability: {metrics['stability']:.6f}"
        )

if __name__ == "__main__":
    print("🔬 Starting Algorithmic Stability Analysis...\n")

    os.makedirs("results", exist_ok=True)

    print("📌 Loading models...")
    models = get_models()

    # Synthetic Dataset
    print("\n📊 Running on Synthetic Dataset...")
    X_syn, y_syn, _ = load_synthetic_data(noise=0.3)
    syn_results = run_experiment(X_syn, y_syn, models)
    plot_results(
        syn_results,
        title="Stability vs RMSE (Synthetic Dataset)",
        save_path="results/synthetic_results.png"
    )
    print_results("Synthetic Dataset Results", syn_results)

    # Real Dataset
    print("\n🌍 Running on Real Dataset (California Housing)...")
    X_real, y_real = load_real_data()
    real_results = run_experiment(X_real, y_real, models)
    plot_results(
        real_results,
        title="Stability vs RMSE (Real Dataset)",
        save_path="results/real_results.png"
    )
    print_results("Real Dataset Results", real_results)

    print("\n✅ Experiment completed successfully.")
    print("📁 Results saved in the 'results/' directory.")
