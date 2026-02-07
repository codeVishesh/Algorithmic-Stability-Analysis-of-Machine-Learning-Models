
import matplotlib.pyplot as plt

def plot_results(results, title, save_path):
    models = list(results.keys())
    stability = [results[m]["stability"] for m in models]
    rmse = [results[m]["rmse"] for m in models]

    plt.figure()
    plt.scatter(stability, rmse)
    for i, m in enumerate(models):
        plt.annotate(m, (stability[i], rmse[i]))

    plt.xlabel("Stability (Prediction Variance)")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
