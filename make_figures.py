import os, json, glob
import matplotlib.pyplot as plt

def load_results(results_dir, task_key):
    items = []
    for fp in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            with open(fp, "r") as f:
                obj = json.load(f)
            if obj.get("task") == task_key:
                lam = float(obj.get("fair_lambda", 0.0))
                tm = obj.get("test_metrics", {})
                items.append((lam, tm, os.path.basename(fp)))
        except Exception as e:
            pass
    items.sort(key=lambda x: x[0])
    return items

def plot_node(results_dir, out_png="fig_node_tradeoff.png"):
    items = load_results(results_dir, "node_classification")
    if not items:
        print("No node_classification JSON found in", results_dir)
        return
    lambdas = [x[0] for x in items]
    accs = [x[1].get("accuracy", float("nan")) for x in items]
    dps = [x[1].get("dp_gap", float("nan")) for x in items]
    eos = [x[1].get("eo_gap", float("nan")) for x in items]

    plt.figure()
    ax1 = plt.gca()
    ax1.plot(lambdas, accs, marker="o", label="Accuracy")
    ax1.set_xlabel("fair_lambda")
    ax1.set_ylabel("Accuracy")
    ax2 = ax1.twinx()
    ax2.plot(lambdas, dps, marker="s", linestyle="--", label="DP gap")
    ax2.plot(lambdas, eos, marker="^", linestyle=":", label="EO gap")
    ax2.set_ylabel("Fairness gaps")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    plt.title("Node Classification: Performance vs Fairness")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved", out_png)

def plot_link(results_dir, out_png="fig_link_tradeoff.png"):
    items = load_results(results_dir, "link_prediction")
    if not items:
        print("No link_prediction JSON found in", results_dir)
        return
    lambdas = [x[0] for x in items]
    aucs = [x[1].get("auc", float("nan")) for x in items]
    aps = [x[1].get("ap", float("nan")) for x in items]
    dps = [x[1].get("dp_gap", float("nan")) for x in items]

    plt.figure()
    ax1 = plt.gca()
    ax1.plot(lambdas, aucs, marker="o", label="AUC")
    ax1.plot(lambdas, aps, marker="x", label="AP")
    ax1.set_xlabel("fair_lambda")
    ax1.set_ylabel("AUC / AP")
    ax2 = ax1.twinx()
    ax2.plot(lambdas, dps, marker="s", linestyle="--", label="DP gap")
    ax2.set_ylabel("Fairness gap")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    plt.title("Link Prediction: Performance vs Fairness")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved", out_png)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results")
    args = p.parse_args()
    plot_node(args.results_dir)
    plot_link(args.results_dir)
