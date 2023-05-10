import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from src.helper import counts_array_to_data_list


def twod_to_threed(twod_arr):
    return np.dstack([twod_arr, twod_arr, twod_arr])


def get_crops(image, train_labels, scores, ood_ind=2, border=50):
    IMW, IMH = train_labels.shape
    xx, yy = np.where(train_labels == ood_ind)
    minx = max(0, np.min(xx) - border)
    maxx = min(np.max(xx) + border, IMW)
    miny = max(0, np.min(yy) - border)
    maxy = min(np.max(yy) + border, IMH)
    return (
        image[minx:maxx, miny:maxy, :],
        train_labels[minx:maxx, miny:maxy],
        scores[minx:maxx, miny:maxy],
    )


def plot_score_graphs(root_path):
    plot_names = [name for name in os.listdir(root_path) if name.endswith("png")]
    f, ax = plt.subplots(1, len(plot_names), figsize=(3 * 10, 10))
    for i, name in enumerate(plot_names):
        ax[i].imshow(Image.open(os.path.join(root_path, name)).convert("RGB"))
        ax[i].set_axis_off()
    plt.show()


def plot_curve_comparisons(eval_runs_dict, offset, x_label, y_label):
    plt.clf()
    for name, eval_run in eval_runs_dict.items():
        try:
            xs = eval_run.scores_dict[f"{offset}_recall"]
            ys = eval_run.scores_dict[f"{offset}_precision"]
        except:
            xs = eval_run.scores_dict[f"{offset}_fpr"]
            ys = eval_run.scores_dict[f"{offset}_tpr"]

        style = "solid" if "FT" not in name else "dashed"
        plt.plot(xs, ys, label=f"{name}", linestyle=style)
        print(
            f"{name} -- scores: TPR95 {eval_run.scores_dict['tpr95']:.4f} - FPR95 {eval_run.scores_dict['fpr95']:.4f} - AUPRC {eval_run.scores_dict['auprc']:.4f} - AUROC {eval_run.scores_dict['auroc']:.4f}"
        )

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{offset} curve comparison")
    plt.show()


def plot_curve(xs, ys, title, x_label, y_label, save_path=None, pts=None):
    plt.clf()
    plt.title(title.replace("_", " "))
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if pts is not None:
        inds_to_annotate = np.arange(0, len(pts), len(pts) / 10, dtype=int)

        for i in inds_to_annotate:
            plt.annotate(
                f"{np.round(pts[i], 4)}",
                (xs[i], ys[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )
            plt.scatter(xs[i], ys[i], color="black")
        plt.annotate(
            f"{np.round(pts[-1], 4)}",
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        plt.scatter([xs[-1]], [ys[-1]], color="black")
    plt.plot(xs, ys)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    if save_path is not None:
        plt.savefig(save_path)


def counts_data_to_lists(data):
    ratio_in = np.sum(data["in"]) / (np.sum(data["in"]) + np.sum(data["out"]))
    ratio_out = 1 - ratio_in
    x1 = counts_array_to_data_list(np.array(data["in"]), 1e7 * ratio_in)
    x2 = counts_array_to_data_list(np.array(data["out"]), 1e7 * ratio_out)
    return x1, x2


def plot_counts_histogram(
    data, score_fn_str, run_name_str, num_bins=100, thresh=None, save_path=None
):
    bins = np.linspace(start=0, stop=num_bins, num=num_bins + 1)
    x1, x2 = counts_data_to_lists(data, num_bins)
    plt.clf()
    plt.hist(
        [x1, x2],
        bins,
        label=["ID", "OOD"],
        weights=[np.ones(len(x1)) / len(x1), np.ones(len(x2)) / len(x2)],
    )
    if thresh is not None:
        plt.axvline(thresh * 100, color="black")

    plt.xlabel("buckets")
    plt.ylabel("relative prevalence")
    plt.legend()
    plt.title(f"ID vs OOD for {score_fn_str}\n{run_name_str.replace('_', ' ')}")
    if save_path is not None:
        plt.savefig(save_path)
