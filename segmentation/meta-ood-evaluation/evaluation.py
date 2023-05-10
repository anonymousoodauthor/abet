import argparse
import json
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from config import config_evaluation_setup
from meta_classification import meta_classification
from PIL import Image
from src.calc import calc_precision_recall, calc_sensitivity_specificity, get_tpr95_ind
from src.helper import concatenate_metrics, counts_array_to_data_list
from src.imageaugmentations import Compose, Normalize, ToTensor
from src.model_utils import inference
from src.scoring_fns import get_score_fn
from tqdm import tqdm


def plot_curve(xs, ys, title, x_label, y_label, save_path, pts=None):
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
    plt.savefig(save_path)


class eval_pixels(object):
    """
    Evaluate in vs. out separability on pixel-level
    """

    def __init__(self, params, roots, dataset, args, run_name_str):
        self.params = params
        self.epoch = params.val_epoch
        self.alpha = params.pareto_alpha
        self.batch_size = params.batch_size
        self.roots = roots
        self.dataset = dataset
        self.run_name_str = run_name_str
        self.save_dir_data = os.path.join(
            self.roots.io_root, self.run_name_str, "results/scoring_counts_per_pixel"
        )
        self.save_path_data = os.path.join(self.save_dir_data, f"{self.run_name_str}_count_data")
        self.args = args
        self.score_fn_str = args["score_function"]
        self.score_fn = get_score_fn(self.score_fn_str, args)

    def counts(
        self,
        loader,
        num_bins=100,
        save_path=None,
        rewrite_counts=False,
        rewrite_preds=False,
        no_transform_datloader=None,
    ):
        """
        Count the number in-distribution and out-distribution pixels
        and get the networks corresponding confidence scores
        :param loader: dataset loader for evaluation data
        :param num_bins: (int) number of bins for histogram construction
        :param save_path: (str) path where to save the counts data
        :param rewrite: (bool) whether to rewrite the data file if already exists
        """
        print("\nCounting in-distribution and out-distribution pixels")
        if save_path is None:
            save_path = self.save_path_data

        counts_pickle_path = os.path.join(save_path, "counts.p")
        visualize_examples_path = os.path.join(
            save_path,
            "visualized_examples",
        )

        if not os.path.exists(counts_pickle_path) or rewrite_counts or rewrite_preds:
            print("starting counts calculation...")
            if not os.path.exists(save_path):
                print("Create directory", save_path)
                os.makedirs(save_path)
            bins = np.linspace(start=0, stop=1, num=num_bins + 1)
            counts = {
                "in": np.zeros(num_bins, dtype="int64"),
                "out": np.zeros(num_bins, dtype="int64"),
            }
            inf = inference(
                self.params,
                self.roots,
                loader,
                self.dataset.num_eval_classes,
                run_name_str=self.run_name_str,
            )
            pbar = tqdm(range(len(loader)), total=len(loader))
            for i in pbar:
                outputs, gt_train, gt_label, im_path = inf.probs_gt_load(
                    i, rewrite_preds=rewrite_preds
                )
                # logic for handling temperature outputs vs normal model outputs
                if isinstance(outputs, tuple):
                    logits, numerators, temperatures = outputs
                else:
                    logits = outputs
                    numerators, temperatures = None, None
                # logic for seperating scores into respective histograms
                scores = self.score_fn(
                    logits, numerators, temperatures, self.dataset.num_eval_classes
                )
                in_scores = scores[gt_train == self.dataset.train_id_in]
                out_scores = scores[gt_train == self.dataset.train_id_out]

                in_mean, in_std = np.mean(in_scores), np.std(in_scores)
                counts["in"] += np.histogram(in_scores, bins=bins, density=False)[0]

                if len(out_scores) > 0:
                    out_mean, out_std = np.mean(out_scores), np.std(out_scores)
                    counts["out"] += np.histogram(out_scores, bins=bins, density=False)[0]
                else:
                    print(f"no OOD labels found for image {i}")
                    out_mean, out_std = -1.0, -1.0

                pbar.set_description(
                    f"IN mean: {in_mean:.4f} std: {in_std:.4f} | OUT mean: {out_mean:.4f} std: {out_std:.4f}"
                )
            pickle.dump(counts, open(counts_pickle_path, "wb"))
        print("\nCounts data saved to:", save_path + "/")

    def save_visualization_examples(self, x, im_path, gt_train, scores, dir_path):
        name = im_path.split("/")[-1][:-16]
        os.makedirs(os.path.join(dir_path, name), exist_ok=True)
        image = Image.open(im_path).convert("RGB")
        x.save(os.path.join(dir_path, name, "loader_image.png"))
        image.save(os.path.join(dir_path, name, "image.png"))
        np.save(os.path.join(dir_path, name, "gt_train.npy"), gt_train)
        np.save(os.path.join(dir_path, name, "scores.npy"), scores)

    def plot_counts_histogram(self, data, save_path, num_bins=100, thresh=None):
        bins = np.linspace(start=0, stop=num_bins, num=num_bins + 1)
        ratio_in = np.sum(data["in"]) / (np.sum(data["in"]) + np.sum(data["out"]))
        ratio_out = 1 - ratio_in
        x1 = counts_array_to_data_list(np.array(data["in"]), 1e7 * ratio_in)
        x2 = counts_array_to_data_list(np.array(data["out"]), 1e7 * ratio_out)
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
        plt.title(f"ID vs OOD for {self.score_fn_str}\n{self.run_name_str.replace('_', ' ')}")
        plt.savefig(save_path)

    def oodd_metrics_pixel(
        self,
        datloader=None,
        load_path=None,
        rewrite_counts=False,
        rewrite_preds=False,
        no_transform_datloader=None,
    ):
        """
        Calculate 3 OoD detection metrics, namely AUROC, FPR95, AUPRC
        :param datloader: dataset loader
        :param load_path: (str) path to counts data (run 'counts' first)
        :return: OoD detection metrics
        """
        # check for existing counts or create them
        if rewrite_preds:
            assert rewrite_counts

        if load_path is None:
            load_path = self.save_path_data
        if not os.path.exists(os.path.join(load_path, "counts.p")) or rewrite_counts:
            if datloader is None:
                print("Please, specify dataset loader")
                exit()
            self.counts(
                loader=datloader,
                save_path=load_path,
                rewrite_counts=rewrite_counts,
                rewrite_preds=rewrite_preds,
                no_transform_datloader=no_transform_datloader,
            )
        counts_path = os.path.join(load_path, "counts.p")
        counts_hist_path = os.path.join(load_path, "counts_hist.png")

        # load the count data and calculate metrics
        data = pickle.load(open(counts_path, "rb"))
        roc_fpr, roc_tpr, roc_thresholds, auroc = calc_sensitivity_specificity(data, balance=True)
        roc_data = dict(
            roc_fpr=roc_fpr.tolist(),
            roc_tpr=roc_tpr.tolist(),
            roc_thresholds=roc_thresholds.tolist(),
            auroc=float(auroc),
        )

        ind, tpr95, fpr95 = get_tpr95_ind(roc_fpr, roc_tpr)
        ind_data = dict(
            ind=int(ind),
            fpr95=float(fpr95),
            tpr95=float(tpr95),
            threshold=float(roc_thresholds[ind]),
        )
        # precision, recall, thresholds
        pr_precision, pr_recall, pr_thresholds, auprc = calc_precision_recall(data)
        pr_data = dict(
            pr_precision=pr_precision.tolist(),
            pr_recall=pr_recall.tolist(),
            pr_thresholds=pr_thresholds.tolist(),
            auprc=float(auprc),
        )

        # print and save metrics and visualizations
        self.print_thresholds(roc_tpr, roc_fpr, roc_thresholds)
        plot_curve(
            roc_fpr,
            roc_tpr,
            f"roc curve for {self.score_fn_str}\n{self.run_name_str}",
            "fpr",
            "tpr",
            os.path.join(load_path, "roc_curve.png"),
            roc_thresholds * 100,
        )
        plot_curve(
            pr_recall,
            pr_precision,
            f"pr curve for {self.score_fn_str}\n{self.run_name_str}",
            "recall",
            "precision",
            os.path.join(load_path, "pr_curve.png"),
            pr_thresholds * 100,
        )
        self.plot_counts_histogram(data, counts_hist_path, thresh=roc_thresholds[ind])

        print("=" * 50)
        print(f"\nOOD Metrics for {self.score_fn_str} at TPR={tpr95}")
        print("AUC/AUROC:", auroc)
        print("FPR95:", fpr95)
        print("AP/AUPRC:", auprc)
        print("ROC TPR95 THRESHOLD: ", roc_thresholds[ind])

        self.save_metrics(roc_data, ind_data, pr_data, load_path)
        return auroc, fpr95, auprc

    def save_metrics(self, roc_data, ind_data, pr_data, save_dir):
        data_dict = dict()
        data_dict.update(roc_data)
        data_dict.update(pr_data)
        data_dict.update(ind_data)
        path = os.path.join(save_dir, "data.json")
        with open(path, "w") as f:
            json.dump(data_dict, f)
        print(f"data dict dumped to {path}")

    def print_thresholds(self, roc_tpr, roc_fpr, roc_thresholds, vals=[0.5, 0.8, 0.9, 0.95, 0.99]):
        print("=" * 50)
        print("thresholds: ")

        if len(roc_thresholds) > 5:
            for val in vals:
                ind = (np.abs(roc_tpr - val)).argmin()
                print(
                    f"{val} - TPR {roc_tpr[ind]:.5f} - FPR {roc_fpr[ind]:.5f} - THRESH {roc_thresholds[ind]:.5f} "
                )
        else:
            print("not enough thresholds to do percentiles...")
            for i in range(len(roc_thresholds)):
                print(f"TPR {roc_tpr[i]} - FPR {roc_fpr[i]} - THRESH {roc_thresholds[i]} ")

        print("=" * 50)


# never used by AbeT!
def oodd_metrics_segment(params, roots, dataset, metaseg_dir=None):
    """
    Compute number of errors before / after meta classification and compare to baseline
    """
    epoch = params.val_epoch
    alpha = params.pareto_alpha
    thresh = params.entropy_threshold
    num_imgs = len(dataset)
    if epoch == 0:
        load_subdir = "baseline" + "_t" + str(thresh)
    else:
        load_subdir = "epoch_" + str(epoch) + "_alpha_" + str(alpha) + "_t" + str(thresh)
    if metaseg_dir is None:
        metaseg_dir = os.path.join(roots.io_root, "metaseg_io")
    try:
        m, _ = concatenate_metrics(
            metaseg_root=metaseg_dir, num_imgs=num_imgs, subdir="baseline" + "_t" + str(thresh)
        )
        fp_baseline = len([i for i in range(len(m["iou0"])) if m["iou0"][i] == 1])
        m, _ = concatenate_metrics(
            metaseg_root=metaseg_dir,
            num_imgs=num_imgs,
            subdir="baseline" + "_t" + str(thresh) + "_gt",
        )
        fn_baseline = len([i for i in range(len(m["iou"])) if m["iou0"][i] == 1])
    except FileNotFoundError:
        fp_baseline, fn_baseline = None, None
    m, _ = concatenate_metrics(
        metaseg_root=metaseg_dir, num_imgs=num_imgs, subdir=load_subdir + "_gt"
    )
    fn_training = len([i for i in range(len(m["iou"])) if m["iou0"][i] == 1])
    fn_meta, fp_training, fp_meta = meta_classification(
        params=params, roots=roots, dataset=dataset
    ).remove()

    if epoch == 0:
        print("\nOoDD Metrics - Epoch %d - Baseline - Entropy Threshold %.2f" % (epoch, thresh))
    else:
        print(
            "\nOoDD Metrics - Epoch %d - Lambda %.2f - Entropy Threshold %.2f"
            % (epoch, alpha, thresh)
        )
    if fp_baseline is not None and fn_baseline is not None:
        print("Num FPs baseline                       :", fp_baseline)
        print("Num FNs baseline                       :", fn_baseline)
    if epoch > 0:
        print("Num FPs OoD training                   :", fp_training)
        print("Num FNs OoD training                   :", fn_training)
    print("Num FPs OoD training + meta classifier :", fp_meta)
    print("Num FNs OoD training + meta classifier :", fn_meta)
    return fp_baseline, fn_baseline, fp_training, fn_training, fp_meta, fn_meta


def main(args):
    config = config_evaluation_setup(args)

    # add new args of temperature model and checkpoint to params
    config.params.temperature_model = args["temperature_model"]
    config.params.checkpoint = args["checkpoint"]
    run_name_str = f"valset_{args['VALSET']}_split_{args['split']}_{args['temperature_model']}_temperature_model_{args['score_function']}_OODFT_{args['ood_finetune']}_ckpt_{args['checkpoint'].split('/')[-1][:-4]}"

    assert args["pixel_eval"]

    print("")
    transform = Compose([ToTensor(), Normalize(config.dataset.mean, config.dataset.std)])
    datloader = config.dataset(
        root=config.roots.eval_dataset_root, transform=transform, split=args["split"]
    )
    no_transform_datloader = config.dataset(
        root=config.roots.eval_dataset_root, split=args["split"]
    )
    start = time.time()

    """Print important evaluation configuration and perform evaluation"""
    print("")
    print("=" * 50)
    print("EVALUATE MODEL: ", config.roots.model_name)
    print(f"TEMPERATURE MODE: {config.params.temperature_model}")
    print(f"LOADING FROM {config.params.checkpoint}")
    print("SCORING MODE: ", args["score_function"])
    print("=" * 50)

    if args["pixel_eval"]:
        print("\nPIXEL-LEVEL EVALUATION")
        eval_pixels(
            config.params,
            config.roots,
            config.dataset,
            args=args,
            run_name_str=run_name_str,
        ).oodd_metrics_pixel(
            datloader=datloader,
            rewrite_counts=True,
            rewrite_preds=False,
            no_transform_datloader=no_transform_datloader,
        )

    if args["segment_eval"]:
        print("not run by AbeT!")
        print("\nSEGMENT-LEVEL EVALUATION")
        oodd_metrics_segment(config.params, config.roots, datloader)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nFINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == "__main__":
    """Get Arguments and setup config class"""
    parser = argparse.ArgumentParser(description="OPTIONAL argument setting, see also config.py")
    parser.add_argument("-train", "--TRAINSET", nargs="?", type=str)
    parser.add_argument("-val", "--VALSET", nargs="?", type=str)
    parser.add_argument("-model", "--MODEL", nargs="?", type=str)
    parser.add_argument("-epoch", "--val_epoch", nargs="?", type=int)
    parser.add_argument("-alpha", "--pareto_alpha", nargs="?", type=float)
    parser.add_argument("-pixel", "--pixel_eval", action="store_true")
    parser.add_argument("-segment", "--segment_eval", action="store_true")
    ## new args
    parser.add_argument(
        "-temp",
        "--temperature_model",
        type=str,
        default="none",
        help="whether or not to use a temperature model; options are [none, baseline, and learned]",
    )
    parser.add_argument("-ckpt", "--checkpoint", type=str, help="path to model to evaluate")
    parser.add_argument(
        "-score_fn", "--score_function", type=str, help="which score function to use for evaluation"
    )
    parser.add_argument(
        "-split", "--split", type=str, help="what split of the dataset to use", default="test"
    )
    parser.add_argument(
        "-ood_ft",
        "--ood_finetune",
        type=str,
        default="FALSE",
        help="whether or not this model was finetuned on an OOD dataset, to be used in the run name string",
    )
    main(vars(parser.parse_args()))
