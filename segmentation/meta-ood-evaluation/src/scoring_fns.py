import os

import numpy as np
from scipy.stats import entropy

"""
Score functions need to be bounded [0,1] to work with the fast histogram evaluation functions
For semseg, we match Meta-OoD, pushing ID scores to 0 and OOD scores to 1 to work with their evaluation metricss
"""


def get_score_fn(score_fn_str, args):
    score_fn_dict = {
        "entropy": entropy_score,
        "abet": abet_score,
        "msp": msp_score,
        "max_logit": max_logit_score,
        "godin": godin_score,
    }
    if score_fn_str == "sml":
        return SML_Score(args)
    elif score_fn_str in score_fn_dict:
        return score_fn_dict[score_fn_str]
    else:
        raise KeyError(f"score fn string {score_fn_str} not supported")


# helper fn
def np_softmax(logits):
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=0)
    return probs


# large one hot ID logits --> low score
# small uniform OOD logits --> high score
def entropy_score(logits, numerators, temperatures, num_classes):
    probs = np_softmax(logits)
    ent = entropy(probs, axis=0) / np.log(num_classes)
    return ent


# large ID logits --> very neg log sum exp --> norm towards to 0
# small OOD logits --> small neg log sum exp --> norm towards to 1
def abet_score(logits, numerators, temperatures, num_classes):
    return -(np.log(np.sum(np.exp(logits), axis=0)) / num_classes) + 1


# large one hot ID logits --> high score
# small one hot OOD logits --> low score
# need to invert for our scoring fns
def msp_score(logits, numerators, temperatures, num_classes):
    probs = np_softmax(logits)
    return (-np.max(probs, axis=0)) + 1


def max_logit_score(logits, numerators, temperatures, num_classes):
    max_logits = logits.max(0) / 30
    return -max_logits + 1


# temperatures high on OOD --> no need to invert, already [0,1]
def godin_score(logits, numerators, temperatures, num_classes):
    return temperatures


# need to normalize and invert, similar to MSP
class SML_Score:
    def __init__(self, args, save_dir="/your/path/to/Abet/io/sml_stats") -> None:
        self.args = args
        mean_path = os.path.join(
            save_dir, f"cityscapes_{args['checkpoint'].split('/')[-1][:-4]}_mean.npy"
        )
        var_path = os.path.join(
            save_dir, f"cityscapes_{args['checkpoint'].split('/')[-1][:-4]}_var.npy"
        )
        if os.path.exists(mean_path) and os.path.exists(var_path):
            self.mean_dict = np.load(mean_path, allow_pickle=True).item()
            self.var_dict = np.load(var_path, allow_pickle=True).item()
        else:
            raise KeyError(
                f"SML Score cannot be created as mean path {mean_path} and var path {var_path} do not exist"
            )

    def __call__(self, logits, numerators, temperatures, num_classes):
        assert num_classes == len(self.mean_dict) == len(self.var_dict)
        scores, max_preds = logits.max(0), logits.argmax(0)

        for c in range(num_classes):
            scores = np.where(
                max_preds == c, (scores - self.mean_dict[c]) / (np.sqrt(self.var_dict[c])), scores
            )

        norm_scores = scores / 10  # most norm vals should be within [-.2, .2]
        norm_scores = -norm_scores + 0.3  # need to invert so large SML values to go towards 0

        return norm_scores
