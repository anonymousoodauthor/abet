import os

from src.dataset.cityscapes import Cityscapes
from src.dataset.lost_and_found import LostAndFound
from src.dataset.road_anomaly import RoadAnomaly

# AbeT note: COCO is never used in our code; we only utilize this config for evaluation purposes.

TRAINSETS = ["Cityscapes+COCO"]
VALSETS = ["LostAndFound", "RoadAnomaly"]
MODELS = ["DeepLabV3+_WideResNet38"]

TRAINSET = TRAINSETS[0]
VALSET = VALSETS[0]
MODEL = MODELS[0]
IO = "/your/path/to/Abet/io/"


class cs_coco_roots:
    """
    OoD training roots for Cityscapes + COCO mix
    """

    model_name = MODEL
    init_ckpt = "never_used_by_AbeT"
    cs_root = "/your/path/to/Abet/datasets/cityscapes"
    coco_root = "/your/path/to/Abet/datasets/coco/2017"
    io_root = IO + "meta_ood_" + model_name
    weights_dir = os.path.join(io_root, "weights/")


class laf_roots:
    """
    LostAndFound config class
    """

    model_name = MODEL
    init_ckpt = "never_used_by_AbeT"
    eval_dataset_root = "/your/path/to/Abet/datasets/lost_and_found"
    eval_sub_dir = "laf_eval"
    io_root = os.path.join(IO + "meta_ood_" + model_name, eval_sub_dir)
    weights_dir = os.path.join(io_root, "..", "weights/")


class ra_roots:
    """
    RoadAnomaly config class
    """

    model_name = MODEL
    init_ckpt = "never_used_by_AbeT"
    eval_dataset_root = "/your/path/to/Abet/datasets/road_anomaly"
    eval_sub_dir = "ra_eval"
    io_root = os.path.join(IO + "meta_ood_" + model_name, eval_sub_dir)
    weights_dir = os.path.join(io_root, "..", "weights/")


class params:
    """
    Set pipeline parameters
    """

    training_starting_epoch = 0
    num_training_epochs = 1
    pareto_alpha = 0.9
    ood_subsampling_factor = 0.1
    learning_rate = 1e-5
    crop_size = 480
    val_epoch = num_training_epochs
    batch_size = 8
    entropy_threshold = 0.7


#########################################################################

# never used in AbeT code
class config_training_setup(object):
    """
    Setup config class for training
    If 'None' arguments are passed, the settings from above are applied
    """

    def __init__(self, args):
        if args["TRAINSET"] is not None:
            self.TRAINSET = args["TRAINSET"]
        else:
            self.TRAINSET = TRAINSET
        if self.TRAINSET == "Cityscapes+COCO":
            self.roots = cs_coco_roots
            self.dataset = CityscapesCocoMix
        else:
            print("TRAINSET not correctly specified... bye...")
            exit()
        if args["MODEL"] is not None:
            tmp = getattr(self.roots, "model_name")
            roots_attr = [attr for attr in dir(self.roots) if not attr.startswith("__")]
            for attr in roots_attr:
                if tmp in getattr(self.roots, attr):
                    rep = getattr(self.roots, attr).replace(tmp, args["MODEL"])
                    setattr(self.roots, attr, rep)
        self.params = params
        params_attr = [attr for attr in dir(self.params) if not attr.startswith("__")]
        for attr in params_attr:
            if attr in args:
                if args[attr] is not None:
                    setattr(self.params, attr, args[attr])
        roots_attr = [self.roots.weights_dir]
        for attr in roots_attr:
            if not os.path.exists(attr):
                print("Create directory:", attr)
                os.makedirs(attr)


class config_evaluation_setup(object):
    """
    Setup config class for evaluation
    If 'None' arguments are passed, the settings from above are applied
    """

    def __init__(self, args):
        if args["VALSET"] is not None:
            self.VALSET = args["VALSET"]
        else:
            self.VALSET = VALSET

        if self.VALSET == "LostAndFound":
            self.roots = laf_roots
            self.dataset = LostAndFound
        elif self.VALSET == "RoadAnomaly":
            self.roots = ra_roots
            self.dataset = RoadAnomaly

        else:
            raise ValueError(
                f"Could not match VALSET {self.VALSET} to an original configuration in [LostAndFound|RoadAnomaly] or new config in [Cityscapes]"
            )

        self.params = params
        params_attr = [attr for attr in dir(self.params) if not attr.startswith("__")]
        for attr in params_attr:
            if attr in args:
                if args[attr] is not None:
                    setattr(self.params, attr, args[attr])
        roots_attr = [self.roots.io_root]
        for attr in roots_attr:
            if not os.path.exists(attr):
                print("Create directory:", attr)
                os.makedirs(attr)
