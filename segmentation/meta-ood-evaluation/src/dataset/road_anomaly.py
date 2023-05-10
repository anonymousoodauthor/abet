import glob
import os
from collections import namedtuple

import numpy as np
from PIL import Image
from src.dataset.cityscapes import Cityscapes
from src.dataset.lost_and_found import LostAndFound
from torch.utils.data import Dataset


class RoadAnomaly(Dataset):
    RoadAnomaly_class = namedtuple(
        "RoadAnomalyClass", ["name", "id", "train_id", "hasinstances", "ignoreineval", "color"]
    )
    # --------------------------------------------------------------------------------
    # A list of all Lost & Found labels
    # --------------------------------------------------------------------------------
    labels = [
        RoadAnomaly_class("in-distribution", 0, 0, False, False, (144, 238, 144)),
        RoadAnomaly_class("out-distribution", 1, 1, False, False, (255, 102, 102)),
    ]

    train_id_in = 0
    train_id_out = 1
    num_eval_classes = 19
    label_id_to_name = {label.id: label.name for label in labels}
    train_id_to_name = {label.train_id: label.name for label in labels}
    trainid_to_color = {label.train_id: label.color for label in labels}
    label_name_to_id = {label.name: label.id for label in labels}
    cs = Cityscapes()
    mean = cs.mean
    std = cs.std

    def __init__(
        self,
        root="/your/path/to/Abet/datasets/road_anomaly/",
        transform=None,
        split="test",
    ):
        """Load all filenames."""
        self.transform = transform
        self.split = split
        self.root = root
        self.images = []  # list of all raw input images
        self.targets = []  # list of all ground truth TrainIds images

        for filename in os.listdir(os.path.join(self.root, "original")):
            self.images.append(os.path.join(self.root, "original", filename))
            self.targets.append(os.path.join(self.root, "labels", filename[:-3] + "png"))
        self.images = sorted(self.images)
        self.targets = sorted(self.targets)

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        image = Image.open(self.images[i]).convert("RGB")
        target = Image.open(self.targets[i]).convert("L")
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = "Road anomaly Dataset: \n"
        fmt_str += "----Number of images: %d\n" % len(self.images)
        return fmt_str.strip()
