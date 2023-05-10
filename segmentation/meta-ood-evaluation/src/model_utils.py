import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from src.model.deepv3 import DeepWV3Plus
from src.model.DualGCNNet import DualSeg_res50
from src.model.temperature_networks import BaselineTemperatureSemSegNet, LearnedTemperatureSemSegNet
from torch.utils.data import DataLoader


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            print(f"skipping loading parameter {k}")
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net


def load_network(params, model_name, num_classes):
    network = None
    print("Checkpoint file:", params.checkpoint)
    print("Temperature model? ", params.temperature_model)
    print("Load model:", model_name, end="", flush=True)

    penultimate_model = params.temperature_model in ["learned", "baseline"]
    print("\npenultimate model: ", penultimate_model)

    if model_name == "DeepLabV3+_WideResNet38":
        network = nn.DataParallel(DeepWV3Plus(num_classes, penultimate=penultimate_model))
    elif model_name == "DualGCNNet_res50":
        network = DualSeg_res50(num_classes)
    else:
        print("\nModel is not known")
        exit()

    if penultimate_model:
        if params.temperature_model == "learned":
            network = LearnedTemperatureSemSegNet(network, num_classes)
        else:
            network = BaselineTemperatureSemSegNet(network, num_classes)

    if torch.cuda.is_available():
        load_state_dict = torch.load(params.checkpoint)["state_dict"]
        # to avoid an issue where instantiated models and loaded models have different ordering of paralellized modules
        no_module_loaded_state_dict = {
            k.replace("module.", ""): v for k, v in load_state_dict.items()
        }
        no_module_net_state_dict = {
            k.replace("module.", ""): v for k, v in network.state_dict().items()
        }
        no_module_to_original_net_key = {
            k.replace("module.", ""): k for k in network.state_dict().keys()
        }

        new_state_dict = {
            no_module_to_original_net_key[no_module_k]: no_module_loaded_state_dict[no_module_k]
            for no_module_k in no_module_net_state_dict
        }
        network.load_state_dict(new_state_dict, strict=True)
        network = network.cuda()
    else:
        checkpoint = torch.load(params.checkpoint, map_location=torch.device("cpu"))
        network = forgiving_state_restore(network, checkpoint["state_dict"])
    network.eval()
    print("\nnetwork type: ", type(network))
    return network


def prediction(net, image):
    if torch.cuda.is_available():
        image = image.cuda()
    with torch.no_grad():
        out = net(image)
    if isinstance(out, tuple):
        out = tuple(t.data.cpu().numpy() for t in out)
    else:
        out = out.data.cpu().numpy()
    return out


class inference(object):
    def __init__(self, params, roots, loader, num_classes, run_name_str, init_net=True):
        self.epoch = params.val_epoch
        self.alpha = params.pareto_alpha
        self.batch_size = params.batch_size
        self.model_name = roots.model_name
        self.batch = 0
        self.batch_max = int(len(loader) / self.batch_size) + (len(loader) % self.batch_size > 0)
        self.loader = loader
        self.run_name_str = run_name_str
        self.batchloader = iter(DataLoader(loader, batch_size=self.batch_size, shuffle=False))
        data_model_name = f"dataset_{self.loader.root.split('/')[-1]}_split_{self.loader.split}_model_{params.checkpoint.split('/')[-1][:-4]}"
        self.probs_root = os.path.join(
            roots.io_root,
            "probs",
            data_model_name,
        )
        self.probs_load_dir = os.path.join(self.probs_root, f"cls_ct_{num_classes}")

        # do not use the meta-ood system of deciphering the path name; instead, just use the checkpoint path passed in params
        self.net = load_network(params, self.model_name, num_classes)

    def probs_gt_load(self, i, load_dir=None, rewrite_preds=False):
        if load_dir is None:
            load_dir = self.probs_load_dir
        if rewrite_preds:
            outputs, gt_train, gt_label, im_path = self.probs_gt_save(i)
        else:
            try:
                filename = os.path.join(load_dir, "probs" + str(i) + ".hdf5")
                f_probs = h5py.File(filename, "r")
                if "outputs" in f_probs.keys():
                    outputs = np.asarray(f_probs["outputs"])
                    outputs = np.squeeze(outputs)
                else:
                    logits = np.asarray(f_probs["logits"])
                    numerators = np.asarray(f_probs["numerators"])
                    temperatures = np.asarray(f_probs["temperatures"])
                    outputs = (logits, numerators, temperatures)
                    outputs = tuple(np.squeeze(out) for out in outputs)

                gt_train = np.asarray(f_probs["gt_train_ids"])
                gt_label = None
                gt_train = np.squeeze(gt_train)
                im_path = f_probs["image_path"][0].decode("utf8")
            except:
                outputs, gt_train, gt_label, im_path = self.prob_gt_calc(i)
        return outputs, gt_train, gt_label, im_path

    def probs_gt_save(self, i, save_dir=None):
        if save_dir is None:
            save_dir = self.probs_load_dir
        if not os.path.exists(save_dir):
            print("Create directory:", save_dir)
            os.makedirs(save_dir)
        outputs, gt_train, gt_label, im_path = self.prob_gt_calc(i)
        file_name = os.path.join(save_dir, "probs" + str(i) + ".hdf5")
        f = h5py.File(file_name, "w")
        if isinstance(outputs, tuple):
            f.create_dataset("logits", data=outputs[0])
            f.create_dataset("numerators", data=outputs[1])
            f.create_dataset("temperatures", data=outputs[2])
        else:
            f.create_dataset("outputs", data=outputs)
        f.create_dataset("gt_train_ids", data=gt_train)
        f.create_dataset("image_path", data=[im_path.encode("utf8")])
        f.close()
        return outputs, gt_train, gt_label, im_path

    def prob_gt_calc(self, i):
        x, y = self.loader[i]
        outputs = prediction(self.net, x.unsqueeze_(0))
        if isinstance(outputs, tuple):
            outputs = tuple(np.squeeze(out) for out in outputs)
        else:
            outputs = np.squeeze(outputs)
        gt_train = y.numpy()
        try:
            gt_label = np.array(Image.open(self.loader.annotations[i]).convert("L"))
        except AttributeError:
            gt_label = np.zeros(gt_train.shape)
        im_path = self.loader.images[i]
        return outputs, gt_train, gt_label, im_path

    # def probs_gt_load_batch(self):
    #     assert (
    #         self.batch_size > 1
    #     ), "Please use batch size > 1 or use function 'probs_gt_load()' instead, bye bye..."
    #     x, y, z, im_paths = next(self.batchloader)
    #     probs = prediction(self.net, x)
    #     gt_train = y.numpy()
    #     gt_label = z.numpy()
    #     self.batch += 1
    #     print("\rBatch %d/%d processed" % (self.batch, self.batch_max))
    #     sys.stdout.flush()
    #     return probs, gt_train, gt_label, im_paths


def probs_gt_load(i, load_dir):
    try:
        filepath = os.path.join(load_dir, "probs" + str(i) + ".hdf5")
        f_probs = h5py.File(filepath, "r")
        probs = np.asarray(f_probs["probabilities"])
        gt_train = np.asarray(f_probs["gt_train_ids"])
        gt_label = np.asarray(f_probs["gt_label_ids"])
        probs = np.squeeze(probs)
        gt_train = np.squeeze(gt_train)
        gt_label = np.squeeze(gt_label)
        im_path = f_probs["image_path"][0].decode("utf8")
    except OSError:
        probs, gt_train, gt_label, im_path = None, None, None, None
        print("No probs file, see src.model_utils")
        exit()
    return probs, gt_train, gt_label, im_path
