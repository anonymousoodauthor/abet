"""
Network Initializations
"""

import importlib
import logging

import torch

from network.temperature_networks import BaselineTemperatureSemSegNet, LearnedTemperatureSemSegNet


def get_net(args, criterion):
    """
    Get Network Architecture based on arguments provided
    """
    penultimate_model = args.temperature_model in ["learned", "baseline"]
    net = get_model(
        network=args.arch,
        num_classes=args.dataset_cls.num_classes,
        criterion=criterion,
        penultimate=args.temperature_model in ["learned", "baseline"],
    )
    if penultimate_model:
        if args.temperature_model == "learned":
            net = LearnedTemperatureSemSegNet(net, num_classes=args.dataset_cls.num_classes)
        else:
            net = BaselineTemperatureSemSegNet(net, num_classes=args.dataset_cls.num_classes)

    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info("Model params = {:2.1f}M".format(num_params / 1000000))

    net = net.cuda()
    return net


def wrap_network_in_dataparallel(net, use_apex_data_parallel=False):
    """
    Wrap the network in Dataparallel
    """
    if use_apex_data_parallel:
        # import apex
        # net = apex.parallel.DistributedDataParallel(net)
        net = torch.nn.parallel.DistributedDataParallel(net)
    else:
        net = torch.nn.DataParallel(net)
    return net


def get_model(network, num_classes, criterion, penultimate=False):
    """
    Fetch Network Function Pointer
    """
    module = network[: network.rfind(".")]
    model = network[network.rfind(".") + 1 :]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(num_classes=num_classes, criterion=criterion, penultimate=penultimate)
    return net
