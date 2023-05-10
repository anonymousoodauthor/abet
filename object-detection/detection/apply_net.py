"""
Probabilistic Detectron Inference Script
"""
import core
import json
import os
import sys
import torch
import tqdm
from shutil import copyfile

sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

# Detectron imports
from detectron2.engine import launch
from detectron2.data import build_detection_test_loader, MetadataCatalog

# Project imports
import shutil
from core.evaluation_tools.evaluation_utils import get_train_contiguous_id_to_test_thing_dataset_id_dict
from core.setup import setup_config, setup_arg_parser
from offline_evaluation import compute_average_precision, compute_ood_probabilistic_metrics
from inference.inference_utils import instances_to_json, get_inference_output_dir, build_predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # Setup config
    if args.previous_model_weights is not None:
        print('Overwriting random seed')
        args.random_seed = 999
        os.makedirs(f'data/detection/configs/abe_t/random_seed_{args.random_seed}/', exist_ok=True)
        shutil.copyfile(args.previous_model_weights, f'data/detection/configs/abe_t/random_seed_{args.random_seed}/model_final.pth')
    cfg = setup_config(args,
                       random_seed=args.random_seed,
                       is_testing=True)
    # Make sure only 1 data point is processed at a time. This simulates
    # deployment.
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 32
    cfg.SOLVER.IMS_PER_BATCH = 1

    cfg.MODEL.DEVICE = device.type

    # Set up number of cpu threads#
    torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

    # Create inference output directory and copy inference config file to keep
    # track of experimental settings
    inference_output_dir = get_inference_output_dir(
        cfg['OUTPUT_DIR'],
        args.test_dataset,
        args.inference_config,
        args.image_corruption_level)

    os.makedirs(inference_output_dir, exist_ok=True)
    copyfile(args.inference_config, os.path.join(
        inference_output_dir, os.path.split(args.inference_config)[-1]))

    # Get category mapping dictionary:
    train_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id
    test_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        args.test_dataset).thing_dataset_id_to_contiguous_id

    # If both dicts are equal or if we are performing out of distribution
    # detection, just flip the test dict.
    cat_mapping_dict = get_train_contiguous_id_to_test_thing_dataset_id_dict(
        cfg,
        args,
        train_thing_dataset_id_to_contiguous_id,
        test_thing_dataset_id_to_contiguous_id)

    # Build predictor
    predictor = build_predictor(cfg)
    test_data_loader = build_detection_test_loader(
        cfg, dataset_name=args.test_dataset)

    final_output_list = []

    if not args.eval_only:
        with torch.no_grad():
            with tqdm.tqdm(total=len(test_data_loader)) as pbar:
                for idx, input_im in enumerate(test_data_loader):
                    # Apply corruption

                    outputs = predictor(input_im)

                    if args.visualize:
                        if not os.path.exists(args.savefigdir):
                            os.makedirs(args.savefigdir)

                        predictor.visualize_inference(input_im,
                                                      outputs,
                                                      savedir=args.savefigdir,
                                                      name=str(input_im[0]['image_id']),
                                                      cfg=cfg,
                                                      energy_threshold=8.868)

                    final_output_list.extend(
                        instances_to_json(
                            outputs,
                            input_im[0]['image_id'],
                            cat_mapping_dict))
                    pbar.update(1)

        big_inference_output_dir = inference_output_dir
        with open(os.path.join(big_inference_output_dir, 'coco_instances_results.json'), 'w') as fp:
            json.dump(final_output_list, fp, indent=4, separators=(',', ': '))

    if 'ood' in args.test_dataset:
        compute_ood_probabilistic_metrics.main(args, cfg, min_allowed_score=0)
    else:
        compute_average_precision.main(args, cfg)
        compute_ood_probabilistic_metrics.main(args, cfg, min_allowed_score=0)


if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()
    args.num_gpus = 1

    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
