#!/usr/bin/env bash
    CUDA_VISIBLE_DEVICES=1 python evaluation.py \
        --VALSET RoadAnomaly \
        --split test \
        --pixel_eval \
        --temperature_model learned \
        --checkpoint /your/path/to/Abet/weights/best_temperature_model.pth \
        --score_function abet \

# VALSET can be LostAndFound or RoadAnomaly
# split will be test for both
# pixel_eval flag runs only pixel-leval evaluation
# temperature model must be one of [learned|none]
# checkpoint should be a path to the model being evaluated
# score_function must be one of [abet|entropy|msp|max_logit|godin]
