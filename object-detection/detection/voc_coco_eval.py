import pickle
import torch
import argparse
import matplotlib
matplotlib.use('AGG')
from metric_utils import *

recall_level_default = 0.95


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--model', default='faster-rcnn', type=str)
args = parser.parse_args()


concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

# ID data
id_data = pickle.load(open('data/detection/configs/abe_t/random_seed'+'_' +str(args.seed)  +'/inference/voc_custom_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_0.pkl', 'rb'))
ood_data = pickle.load(open('data/detection/configs/abe_t/random_seed' +'_'+str(args.seed)  +'/inference/coco_ood_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_0.pkl', 'rb'))


id = 0
T = 1

id_score = -args.T * torch.logsumexp(torch.stack(id_data['inter_feat'])[:, :-1] / args.T, dim=1).cpu().data.numpy()
ood_score = -args.T * torch.logsumexp(torch.stack(ood_data['inter_feat'])[:, :-1] / args.T, dim=1).cpu().data.numpy()

###########
########
print(len(id_score))
print(len(ood_score))

measures = get_measures(-id_score, -ood_score, plot=False)

print_measures(measures[0], measures[1], measures[2], 'energy')
