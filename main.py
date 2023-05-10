import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math
from tqdm import tqdm
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import copy

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from sklearn.neighbors import KDTree

from torch.utils.data import DataLoader, Subset

from densenet import densenet121
from resnetv2 import KNOWN_MODELS
from resnet import resnet20
from nets import LearnedTemperatureClassificationNet, BaselineClassificationNet
import presets

from generatingloaders import Normalizer

from tqdm import tqdm

r_mean = 125.3/255
g_mean = 123.0/255
b_mean = 113.9/255
r_std = 63.0/255
g_std = 62.1/255
b_std = 66.7/255


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
    
    # Device arguments
    parser.add_argument('--gpu', default = 0, type = int,
                        help = 'gpu index')

    # Model loading arguments
    parser.add_argument('--load-model-path', type = str, 
                        help = 'the path to load the model from')
    parser.add_argument('--save-model-path', type = str, default = "",
                        help = 'the path to save the model to')
    parser.add_argument('--model-dir', default = './models', type = str,
                        help = 'model name for saving')

    # Architecture arguments
    parser.add_argument('--architecture', default = 'densenet', type = str,
                        help = 'underlying architecture (densenet | resnet20 | resnet101 | wideresnet)')

    # Data loading arguments
    parser.add_argument('--data-dir', default='./data', type = str)
    parser.add_argument('--out-dataset', type = str,
                        help = 'out-of-distribution dataset')
    parser.add_argument('--in-dataset', type = str,
                        help = 'in-distribution dataset')
    parser.add_argument('--batch-size', default = 64, type = int,
                        help = 'batch size')

    # Training arguments
    parser.add_argument('--no-train', action='store_false', dest='train')
    parser.add_argument('--weight-decay', default = 0.0001, type = float,
                        help = 'weight decay during training')
    parser.add_argument('--epochs', default = 200, type = int,
                        help = 'number of epochs during training')
    parser.add_argument('--test-during-train', action='store_true', dest='test_during_train')
    parser.add_argument('--baseline', action='store_true', dest='baseline')
    parser.add_argument('--perturb-input', action='store_true', dest='perturb_input')
    parser.add_argument('--inference-mode', type=str)


    # Testing arguments
    parser.add_argument('--no-test', action='store_false', dest='test')
    parser.add_argument('--react', action='store_true', dest='react')
    parser.add_argument('--dice', action='store_true', dest='dice')
    parser.add_argument('--tsne', action='store_true', dest='tsne')
    parser.add_argument('--hist', action='store_true', dest='hist')
    
    
    
    parser.set_defaults(argument=True)
    return parser.parse_args()

def get_datasets(data_dir, out_data_name, in_data_name, batch_size):

    if "CIFAR" in in_data_name or in_data_name == "MNIST":
        crop_size = 32
    elif in_data_name == "Imagenet":
        crop_size = 224


    if in_data_name == "MNIST":
        train_transform = transforms.Compose([
        transforms.RandomCrop(crop_size, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.CenterCrop((crop_size, crop_size)),
            transforms.ToTensor(),
        ])
    elif in_data_name == 'Imagenet':
        train_transform = presets.ClassificationPresetTrain(
                crop_size=224,
                auto_augment_policy='ta_wide',
                random_erase_prob=0.1,
            )

        test_transform = presets.ClassificationPresetEval(
            crop_size=224, resize_size=232
        )
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(crop_size, padding = 4),
            transforms.TrivialAugmentWide(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((r_mean, g_mean, b_mean), (r_std, g_std, b_std)),
        ])
            

        test_transform = transforms.Compose([
            transforms.CenterCrop((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((r_mean, g_mean, b_mean), (r_std, g_std, b_std)),
        ])

    if in_data_name == "CIFAR10":
        train_set_in = torchvision.datasets.CIFAR10(root=f'{data_dir}/cifar10', train=True, download=True, transform=train_transform)
        test_set_in  = torchvision.datasets.CIFAR10(root=f'{data_dir}/cifar10', train=False, download=True, transform=test_transform)
    elif in_data_name == "CIFAR100":
        train_set_in = torchvision.datasets.CIFAR100(root=f'{data_dir}/cifar100', train=True, download=True, transform=train_transform)
        test_set_in  = torchvision.datasets.CIFAR100(root=f'{data_dir}/cifar100', train=False, download=True, transform=test_transform)
    elif in_data_name == "Imagenet":
        train_set_in = torchvision.datasets.ImageFolder(f'{data_dir}/Imagenet/ILSVRC/Data/CLS-LOC/train/', transform=train_transform)
        test_set_in = torchvision.datasets.ImageFolder(f'{data_dir}/Imagenet/ILSVRC/Data/CLS-LOC/val/', transform=test_transform)
    elif in_data_name == "MNIST":
        train_set_in = torchvision.datasets.MNIST(root=f'{data_dir}/mnist', train=True, download=True, transform=train_transform)
        test_set_in  = torchvision.datasets.MNIST(root=f'{data_dir}/mnist', train=False, download=True, transform=test_transform)
    
    if out_data_name in ['SUN', 'Places365', 'iNaturalist']:
        outlier_set  = torchvision.datasets.ImageFolder(f'{data_dir}/{out_data_name}', transform=test_transform)
    elif out_data_name == 'LSUN':
        outlier_set  = torchvision.datasets.LSUN(root=f'{data_dir}/LSUN', classes='test', transform=test_transform)
    elif out_data_name == 'SVHN':
        outlier_set  = torchvision.datasets.SVHN(root=f'{data_dir}/SVHN', download=True, transform=test_transform)
    elif out_data_name == 'Textures':
        outlier_set  = torchvision.datasets.DTD(root=f'{data_dir}/Textures', download=True, transform=test_transform)
    else:
        raise Exception("Out data name not in allowed set: {}".format(out_data_name))
    outlier_loader       =  DataLoader(outlier_set,       batch_size=batch_size, shuffle=False, num_workers=4)
    
    test_indices      = list(range(len(test_set_in)))
    train_loader_in      =  DataLoader(train_set_in,      batch_size=batch_size, shuffle=True,  num_workers=4)
    test_loader_in       =  DataLoader(test_set_in,       batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader_in, test_loader_in, outlier_loader

def main():
    args = get_args()
    
    device           = args.gpu
    
    load_model_path       = args.load_model_path
    save_model_path       = args.save_model_path
    model_dir        = args.model_dir

    architecture     = args.architecture
    baseline = args.baseline
    
    data_dir         = args.data_dir
    out_data_name        = args.out_dataset
    in_data_name        = args.in_dataset
    if in_data_name == "CIFAR10":
        num_classes = 10
    elif in_data_name == "CIFAR100":
        num_classes = 100
    elif in_data_name == "Imagenet":
        num_classes = 1000
    elif in_data_name == "MNIST":
        num_classes = 10
    else:
        raise ValueError(f"in-dataset needs to be one of CIFAR10/100, Imagenet, or MNIST.")
    batch_size       = args.batch_size
    
    train            = args.train
    if train:
        assert save_model_path != ""
    weight_decay     = args.weight_decay
    epochs           = args.epochs
    compute_tsne     = args.tsne
    perturb_input    = args.perturb_input
    react = args.react
    dice = args.dice
    inference_mode = args.inference_mode
    hist = args.hist
    if inference_mode == 'gradnorm':
        batch_size = 1

    test_during_train = args.test_during_train

    test             = args.test

    # Create necessary directories
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if architecture == 'densenet':
        underlying_net = densenet121(pretrained = False)
    elif architecture == 'resnet101':
        underlying_net = KNOWN_MODELS['BiT-S-R101x1']()
    elif architecture == 'resnet20':
        underlying_net = resnet20()
    elif architecture == 'wideresnet':
        underlying_net = WideResNet(depth = 28, num_classes = num_classes, widen_factor = 10)
    
    underlying_net.to(device)

    if baseline:
        net = BaselineClassificationNet(underlying_net, num_classes)
    else:
        net = LearnedTemperatureClassificationNet(underlying_net, num_classes)
    
    net.to(device)

    net = torch.nn.DataParallel(net)

    parameters = []
    for name, parameter in net.named_parameters():
        parameters.append(parameter)

    optimizer = optim.SGD(parameters, lr = 0.1, momentum = 0.9, weight_decay = weight_decay)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100, 150, 180] if in_data_name != "Imagenet" else [20, 30], gamma = 0.1)

    # Load the model (capable of resuming training or inference)
    # from the checkpoint file
    if type(load_model_path) == str:
        checkpoint = torch.load(load_model_path)
        
        epoch_start = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        net.load_state_dict(checkpoint['net'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch_loss = checkpoint['epoch_loss']
    else:
        epoch_start = 0
        epoch_loss = None

    #get outlier data
    train_data, test_data, open_data = get_datasets(data_dir, out_data_name, in_data_name, batch_size)
  
    criterion = nn.CrossEntropyLoss()

    # Train the model
    if train:
        net.train()
        
        num_batches = len(train_data)
        
        epoch_bar = tqdm(total = num_batches * epochs, initial = num_batches * epoch_start)
        for epoch in range(epoch_start, epochs):
            total_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_data):

                if epoch_loss is None:
                    epoch_bar.set_description(f'Training | Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{num_batches}')
                else:
                    epoch_bar.set_description(f'Training | Epoch {epoch + 1}/{epochs} | Epoch loss = {epoch_loss:0.2f} | Batch {batch_idx + 1}/{num_batches}')
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                if in_data_name == "MNIST":
                    inputs = inputs.repeat(1, 3, 1, 1)
                
                if baseline:
                    logits, _, _ = net(inputs, T = 1)
                else:
                    logits, _, _ = net(inputs)

                loss = criterion(logits, targets)
                loss.backward()
                
                optimizer.step()
                total_loss += loss.item()
                
                epoch_bar.update()
            
            epoch_loss = total_loss
            scheduler.step()

            checkpoint = {
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'net': net.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch_loss': epoch_loss,
            }
            torch.save(checkpoint, save_model_path) # For continuing training or inference

            if test_during_train:
                net.eval()
                underlying_net = torch.nn.DataParallel(underlying_net)

                if inference_mode == 'deep_nearest_neighbors' or react or dice:
                    id_train_representations = getRepresentations(underlying_net, device, train_data)
                    if dice:
                        net.module.calculate_mask(id_train_representations)
                    elif inference_mode == 'deep_nearest_neighbors':
                        id_test_representations = getRepresentations(underlying_net, device, test_data)
                    if react:
                        react_threshold = np.percentile(id_train_representations, 90)
                        net.module.react_threshold = react_threshold
                else:
                    id_train_representations = None
                print(f'Score function: {inference_mode}')
                id_test_ood_scores, id_test_classes_hat, id_test_classes_true = getScores(net, underlying_net,device, test_data, criterion, inference_mode, in_data_name, num_classes, id_train_representations, architecture, in_data_name, baseline, perturb_input)
                print("\t\tTest Accuracy: ", np.mean(id_test_classes_hat == id_test_classes_true))                 
                ood_test_ood_scores, _, _ = getScores(net, underlying_net,device, open_data, criterion, inference_mode, out_data_name, num_classes, id_train_representations, architecture, in_data_name, baseline, perturb_input)

                print("\t\tCalculating OOD Metrics")
                auroc = calc_auroc(id_test_ood_scores, ood_test_ood_scores)
                tnrATtpr95 = calc_tnr(id_test_ood_scores, ood_test_ood_scores)
                print('\t\tAUROC:', auroc, 'FPR@TPR95:', 1 - tnrATtpr95)
                net.train()
        
        if epoch_loss is None:
            epoch_bar.set_description(f'Training | Epoch {epochs}/{epochs} | Batch {num_batches}/{num_batches}')
        else:
            epoch_bar.set_description(f'Training | Epoch {epochs}/{epochs} | Epoch loss = {epoch_loss:0.2f} | Batch {num_batches}/{num_batches}')
        epoch_bar.close()
        
    if test:
        net.eval()

        underlying_net = torch.nn.DataParallel(underlying_net)

        if inference_mode == 'deep_nearest_neighbors' or react or dice:
            id_train_representations = getRepresentations(underlying_net, device, train_data)
            if dice:
                net.module.calculate_mask(id_train_representations)
            elif inference_mode == 'deep_nearest_neighbors':
                id_test_representations = getRepresentations(underlying_net, device, test_data)
            if react:
                react_threshold = np.percentile(id_train_representations, 90)
                net.module.react_threshold = react_threshold
        else:
            id_train_representations = None
        
        print(f'Inference Mode: {inference_mode}')
        id_test_ood_scores, id_test_classes_hat, id_test_classes_true = getScores(net, underlying_net,device, test_data, criterion, inference_mode, in_data_name, num_classes, id_train_representations, architecture, in_data_name, baseline, perturb_input)
        ood_test_ood_scores, _, _ = getScores(net, underlying_net,device, open_data, criterion, inference_mode, out_data_name, num_classes, id_train_representations, architecture, in_data_name, baseline, perturb_input)

        print("\t\tCalculating OOD Metrics")
        auroc = calc_auroc(id_test_ood_scores, ood_test_ood_scores)
        tnrATtpr95 = calc_tnr(id_test_ood_scores, ood_test_ood_scores)
        print('\t\tAUROC:', auroc, 'FPR@TPR95:', 1 - tnrATtpr95)

        if hist:
            ood_test_abet_scores, _, _ = getScores(net, underlying_net,device, open_data, criterion, 'ablated_energy', out_data_name, num_classes, id_train_representations, architecture, in_data_name, baseline, perturb_input)
            ood_test_energy_scores, _, _ = getScores(net, underlying_net,device, open_data, criterion, 'energy', out_data_name, num_classes, id_train_representations, architecture, in_data_name, baseline, perturb_input)
            ood_test_temperature_scores, _, _ = getScores(net, underlying_net,device, open_data, criterion, 'GODIN', out_data_name, num_classes, id_train_representations, architecture, in_data_name, baseline, perturb_input)
            ood_test_temperature_scores *= -1

            id_test_abet_scores, _, _ = getScores(net, underlying_net,device, test_data, criterion, 'ablated_energy', in_data_name, num_classes, id_train_representations, architecture, in_data_name, baseline, perturb_input)
            id_test_energy_scores, _, _ = getScores(net, underlying_net,device, test_data, criterion, 'energy', in_data_name, num_classes, id_train_representations, architecture, in_data_name, baseline, perturb_input)
            id_test_temperature_scores, _, _ = getScores(net, underlying_net,device, test_data, criterion, 'GODIN', in_data_name, num_classes, id_train_representations, architecture, in_data_name, baseline, perturb_input)
            id_test_temperature_scores *= -1


            gs = gridspec.GridSpec(1,3)
            fig = plt.figure(figsize = (15, 5))
            axs = []
            for i in range(3):
                ax = fig.add_subplot(gs[i])
                # Hide the right and top spines
                ax.spines.right.set_visible(False)
                ax.spines.top.set_visible(False)
                ax.spines.left.set_visible(False)
                ax.set_yticks([])

                # Only show ticks on the left and bottom spines
                ax.xaxis.set_ticks_position('bottom')
                plt.setp(ax.get_xticklabels(), fontsize=18)

                axs.append(ax)



            axs[0].hist(ood_test_abet_scores, color = 'r', bins = 100, density = True)
            axs[0].hist(id_test_abet_scores, color = 'b', bins = 100, density = True, range = (0, 60))
            axs[0].set_title("|AbeT (Ours)|\nAUROC: {:.2f}".format(auroc), fontsize = 18)

            axs[1].hist(ood_test_energy_scores, color = 'r', bins = 100, density = True)
            axs[1].hist(id_test_energy_scores, color = 'b', bins = 100, density = True)
            axs[1].set_title("|Energy Infused w/\nLearned Temperature\nAUROC: {:.2f}".format(calc_auroc(id_test_energy_scores, ood_test_energy_scores)), fontsize = 18)
            

            axs[2].hist(id_test_temperature_scores, color = 'b', bins = 100, label = "In-Distribution", density = True)
            axs[2].hist(ood_test_temperature_scores, color = 'r', bins = 100, label= "OOD", density = True)
            axs[2].set_title("Learned Temperature\nAUROC: {:.2f}".format(calc_auroc(id_test_temperature_scores, ood_test_temperature_scores)), fontsize = 18)
            axs[2].legend(fontsize = 14)
            
            plt.savefig("pngs/{}_{}_histogram.png".format(out_data_name, in_data_name))


        if compute_tsne :
            id_test_representations = getRepresentations(underlying_net, device, test_data)
            ood_test_representations = getRepresentations(underlying_net, device, open_data)
            print("Reducing Dimensions")
            domainness = np.concatenate([np.ones((len(id_test_representations))), np.zeros((len(ood_test_representations)))], axis = 0)
            tsne_representations = TSNE().fit_transform(X = np.concatenate([id_test_representations, ood_test_representations], axis = 0))
            id_test_tsne_representations , ood_test_tsne_representations = tsne_representations[:len(id_test_representations)], tsne_representations[len(id_test_representations):]

            gs = gridspec.GridSpec(1,3)
            fig = plt.figure(figsize = (15, 5))
            axs = []
            for i in range(3):
                axs.append(fig.add_subplot(gs[i]))
                axs[-1].axis('off')

            axs[0].scatter(id_test_tsne_representations[:, 0], id_test_tsne_representations[:, 1], c = 'b', alpha = .1, label = "In-Distribution")
            axs[0].scatter(ood_test_tsne_representations[:, 0], ood_test_tsne_representations[:, 1], c = 'r', alpha = .1, label = "OOD")
            leg = axs[0].legend(fontsize = 14, loc = 'upper left')
            for lh in leg.legendHandles: 
                lh.set_alpha(1)

            correctness = id_test_classes_hat == id_test_classes_true

            axs[1].scatter(id_test_tsne_representations[np.where(correctness == 1)[0], 0], id_test_tsne_representations[np.where(correctness == 1)[0], 1], c = 'b', alpha = .1, label = "Correctly Classified")
            axs[1].scatter(id_test_tsne_representations[np.where(correctness == 0)[0], 0], id_test_tsne_representations[np.where(correctness == 0)[0], 1], c = 'r', alpha = .1, label = "Misclassified")
            leg = axs[1].legend(fontsize = 14, loc = 'upper left')
            for lh in leg.legendHandles: 
                lh.set_alpha(1)
            
            plt.set_cmap('bwr')
            colors = axs[2].scatter(id_test_tsne_representations[:, 0], id_test_tsne_representations[:, 1], c = -id_test_ood_scores, alpha = .1, vmin = np.percentile(-id_test_ood_scores, 40), vmax = np.percentile(-id_test_ood_scores, 100))
            color_bar = plt.colorbar(colors, ax = axs[2])
            color_bar.solids.set(alpha=1)
            plt.savefig("pngs/{}_{}_tsne.png".format(out_data_name, in_data_name))

def calc_tnr(id_test_ood_scores, ood_test_ood_scores):
    scores = np.concatenate((id_test_ood_scores, ood_test_ood_scores))
    trues = np.array(([1] * len(id_test_ood_scores)) + ([0] * len(ood_test_ood_scores)))
    fpr, tpr, thresholds = roc_curve(trues, scores)
    return 1 - fpr[np.argmax(tpr>=.95)]

def calc_auroc(id_test_ood_scores, ood_test_ood_scores):
    #calculate the AUROC
    scores = np.concatenate((id_test_ood_scores, ood_test_ood_scores))
    trues = np.array(([1] * len(id_test_ood_scores)) + ([0] * len(ood_test_ood_scores)))
    result = roc_auc_score(trues, scores)

    return result

def getScores(model, representation_function, CUDA_DEVICE, data_loader, criterion, inference_mode, data_name, num_classes, id_train_representations, architecture, in_data_name, baseline, perturb_input):
    model.eval()
    num_batches = len(data_loader)
    scores_total = []
    classes_hat = []
    classes_true = []
    epoch_bar = tqdm(total = num_batches, initial = 0)
    if inference_mode == "deep_nearest_neighbors":
        kd = KDTree(id_train_representations)
    for j, (images, labels) in enumerate(data_loader):

        images = Variable(images.to(CUDA_DEVICE), requires_grad = True)
        
        if data_name == "MNIST":
            images = images.repeat(1, 3, 1, 1)
        
        if baseline:
            logits, logit_numerators, temperature = model(images, T = 1)
        else:
            logits, logit_numerators, temperature = model(images)

        if inference_mode == 'energy':
            scores = torch.unsqueeze(temperature[:,0] * torch.log(torch.exp(logits).sum(dim = 1)), dim = 1)
        elif inference_mode == 'ablated_energy':
            scores = torch.unsqueeze(torch.log(torch.exp(logits).sum(dim = 1)), dim = 1)
        elif inference_mode == 'deep_nearest_neighbors':
            neighbor_distances, _ = kd.query(representation_function(images).detach().cpu().numpy(), k = 200 if in_data_name == 'CIFAR10' else 200)
            scores = np.max(neighbor_distances, axis = 1)
        elif inference_mode == "MSP":
            scores = torch.nn.functional.softmax(logits, dim = 1)
        elif inference_mode == "GODIN":
            scores = -temperature

        if perturb_input:
            if inference_mode == 'deep_nearest_neighbors':
                raise Exception("Cannot perturb input of deep nearest neighbors")
            max_scores = torch.max(scores, dim=1)[0]
            max_scores.backward(torch.ones(len(max_scores)).to(CUDA_DEVICE))
            gradient = torch.ge(images.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            # Normalizing the gradient to the same space of image
            gradient[::, 0] = (gradient[::, 0] )/(63.0/255.0)
            gradient[::, 1] = (gradient[::, 1] )/(62.1/255.0)
            gradient[::, 2] = (gradient[::, 2] )/(66.7/255.0)
            # Adding small perturbations to images
            tempInputs = torch.add(images.data, gradient, alpha=.0014 if data_name == 'CIFAR10' else .002)

            if baseline:
                logits, logit_numerators, temperature = model(tempInputs, T = 1)
            else:
                logits, logit_numerators, temperature = model(tempInputs)

            if inference_mode == 'energy':
                scores = torch.unsqueeze(temperature[:,0] * torch.log(torch.exp(logits).sum(dim = 1)), dim = 1)
            elif inference_mode == 'ablated_energy':
                scores = torch.unsqueeze(torch.log(torch.exp(logits).sum(dim = 1)), dim = 1)
            elif inference_mode == "MSP":
                scores = torch.nn.functional.softmax(logits, dim = 1)
            elif inference_mode == "GODIN":
                scores = -temperature
            

        if inference_mode == 'deep_nearest_neighbors':
            scores_total.extend(scores)
        else:
            scores_total.extend(torch.max(scores, dim=1)[0].data.cpu().numpy())
                
        classes_hat.extend(torch.max(logits, dim=1)[1].data.cpu().numpy())
        classes_true.extend(labels.cpu().numpy())
        epoch_bar.update()
    epoch_bar.close()

    return np.array(scores_total), np.array(classes_hat), np.array(classes_true)

def getRepresentations(representation_model, CUDA_DEVICE, data_loader):
    print("Getting representations")
    representation_model.eval()
    num_batches = len(data_loader)
    representations = []
    epoch_bar = tqdm(total = num_batches, initial = 0)
    for j, (images, labels) in enumerate(data_loader):
        images = images.to(CUDA_DEVICE)
        representations.extend(representation_model(images).detach().cpu().numpy())
        epoch_bar.update()
    epoch_bar.close()

    return np.array(representations)


if __name__ == '__main__':
    main()
