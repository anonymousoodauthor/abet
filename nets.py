import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x

class AbstractClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes, bias):
        super(AbstractClassificationHead, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias= bias)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")

class CosineClassificationHead(AbstractClassificationHead):
    def __init__(self, in_features, num_classes):
        super(CosineClassificationHead, self).__init__(in_features, num_classes, False)

    def forward(self, x):
        x = norm(x)
        w = norm(self.h.weight)

        ret = (torch.matmul(x.double(),w.T.double()))
        return ret

class InnerClassificationHead(AbstractClassificationHead):
    def __init__(self, in_features, num_classes):
        super(InnerClassificationHead, self).__init__(in_features, num_classes, True)
    
    def init_weights(self):
        super().init_weights()
        self.h.bias.data = torch.zeros(size = self.h.bias.size())

    def forward(self, x):
        return self.h(x)

class AbstractClassificationNet(nn.Module):
    def __init__(self, underlying_model, num_classes):
        super(AbstractClassificationNet, self).__init__()
        
        self.num_classes = num_classes

        self.underlying_model = underlying_model
        
        self.in_features = underlying_model.output_size
        
    def calculate_mask(self, id_train_representations):
        w = self.h.h.weight 
        mean_act = id_train_representations.mean(0)
        contrib = np.abs(mean_act[None, :] * w.data.squeeze().cpu().numpy())
        self.thresh = np.percentile(contrib, 90)
        mask = torch.Tensor((contrib > self.thresh)).cuda()
        self.h.h.weight.data = w * mask

class LearnedTemperatureClassificationNet(AbstractClassificationNet):
    def __init__(self, underlying_model, num_classes):
        super().__init__(underlying_model, num_classes)

        self.h = CosineClassificationHead(self.in_features, num_classes)

        self.temperature = nn.Sequential(
                nn.Linear(self.in_features, 1),
                nn.BatchNorm1d(1),
                nn.Sigmoid()
            )

        self.react_threshold = 0

    def forward(self, x):
        penultimate_representations = self.underlying_model(x)
        temperature = self.temperature(penultimate_representations.float()) + 1e-12
        if self.react_threshold > 0:
            penultimate_representations = penultimate_representations.clip(max = self.react_threshold)
        # Compute temperature
        logit_numerators = self.h(penultimate_representations)

        logits = logit_numerators / temperature
        # logits, logit_numerators, and temperature
        return logits, logit_numerators, temperature


class BaselineClassificationNet(AbstractClassificationNet):
    def __init__(self, underlying_model, num_classes):
        super().__init__(underlying_model, num_classes)

        self.h = InnerClassificationHead(self.in_features, num_classes, True)
    
    def forward(self, x, T):
        penultimate_representations = self.underlying_model(x)
        # Temperature is the default used in GradNorm, ODIN, GODIN
        temperature = T * torch.unsqueeze(torch.ones(len(penultimate_representations.float())).cuda(), dim = 1)
        logit_numerators = self.h(penultimate_representations)

        logits = logit_numerators / temperature
        # logits, logit_numerators, and temperature
        return logits, logit_numerators, temperature