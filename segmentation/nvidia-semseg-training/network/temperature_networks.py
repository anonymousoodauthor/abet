import torch
from torch import nn


# util fn
def semseg_norm(x, dim=1):
    norm = torch.norm(x, p=2, dim=dim)
    x = x / (torch.unsqueeze(norm, dim=dim) + 0.0001)
    return x


"""
SemSeg Heads to replace the final layer
"""


class AbstractSemSegHead(nn.Module):
    """
    Base class for the output of a segmentation network, consisting of a
    IN_FEATURES x NUM_CLASSES (1,1) kernel
    """

    def __init__(self, in_features, num_classes, bias):
        super(AbstractSemSegHead, self).__init__()

        self.h = torch.nn.Conv2d(
            in_channels=in_features,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            bias=bias,
        )
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity="relu")


class CosineSemSegHead(AbstractSemSegHead):
    # TODO: can you do this without permute
    """
    Cosine similarity output head for a segmentation network. Instead of using the "inner product" of the penultimate representation with the final layer,
    use cosine similarities as logits.

    """

    def forward(self, x):  # BPHW, where P is Penultimate Features
        x = semseg_norm(x, 1)  # BPHW
        c_last_x = x.permute(0, 2, 3, 1)  # BHWP --> need channels last for matmul to broadcast
        w = torch.squeeze(semseg_norm(self.h.weight, 1))  # CP11 --> CP
        ret = torch.matmul(c_last_x, w.T)  # BHWP x PC --> BHWC
        ret = ret.permute(0, 3, 1, 2)  # BCHW
        return ret

    # def forward(self, x):
    #     return torch_cosine_similarity(x, self.h.weight)


class InnerSemSegHead(AbstractSemSegHead):
    def init_weights(self):
        super().init_weights()
        self.h.bias.data = torch.zeros(size=self.h.bias.size())

    def forward(self, x):
        return self.h(x)  # BPHW --> BCHW


"""
SemSeg Temperature Networks to wrap an underlying model with a SemSegHead and Temperature
"""


class AbstractSemSegNet(nn.Module):
    def __init__(self, underlying_model, num_classes, in_features=256):
        super(AbstractSemSegNet, self).__init__()
        self.num_classes = num_classes
        self.underlying_model = underlying_model
        self.in_features = in_features


class LearnedTemperatureSemSegNet(AbstractSemSegNet):
    """
    Wraps an underlying model with a Cosine semseg head and a learned temperature layer
    """

    def __init__(self, underlying_model, num_classes, in_features=256):
        super().__init__(underlying_model, num_classes, in_features=in_features)

        self.h = CosineSemSegHead(self.in_features, num_classes, False)
        self.temperature = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_features,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        penultimate_representations = self.underlying_model(x)  # BPHW
        temperature = self.temperature(penultimate_representations) + 1e-12  # B1HW

        logit_numerators = self.h(penultimate_representations)  # BCHW
        logits = logit_numerators / temperature  # BCHW

        return logits, logit_numerators, temperature


class BaselineTemperatureSemSegNet(AbstractSemSegNet):
    def __init__(self, underlying_model, num_classes, in_features=128):
        super().__init__(underlying_model, num_classes, in_features=in_features)

        self.h = InnerSemSegHead(self.in_features, num_classes, True)

    def forward(self, x, gts=None, T=1):
        penultimate_representations = self.underlying_model(x, gts=None)  # BPHW

        # this shape doesn't really matter but keeping it consistent with
        # learned temperature shape instead of just 1
        ones_shape = [
            penultimate_representations.shape[0],
            1,
            *penultimate_representations.shape[2:],
        ]
        temperature = (T * torch.ones(ones_shape)).cuda()  # B1HW

        logit_numerators = self.h(penultimate_representations)  # BCHW
        logits = logit_numerators / temperature  # BCHW
        # logits, logit_numerators, and temperature
        return (
            logits,
            logit_numerators,
        )
