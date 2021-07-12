# Code is modified from MEAL (https://arxiv.org/abs/1812.02425) and Label Refinery (https://arxiv.org/abs/1805.02641).

import torch
from torch.nn import functional as F
from torch.nn.modules import loss


class DistributionLoss(loss._Loss):
    """The KL-Divergence loss for the binary student model and real teacher output.
    output must be a pair of (model_output, real_output), both NxC tensors.
    The rows of real_output must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, model_output, real_output):

        size_average = True

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        real_output_soft = F.softmax(real_output, dim=1)
        del model_output, real_output

        # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
        # for batch matrix multiplicatio
        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
        if size_average:
             cross_entropy_loss = cross_entropy_loss.mean()
        else:
             cross_entropy_loss = cross_entropy_loss.sum()

        return cross_entropy_loss