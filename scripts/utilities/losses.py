import math

import torch

# import torch.distributed as dist
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from lightly.loss.msn_loss import prototype_probabilities, sharpen, sinkhorn


class PowerLaw(D.Distribution):
    def __init__(self, mean_anchor_probs, tau):
        self.mean_anchor_probs = mean_anchor_probs
        self.tau = tau

    def log_prob(self, value):
        return (1 - self.tau) * torch.log(value)

    def entropy(self):
        return None  # Not implemented


# def kl_divergence_power_law(mean_anchor_probs, tau):
#     # Ensure that the mean_anchor_probs tensor has values in the range [0, 1]
#     mean_anchor_probs = torch.clamp(mean_anchor_probs, 1e-7, 1 - 1e-7)

#     # Compute the log of the power-law distribution
#     log_power_law = (1 - tau) * torch.log(mean_anchor_probs)

#     # Compute the KL-divergence between the two distributions
#     kl_div = (mean_anchor_probs * (torch.log(mean_anchor_probs) - log_power_law)).sum(
#         dim=-1
#     )

#     return kl_div


def kl_divergence_power_law(mean_anchor_probs, tau):
    mean_anchor_probs = torch.clamp(mean_anchor_probs, 1e-7, 1 - 1e-7)
    power_law = PowerLaw(mean_anchor_probs, tau)
    return D.kl_divergence(mean_anchor_probs, power_law)


class PMSNLoss(nn.Module):
    """Implementation of the loss function from PMSN

    Attributes:
        temperature:
            Similarities between anchors and targets are scaled by the inverse of
            the temperature. Must be in (0, 1].
        sinkhorn_iterations:
            Number of sinkhorn normalization iterations on the targets.
        me_max_weight:
            Weight factor lambda by which the mean entropy maximization regularization
            loss is scaled. Set to 0 to disable the reguliarization.

     Examples:

        >>> # initialize loss function
        >>> loss_fn = PMSNLoss()
        >>>
        >>> # generate anchors and targets of images
        >>> anchors = transforms(images)
        >>> targets = transforms(images)
        >>>
        >>> # feed through MSN model
        >>> anchors_out = model(anchors)
        >>> targets_out = model.target(targets)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(anchors_out, targets_out, prototypes=model.prototypes)

    """

    def __init__(
        self,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        pmsn_weight: float = 1.0,
        tau: float = 0.75,
        gather_distributed: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.pmsn_weight = pmsn_weight
        self.tau = tau
        self.gather_distributed = gather_distributed

    def forward(
        self,
        anchors: torch.Tensor,
        targets: torch.Tensor,
        prototypes: torch.Tensor,
        target_sharpen_temperature: float = 0.25,
    ) -> torch.Tensor:
        """Computes the MSN loss for a set of anchors, targets and prototypes.

        Args:
            anchors:
                Tensor with shape (batch_size * anchor_views, dim).
            targets:
                Tensor with shape (batch_size, dim).
            prototypes:
                Tensor with shape (num_prototypes, dim).
            target_sharpen_temperature:
                Temperature used to sharpen the target probabilities.

        Returns:
            Mean loss over all anchors.

        """
        num_views = anchors.shape[0] // targets.shape[0]
        anchors = F.normalize(anchors, dim=1)
        targets = F.normalize(targets, dim=1)
        prototypes = F.normalize(prototypes, dim=1)

        # anchor predictions
        anchor_probs = prototype_probabilities(
            anchors, prototypes, temperature=self.temperature
        )

        # target predictions
        with torch.no_grad():
            target_probs = prototype_probabilities(
                targets, prototypes, temperature=self.temperature
            )
            target_probs = sharpen(target_probs, temperature=target_sharpen_temperature)
            if self.sinkhorn_iterations > 0:
                target_probs = sinkhorn(
                    probabilities=target_probs,
                    iterations=self.sinkhorn_iterations,
                    gather_distributed=self.gather_distributed,
                )
            target_probs = target_probs.repeat((num_views, 1))

        # cross entropy loss
        loss = torch.mean(torch.sum(torch.log(anchor_probs ** (-target_probs)), dim=1))

        # # PMSN loss replaces mean entropy maximization regularization with
        # # KL divergence to a power law distribution parameterized by tau
        # if self.pmsn_weight > 0:
        #     mean_anchor_probs = torch.mean(anchor_probs, dim=0)
        #     target_dist = (torch.arange(len(mean_anchor_probs)) + 1) ** (-self.tau)
        #     target_dist = target_dist / torch.sum(target_dist)
        #     target_dist = target_dist.to(mean_anchor_probs.device)
        #     pmsn_loss = F.kl_div(
        #         torch.log(mean_anchor_probs),
        #         torch.log(target_dist),
        #         # reduction="batchmean",
        #     )
        #     # loss -= self.pmsn_weight * pmsn_loss

        # if self.pmsn_weight > 0:
        #     mean_anchor_probs = torch.mean(anchor_probs, dim=0)
        #     mean_anchor_probs = torch.clamp(mean_anchor_probs, 1e-7, 1 - 1e-7)

        #     power_law_probs = mean_anchor_probs ** (-self.tau)
        #     power_law_probs = power_law_probs / power_law_probs.sum()
        #     power_law_probs = torch.clamp(power_law_probs, 1e-7, 1 - 1e-7)

        #     pmsn_loss = F.kl_div(
        #         torch.log(mean_anchor_probs),
        #         torch.log(power_law_probs),
        #         reduction="batchmean",
        #     )
        #     loss += self.pmsn_weight * pmsn_loss

        if self.pmsn_weight > 0:
            mean_anchor_probs = torch.mean(anchor_probs, dim=0)
            kl_div = kl_divergence_power_law(mean_anchor_probs, self.tau)
            print("loss", loss)
            print("kl_div", kl_div)
            loss += self.pmsn_weight * kl_div

        return loss
