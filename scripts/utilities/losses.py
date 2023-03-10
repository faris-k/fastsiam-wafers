import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from lightly.loss.msn_loss import prototype_probabilities, sharpen, sinkhorn


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
        tau: float = 0.25,
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
        """Computes the PMSN loss for a set of anchors, targets and prototypes.

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

        # PMSN loss replaces mean entropy maximization regularization with
        # KL divergence to a power law distribution parameterized by tau
        if self.pmsn_weight > 0:
            mean_anchor_probs = torch.mean(anchor_probs, dim=0)

            # Create a tensor of power-law distributed probabilities of size n
            n = len(mean_anchor_probs)
            power_law = np.random.power(self.tau, size=n)
            power_law /= power_law.sum()
            power_law = torch.from_numpy(power_law).to(mean_anchor_probs.device)

            kl_div = F.kl_div(
                power_law.log(),
                mean_anchor_probs.log(),
                reduction="none",
                log_target=True,
            ).sum()

            # Calculate KL divergence manually: D_KL(P || Q)
            # P, Q = mean_anchor_probs, power_law
            # kl_div_true = (P * (P / Q).log()).sum()
            # Same as doing F.kl_div(Q.log(), P, reduction="batchmean")

            # print(f"loss: {loss}")
            # print(f"kl_div: {kl_div}")
            # print(f"kl_div_true: {kl_div_true}")
            loss += self.pmsn_weight * kl_div

        return loss
