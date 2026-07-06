import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_VALUE = -99999.0


class MaxpoolReadoutLayer(nn.Module):
    """Max-pool readout over the (padded) child dimension.

    Padded child positions beyond each node's real child count are masked
    before pooling. The pooling indices are stored so the deconvolution pass
    can unpool (paper, Algorithm 4-5).
    """

    def forward(self, hidden, child_counts):
        node_count = hidden.shape[1]

        counts = torch.as_tensor(child_counts, device=hidden.device)
        counts = counts.clamp(min=1)  # count 0: keep the zero-vector padding row
        mask = torch.arange(node_count, device=hidden.device).unsqueeze(0) < counts.unsqueeze(1)
        masked = torch.where(mask.unsqueeze(-1), hidden, torch.full_like(hidden, MASK_VALUE))

        masked = masked.unsqueeze(1)
        pooled, indices = F.max_pool2d(masked, kernel_size=(node_count, 1), return_indices=True)
        return pooled.squeeze(1), indices
