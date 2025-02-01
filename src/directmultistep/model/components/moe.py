import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
activation_dict = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
}


class PositionwiseFeedforwardLayer(nn.Module):
    """Positionwise feedforward layer.

    Applies a two-layer feedforward network to the input.

    Shape suffixes:
        B: batch size
        L: sequence length
        D: model dimension
        F: feed-forward subnetwork hidden size
    """

    def __init__(
        self,
        hid_dim: int,
        ff_mult: int,
        ff_activation: nn.Module,
        dropout: float,
    ):
        """Initializes the PositionwiseFeedforwardLayer.

        Args:
            hid_dim: The hidden dimension size (D).
            ff_mult: The feed-forward expansion factor.
            ff_activation: The activation function.
            dropout: The dropout rate.
        """
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, ff_mult * hid_dim)
        self.activ = ff_activation
        self.fc_2 = nn.Linear(hid_dim * ff_mult, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_BLD: Tensor) -> Tensor:
        """Forward pass of the PositionwiseFeedforwardLayer.

        Args:
            x_BLD: The input tensor of shape (B, L, D).

        Returns:
            The output tensor of shape (B, L, D).
        """
        x_BLF = self.dropout(self.activ(self.fc_1(x_BLD)))
        x_BLD = self.fc_2(x_BLF)
        return x_BLD


class NoisyTopkRouter(nn.Module):
    """Noisy top-k router for MoE.

    Routes inputs to the top-k experts based on noisy logits.

    Shape suffixes:
        B: batch size
        L: sequence length
        D: model dimension
        E: number of experts
        K: top_k
    """

    def __init__(self, hid_dim: int, n_experts: int, top_k: int):
        """Initializes the NoisyTopkRouter.

        Args:
            hid_dim: The hidden dimension size (D).
            n_experts: The number of experts (E).
            top_k: The number of top experts to route to (K).
        """
        super().__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(hid_dim, n_experts)
        self.noise_linear = nn.Linear(hid_dim, n_experts)

    def forward(self, x_BLD: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the NoisyTopkRouter.

        Args:
            x_BLD: The input tensor of shape (B, L, D).

        Returns:
            A tuple containing:
                - The router output tensor of shape (B, L, E).
                - The indices of the top-k experts of shape (B, L, K).
        """
        logits_BLE = self.topkroute_linear(x_BLD)
        noise_logits_BLE = self.noise_linear(x_BLD)
        # Adding scaled unit gaussian noise to the logits
        noise_BLE = torch.randn_like(logits_BLE) * F.softplus(noise_logits_BLE)
        noisy_logits_BLE = logits_BLE + noise_BLE

        top_k_logits_BLE, indices_BLK = noisy_logits_BLE.topk(self.top_k, dim=-1)
        zeros_BLE = torch.full_like(noisy_logits_BLE, float("-inf"))
        # creating a sparse tensor with top-k logits
        sparse_logits_BLE = zeros_BLE.scatter(-1, indices_BLK, top_k_logits_BLE)
        router_output_BLE = F.softmax(sparse_logits_BLE, dim=-1)
        return router_output_BLE, indices_BLK


class Expert(nn.Module):
    """A single expert in the MoE layer.

    Applies a two-layer feedforward network to the input.

    Shape suffixes:
        B: batch size
        L: sequence length
        D: model dimension
        F: feed-forward subnetwork hidden size
    """

    def __init__(
        self,
        hid_dim: int,
        ff_mult: int,
        ff_activation: str,
        dropout: float,
    ):
        """Initializes the Expert.

        Args:
            hid_dim: The hidden dimension size (D).
            ff_mult: The feed-forward expansion factor.
            ff_activation: The activation function type.
            dropout: The dropout rate.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hid_dim, ff_mult * hid_dim),
            activation_dict[ff_activation],
            nn.Linear(ff_mult * hid_dim, hid_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x_BLD: Tensor) -> Tensor:
        """Forward pass of the Expert.

        Args:
            x_BLD: The input tensor of shape (B, L, D).

        Returns:
            The output tensor of shape (B, L, D).
        """
        return self.net(x_BLD)  # type: ignore


class SparseMoE(nn.Module):
    """Sparse Mixture of Experts layer.

    Routes inputs to a subset of experts and combines their outputs.

    Shape suffixes:
        B: batch size
        L: sequence length
        D: model dimension
        E: number of experts
        K: top_k
        S: number of selected tokens for an expert
    """

    def __init__(
        self,
        hid_dim: int,
        n_experts: int,
        top_k: int,
        ff_mult: int,
        ff_activation: str,
        dropout: float,
        capacity_factor: float,
    ):
        """Initializes the SparseMoE layer.

        Args:
            hid_dim: The hidden dimension size (D).
            n_experts: The number of experts (E).
            top_k: The number of top experts to route to (K).
            ff_mult: The feed-forward expansion factor.
            ff_activation: The activation function type.
            dropout: The dropout rate.
            capacity_factor: The capacity factor for each expert.
        """
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(hid_dim, n_experts, top_k)
        self.experts = nn.ModuleList([Expert(hid_dim, ff_mult, ff_activation, dropout) for _ in range(n_experts)])
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

    def forward(self, x_BLD: Tensor) -> Tensor:
        """Forward pass of the SparseMoE layer.

        Args:
            x_BLD: The input tensor of shape (B, L, D).

        Returns:
            The output tensor of shape (B, L, D).
        """
        B, L, _ = x_BLD.shape
        gating_output_BLE, indices_BLK = self.router(x_BLD)
        final_output_BLD = torch.zeros_like(x_BLD)

        flat_x_FD = x_BLD.view(-1, x_BLD.size(-1))  # [B*L, D], define B*L=F
        flat_gating_output_FE = gating_output_BLE.view(-1, gating_output_BLE.size(-1))
        n_tkns = B * L * self.top_k
        capacity = int((n_tkns / self.n_experts) * self.capacity_factor)

        updates_FD = torch.zeros_like(flat_x_FD)
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask_BL = (indices_BLK == i).any(dim=-1)
            flat_mask_F = expert_mask_BL.view(-1)
            selected_idxs_F = torch.nonzero(flat_mask_F).squeeze(-1)

            if selected_idxs_F.numel() > capacity:
                limited_idxs_F = selected_idxs_F[:capacity]
            else:
                limited_idxs_F = selected_idxs_F

            if limited_idxs_F.numel() > 0:
                expert_input_SD = flat_x_FD[limited_idxs_F]  # S = sum(flat_mask_F)
                expert_output_SD = expert(expert_input_SD)

                # Extract and apply gating scores, [S] -> [S, 1]
                gating_scores_S1 = flat_gating_output_FE[limited_idxs_F, i].unsqueeze(1)
                weighted_output_SD = expert_output_SD * gating_scores_S1

                updates_FD.index_add_(0, limited_idxs_F, weighted_output_SD)

        final_output_BLD += updates_FD.view(B, L, -1)

        return final_output_BLD
