import torch
from torch import nn


class MarketWindowCoEUMoE(nn.Module):
    """
    input: x_market [B, T, M]
    output: [B, F]
    """
    def __init__(self,
                 market_dim: int,
                 factor_dim: int,
                 market_hidden_dim: int,
                 num_experts: int,
                 market_chain_depth: int,
                 market_num_heads: int,
                 dropout: float = 0.5,
                 beta: float = 5):
        super().__init__()
        self.market_dim = market_dim
        self.factor_dim = factor_dim
        self.num_experts = num_experts
        self.market_chain_depth = market_chain_depth
        self.beta = beta
        # market experts: M -> M
        self.experts_M = nn.ModuleList([
            nn.Sequential(
                nn.Linear(market_dim, market_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(market_hidden_dim, market_dim)
            )
            for _ in range(num_experts)
        ])
        # M -> F
        self.experts_F = nn.ModuleList([
            nn.Sequential(
                nn.Linear(market_dim, market_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(market_hidden_dim, factor_dim)
            )
            for _ in range(num_experts)
        ])
        self.market_gate = nn.Linear(market_dim, num_experts, bias=False)
        self.attn_layer = nn.MultiheadAttention(embed_dim=market_dim,
                                                num_heads=market_num_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.fusion = nn.Linear(market_dim * 2, market_dim)  # 融合层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_market: torch.Tensor) -> torch.Tensor:
        # x_market: [B, T, M]
        B, T, M = x_market.shape
        h = x_market.mean(dim=1)  # [B, M]
        for i in range(self.market_chain_depth):
            logits = self.market_gate(h)  # [B, E]
            scores = torch.softmax(logits / self.beta, dim=-1)  # [B, E]
            expert_outputs = torch.stack([exp(h) for exp in self.experts_M], dim=-1)  # [B, M, E]
            scores = scores.unsqueeze(1)  # [B,1,E]
            h = (expert_outputs * scores).sum(dim=-1)  # [B, M]
        # MultiheadAttention expects (B, T, M) with batch_first=True
        attn_output, _ = self.attn_layer(x_market, x_market, x_market)  # [B, T, M]
        z = attn_output.mean(dim=1)  # [B, M]

        fusion_input = torch.cat([h, z], dim=-1)  # [B, 2M]
        z = self.fusion(fusion_input)         # [B, M]
        logits2 = self.market_gate(z)  # [B, E]
        scores2 = torch.softmax(logits2 / self.beta, dim=-1)  # [B, E]
        expert_outputs2 = torch.stack([exp(z) for exp in self.experts_F], dim=-1)  # [B, F, E]
        scores2 = scores2.unsqueeze(1)  # [B, 1, E]
        out = (expert_outputs2 * scores2).sum(dim=-1)  # [B, F]

        return self.dropout(out)



class MetricWindowCoEUMoE(nn.Module):
    """
    input: metric_input [B, F, metric_dim]
    output: [B, F]
    """
    def __init__(self,
                 metric_dim: int,
                 factor_dim: int,
                 metric_hidden_dim: int,
                 num_experts: int,
                 metric_chain_depth: int,
                 metric_num_heads: int,
                 dropout: float,
                 beta: float):
        super().__init__()
        self.metric_dim = metric_dim
        self.factor_dim = factor_dim
        self.num_experts = num_experts
        self.metric_chain_depth = metric_chain_depth
        self.beta = beta
        self.experts_M = nn.ModuleList([
            nn.Sequential(
                nn.Linear(metric_dim, metric_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(metric_hidden_dim, metric_dim)
            )
            for _ in range(num_experts)
        ])
        self.experts_F = nn.ModuleList([
            nn.Sequential(
                nn.Linear(metric_dim, metric_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(metric_hidden_dim, 1)
            )
            for _ in range(num_experts)
        ])

        self.metric_gate = nn.Linear(metric_dim, num_experts, bias=False)
        self.attn_layer = nn.MultiheadAttention(embed_dim=metric_dim,
                                                num_heads=metric_num_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.fusion = nn.Linear(metric_dim * 2, metric_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, metric_input: torch.Tensor) -> torch.Tensor:
        # metric_input: [B, F, metric_dim]
        B, F, D = metric_input.shape
        h = metric_input  # [B, F, metric_dim]
        # —— CoE  —— #
        for i in range(self.metric_chain_depth):
            gate = self.metric_gate
            experts = self.experts_M
            input_h = h  # [B, F, metric_dim]
            logits = gate(input_h)  # [B, F, E]
            scores = torch.softmax(logits / self.beta, dim=-1)  # [B, F, E]
            expert_outputs = torch.stack(
                [exp(input_h) for exp in experts], dim=-1
            )  # [B, F, metric_dim, E]
            scores = scores.unsqueeze(-2)  # [B, F, 1, E]
            h = (expert_outputs * scores).sum(dim=-1)  # [B, F, metric_dim]
        attn_output, _ = self.attn_layer(metric_input, metric_input, metric_input)  # [B, F, metric_dim]
        z = attn_output  # [B, F, metric_dim]
        fusion_input = torch.cat([h, z], dim=-1)  # [B, F, 2*metric_dim]
        z = self.fusion(fusion_input)         # [B, F, metric_dim]
        logits2 = self.metric_gate(z)  # [B, F, E]
        scores2 = torch.softmax(logits2 / self.beta, dim=-1)  # [B, F, E]
        expert_outputs2 = torch.stack(
            [exp(z) for exp in self.experts_F], dim=-1
        )  # [B, F, 1, E]
        scores2 = scores2.unsqueeze(-2)  # [B, F, 1, E]
        out = (expert_outputs2 * scores2).sum(dim=-1).squeeze(-1)  # [B, F]

        return self.dropout(out)

class FactorDynamicGate(nn.Module):
    def __init__(self, factor_dim: int, k: int, temperature:float):
        super().__init__()
        self.factor_dim = factor_dim
        self.k = k
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.temperature = temperature

    def forward(self, x_factors: torch.Tensor, market_mixed: torch.Tensor, metric_mixed: torch.Tensor) -> torch.Tensor:
        # x_factors: [B, T, N, F]
        # market_mixed: [B, F]
        # metric_mixed: [B, F]
        B, T, N, F = x_factors.shape
        assert F == self.factor_dim

        alpha = torch.sigmoid(self.alpha)
        scores = alpha * market_mixed + (1 - alpha) * metric_mixed  # [B, F]
        scores = scores / self.temperature
        gate_scores = torch.softmax(scores, dim=-1)  # [B, F]
        #Top-k
        gumbel_noise = -torch.empty_like(gate_scores).exponential_().log()
        noisy_scores = (gate_scores + 1e-8).log() + gumbel_noise
        topk_vals, topk_idx = torch.topk(noisy_scores, self.k, dim=-1)
        mask = torch.zeros_like(gate_scores)
        mask.scatter_(1, topk_idx, 1.0)
        gate_scores = gate_scores * mask
        gate_scores = gate_scores / (gate_scores.sum(dim=-1, keepdim=True) + 1e-6)
        gate = gate_scores.view(B, 1, 1, F)  # [B,1,1,F]
        mask = mask.view(B, 1, 1, F)         # [B,1,1,F]
        out = x_factors * gate + x_factors * mask
        return out

class FactorMoE(nn.Module):
    """
    output: [B, N]。
    """
    def __init__(self,
                 gate_input_start_index: int,
                 gate_input_end_index: int,
                 market_dim: int,
                 metric_dim: int,
                 factor_dim: int,
                 market_hidden_dim: int,
                 market_num_experts: int,
                 market_chain_depth: int,
                 market_num_heads: int,
                 market_dropout: float,
                 market_bete: float,
                 metric_hidden_dim:int,
                 metric_chain_depth: int,
                 metric_num_heads: int,
                 metric_num_experts:int,
                 metric_dropout: float,
                 metric_bete: float,
                 k: int,
                 temperature:float,

                 ):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)
        self.moe1 = MarketWindowCoEUMoE(market_dim, factor_dim, market_hidden_dim=market_hidden_dim, num_experts=market_num_experts, dropout=market_dropout, beta=market_bete, market_chain_depth=market_chain_depth, market_num_heads=market_num_heads)
        self.moe2 = MetricWindowCoEUMoE(metric_dim, factor_dim, metric_hidden_dim=metric_hidden_dim, num_experts=metric_num_experts, dropout=metric_dropout, beta=metric_bete, metric_chain_depth=metric_chain_depth, metric_num_heads=metric_num_heads)
        self.gate = FactorDynamicGate(factor_dim, k, temperature=temperature)
        self.readout = nn.Linear(factor_dim, 1)


    def forward(self, x_input: torch.Tensor, metric_input: torch.Tensor) -> torch.Tensor:
        # x_input: [B, T, N, F + M], metric_input:[B, F, 9],market_input: [B, T, M]
        fct = x_input[:, :, :, :self.gate_input_start_index]
        market_input = x_input[:, :, -1, self.gate_input_start_index:self.gate_input_end_index]
        market_mixed = self.moe1(market_input)                  # [B, F]
        metric_mixed = self.moe2(metric_input)
        x_dyn = self.gate(fct, market_mixed, metric_mixed) # [B, T, N, F]
        x_last = x_dyn[:, -1, :, :]                 # [B, N, F]
        out = self.readout(x_last).squeeze(-1)     # [B, N]
        return out

