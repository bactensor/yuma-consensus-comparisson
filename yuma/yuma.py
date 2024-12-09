
import math
import torch
import json
from typing import Dict, Optional, Union
from dataclasses import dataclass
from yuma.cases import cases
from yuma.utils.utils import run_simulation, calculate_total_dividends

@dataclass
class YumaConfig:
    kappa: float = 0.5
    bond_penalty: float = 0.99
    bond_alpha: float = 0.1
    decay_rate: float = 0.1
    capacity_alpha: float = 0.1
    liquid_alpha: bool = False
    alpha_high: float = 0.9
    alpha_low: float = 0.7
    precision: int = 100_000
    override_consensus_high: Optional[float] = None
    override_consensus_low: Optional[float] = None
    total_epoch_emission = 100
    validator_emission_ratio = 0.41
    total_subnet_stake = 1_000_000

def Yuma3(
    W: torch.Tensor,
    S: torch.Tensor,
    B_old: Optional[torch.Tensor] = None,
    config: YumaConfig = YumaConfig(),
    maxint: int = 2 ** 64 - 1,
) -> Dict[str, Union[torch.Tensor, float]]:
    """
    Original Yuma function with bonds and EMA calculation.
    """

    # === Weight ===
    W = (W.T / (W.sum(dim=1) + 1e-6)).T

    # === Stake ===
    S = S / S.sum()

    # === Prerank ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus ===
    C = torch.zeros(W.shape[1])

    for i, miner_weight in enumerate(W.T):
        c_high = 1.0
        c_low = 0.0

        while (c_high - c_low) > 1 / config.precision:
            c_mid = (c_high + c_low) / 2.0
            _c_sum = (miner_weight > c_mid) * S
            if _c_sum.sum() > config.kappa:
                c_low = c_mid
            else:
                c_high = c_mid

        C[i] = c_high

    C = (C / C.sum() * 65_535).int() / 65_535

    # === Consensus clipped weight ===
    W_clipped = torch.min(W, C)

    # === Rank ===
    R = (S.view(-1, 1) * W_clipped).sum(dim=0)

    # === Incentive ===
    I = (R / R.sum()).nan_to_num(0)

    # === Trusts ===
    T = (R / P).nan_to_num(0)
    T_v = W_clipped.sum(dim=1) / W.sum(dim=1)

    # === Bonds ===
    if B_old is None:
        B_old = torch.zeros_like(W)

    capacity = S * maxint

    # Compute Remaining Capacity
    capacity_per_bond = S.unsqueeze(1) * maxint
    remaining_capacity = capacity_per_bond - B_old
    remaining_capacity = torch.clamp(remaining_capacity, min=0.0)

    # Compute Purchase Capacity
    capacity_alpha = (config.capacity_alpha * capacity).unsqueeze(1)
    purchase_capacity = torch.min(capacity_alpha, remaining_capacity)

    # Allocate Purchase to Miners
    purchase = purchase_capacity * W

    # Update Bonds with Decay and Purchase
    decay = 1 - config.decay_rate
    B = decay * B_old + purchase
    B = torch.min(B, capacity_per_bond)  # Enforce capacity constraints

    # === Validator reward ===
    D = (B * I).sum(dim=1)
    D_normalized = D / (D.sum() + 1e-6)

    return {
        "weight": W,
        "stake": S,
        "server_prerank": P,
        "server_consensus_weight": C,
        "consensus_clipped_weight": W_clipped,
        "server_rank": R,
        "server_incentive": I,
        "server_trust": T,
        "validator_trust": T_v,
        "validator_bonds": B,
        "validator_reward": D,
        "validator_reward_normalized": D_normalized
    }

if __name__ == "__main__":
    config = YumaConfig()
    total_dividends_per_case = {}
    for case in cases:
        dividends_per_validator = run_simulation(
            validators=case.validators,
            stakes=case.stakes_epochs,
            weights=case.weights_epochs,
            num_epochs=case.num_epochs,
            config=config,
        )
        total_dividends, _ = calculate_total_dividends(
            validators=case.validators,
            dividends_per_validator=dividends_per_validator,
            base_validator=case.base_validator,
            num_epochs=case.num_epochs,
        )
        total_dividends_per_case[case.name] = total_dividends
    
    print(json.dumps(total_dividends_per_case, indent=4))