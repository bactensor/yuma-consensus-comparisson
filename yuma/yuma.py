
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

def Yuma4(
    W: torch.Tensor,
    S: torch.Tensor,
    B_old: Optional[torch.Tensor] = None,
    config: YumaConfig = YumaConfig(),
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

    # === Liquid Alpha Adjustment ===
    a = b = None
    bond_alpha = config.bond_alpha
    if config.liquid_alpha:
        consensus_high = config.override_consensus_high if config.override_consensus_high is not None else C.quantile(0.75)
        consensus_low = config.override_consensus_low if config.override_consensus_low is not None else C.quantile(0.25)

        if consensus_high == consensus_low:
            consensus_high = C.quantile(0.99)

        a = (math.log(1 / config.alpha_high - 1) - math.log(1 / config.alpha_low - 1)) / (consensus_low - consensus_high)
        b = math.log(1 / config.alpha_low - 1) + a * consensus_low
        alpha = 1 / (1 + math.e ** (-a * C + b))  # alpha to the old weight
        bond_alpha = 1 - torch.clamp(alpha, config.alpha_low, config.alpha_high)

    # === Bonds ===
    if B_old is None:
        B_old = torch.zeros_like(W)

    B_decayed = B_old *  (1 - bond_alpha)
    remaining_capacity = 1.0 - B_decayed
    remaining_capacity = torch.clamp(remaining_capacity, min=0.0)

    # Each validator can increase bonds by at most bond_alpha per epoch towards the cap
    purchase_increment = bond_alpha * W  # Validators allocate their purchase across miners based on weights
    # Ensure that purchase does not exceed remaining capacity
    purchase = torch.min(purchase_increment, remaining_capacity)

    B = B_decayed + purchase
    B = torch.clamp(B, max=1.0)

    # === Dividends Calculation ===
    total_bonds_per_validator = (B * I).sum(dim=1)  # Sum over miners for each validator
    D = S * total_bonds_per_validator  # Element-wise multiplication

    # Normalize dividends
    D_normalized = D / (D.sum() + 1e-6)


    return {
        "weight": W,
        "stake": S,
        "server_prerank": P,
        "server_consensus_weight": C,
        "consensus_clipped_weight": W_clipped,
        "server_rank": R,
        "server_incentive": I,
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
            reset_bonds_epoch=case.reset_bonds_epoch,
            reset_bonds_miner_index=case.reset_bonds_index,
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