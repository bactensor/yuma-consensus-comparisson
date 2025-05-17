
import torch
from dataclasses import dataclass
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class SimulationHyperparameters:
    kappa: float = 0.5
    bond_penalty: float = None
    total_epoch_emission: float = 100.0
    validator_emission_ratio: float = 0.41
    total_subnet_stake: float = 1_000_000.0
    consensus_precision: int = 100_000
    liquid_alpha_consensus_mode: Literal["CURRENT", "PREVIOUS", "MIXED"] = "CURRENT"

@dataclass
class YumaParams:
    bond_moving_avg: float = 0.1
    liquid_alpha: bool = False
    alpha_high: float = 0.3
    alpha_low: float = 0.1
    decay_rate: float = 0.1
    capacity_alpha: float = 0.1
    alpha_sigmoid_steepness: float = 10.0
    override_consensus_high: float | None = None
    override_consensus_low: float | None = None

@dataclass
class YumaConfig:
    simulation: SimulationHyperparameters = field(
        default_factory=SimulationHyperparameters
    )
    yuma_params: YumaParams = field(default_factory=YumaParams)

    def __post_init__(self):
        # Flatten fields for direct access
        simulation_dict = asdict(self.simulation)
        yuma_params_dict = asdict(self.yuma_params)

        for key, value in simulation_dict.items():
            setattr(self, key, value)

        for key, value in yuma_params_dict.items():
            setattr(self, key, value)


def Yuma2c(
    W: torch.Tensor,
    S: torch.Tensor,
    B_old: torch.Tensor | None = None,
    config: YumaConfig = YumaConfig(),
    maxint: int = 2**64 - 1,
) -> dict[str, torch.Tensor | None | float]:
    """
    Implements the Yuma2C algorithm for managing validator bonds, weights, and incentives
    in a decentralized system.

    Yuma2C addresses the shortcomings of the Yuma2B algorithm, which does not solve the
    problem of weight clipping influencing bonds effectively. Yuma2 assumes that the
    "Big Validator" will allocate weights to the "new best" server in the next epoch
    after it is discovered by the "Small Validators." However, this leads to a drop in
    the bonds of the "Small Validators" after the next epoch, highlighting the need for
    a more robust solution.

    Yuma2C introduces a robust bond accumulation mechanism that allows validators to accrue
    bonds over time. This mitigates the issues caused by weight clipping influencing bonds
    and ensures sustained validator engagement by tying bond accrual to stake and weights.

    Key Features:
    - Validators with higher stakes can accumulate more bonds, directly influencing their dividends.
    - Bonds are capped by the maximum capacity per validator-server relation, which is proportional
      to the validator's stake.
    - Bonds are adjusted per epoch based on the `capacity_alpha` parameter, which limits the bond
      purchase power.
    - A decay mechanism ensures that bonds associated with unsupported servers decrease over time.
    """

    # === Weight ===
    W = (W.T / (W.sum(dim=1) + 1e-6)).T

    # === Stake ===
    S = S / S.sum()

    # === Prerank ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus ===
    C = torch.zeros(W.shape[1], dtype=torch.float64)

    for i, miner_weight in enumerate(W.T):
        c_high = 1.0
        c_low = 0.0

        while (c_high - c_low) > 1 / config.consensus_precision:
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

    B_norm = B / (B.sum(dim=0, keepdim=True) + 1e-6)

    # === Dividends Calculation ===
    D = (B_norm * I).sum(dim=1)

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
        "server_trust": T,
        "validator_trust": T_v,
        "validator_bonds": B,
        "validator_reward": D,
        "validator_reward_normalized": D_normalized,
    }