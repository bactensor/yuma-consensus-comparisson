
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


def Yuma2b(
    W: torch.Tensor,
    W_prev: torch.Tensor,
    S: torch.Tensor,
    num_servers: int,
    num_validators: int,
    use_full_matrices: bool,
    B_old: torch.Tensor | None = None,
    C_old: torch.Tensor | None = None,
    config: YumaConfig = YumaConfig(),
) -> dict[str, torch.Tensor | None | float]:
    """
    Yuma2B is designed to solve the problem of weight clipping influencing the current bond values for small stakers (validators).
    The Bonds from the previous epoch are used to calculate Bonds EMA.
    """

    # === Weight ===
    W = (W.T / (W.sum(dim=1) + 1e-6)).T

    if W_prev is None:
        W_prev = W

    # === Stake ===
    S = S / S.sum()

    # === Prerank ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus ===
    C = torch.zeros(W.shape[1])

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
    W_clipped = torch.min(W_prev, C)

    # === Rank ===
    R = (S.view(-1, 1) * W_clipped).sum(dim=0)

    # === Incentive ===
    I = (R / R.sum()).nan_to_num(0)

    # === Trusts ===
    T = (R / P).nan_to_num(0)
    T_v = W_clipped.sum(dim=1) / W.sum(dim=1)

    # === Bonds ===
    W_b = (1 - config.bond_penalty) * W_prev + config.bond_penalty * W_clipped
    B = S.view(-1, 1) * W_b / (S.view(-1, 1) * W_b).sum(dim=0)
    B = B.nan_to_num(0)

    a = b = torch.tensor(float("nan"))
    alpha = 1 - config.bond_moving_avg
    if config.liquid_alpha and (C_old is not None):
        from .simulation_utils import _compute_liquid_alpha
        alpha = _compute_liquid_alpha(
            W=W,
            B=B_old,
            C=C,
            alpha_sigmoid_steepness=config.alpha_sigmoid_steepness,
            alpha_low=config.alpha_low,
            alpha_high=config.alpha_high,
            num_validators=num_validators,
            num_servers=num_servers,
            use_full_matrices=use_full_matrices,
        )

    if B_old is not None:
        B_ema = alpha * B + (1 - alpha) * B_old
    else:
        B_ema = B

    # === Dividend ===
    D = (B_ema * I).sum(dim=1)
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
        "weight_for_bond": W_b,
        "validator_bond": B,
        "validator_ema_bond": B_ema,
        "validator_reward": D,
        "validator_reward_normalized": D_normalized,
        "alpha": alpha,
        "alpha_a": a,
        "alpha_b": b,
    }