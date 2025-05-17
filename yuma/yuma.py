
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


def Yuma3(
    W: torch.Tensor,
    S: torch.Tensor,
    num_servers: int,
    num_validators: int,
    use_full_matrices: bool,
    B_old: torch.Tensor | None = None,
    C_old: torch.Tensor | None = None,
    config: YumaConfig = YumaConfig(),
) -> dict[str, torch.Tensor | None | float]:
    """
    Implements the Yuma3 algorithm for managing validator bonds, weights, and incentives
    in a decentralized system.

    Yuma3 addresses the shortcomings of the Yuma2B algorithm, which does not resolve the
    problem of stake dynamics differences between validators concerning their bonds. The case 9 scenario shows that
    when a validator reaches its maximum bond cap and subsequently increases its stake relative to other validators,
    its bonds will not scale proportionally with its stake. This results in reduced dividends compared to other validators.

    Yuma3 adjusts the bond accumulation mechanism to ensure that each Validator/Server relationship has its own relative scale of bonds, capped at 1.
    Stake is no longer considered when calculating bonds but is only used to calculate the final dividends.
    This resolves the stake-bond dynamic issues caused by the Yuma2B implementation.

    Key Features:
    - Each Validator/Server relationship has its own relative scale of bonds, capped at 1, ensuring fairness among validators regardless of stake size.
    - Bonds are no longer directly tied to stake for their accumulation. Instead, stake is only used to calculate final dividends, decoupling the stake-bond dynamics.
    - Validators allocate bond purchases based on weights assigned to servers, reflecting their intended distribution of influence.
    - Bonds are adjusted per epoch according to the `alpha` parameter, which determines the maximum increment per epoch.
    - A decay mechanism ensures that bonds for unsupported servers decrease over time.
    - The `liquid_alpha` adjustment, when enabled, dynamically adapts bond accumulation based on the consensus range of server weights, providing a more responsive and adaptive allocation mechanism.
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

    # === Bond Initialization ====
    if B_old is None:
        B_old = torch.zeros_like(W)

    # === Liquid Alpha Adjustment ===
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

    # === Bonds ===
    B_decayed = B_old * (1 - alpha)
    remaining_capacity = 1.0 - B_decayed
    remaining_capacity = torch.clamp(remaining_capacity, min=0.0)

    # Each validator can increase bonds by at most alpha per epoch towards the cap
    purchase_increment = (
        alpha * W
    )  # Validators allocate their purchase across miners based on weights
    # Ensure that purchase does not exceed remaining capacity
    purchase = torch.min(purchase_increment, remaining_capacity)

    B = B_decayed + purchase
    B = torch.clamp(B, max=1.0)

    # === Dividends Calculation ===
    B_norm = B / (B.sum(dim=0, keepdim=True) + 1e-6) # Normalized Bonds only for Dividends calculations purpose
    total_bonds_per_validator = (B_norm * I).sum(dim=1)  # Sum over miners for each validator
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
        "validator_reward_normalized": D_normalized,
    }