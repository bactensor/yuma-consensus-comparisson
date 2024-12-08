import torch
from typing import Optional


def run_simulation(
    validators: list[str],
    stakes: list[torch.Tensor],
    weights: list[torch.Tensor],
    num_epochs: int,
    config,
) -> tuple[dict[str, list[float]], list[torch.Tensor]]:
    """
    Runs the simulation over multiple epochs using the specified Yuma function.
    """
    from yuma.yuma import Yuma2, YumaConfig
    dividends_per_validator = {validator: [] for validator in validators}
    B_state: Optional[torch.Tensor] = None
    W_prev: Optional[torch.Tensor] = None

    for epoch in range(num_epochs):
        W = weights[epoch]
        S = stakes[epoch]

        stakes_tao = S * config.total_subnet_stake
        stakes_units = stakes_tao / 1_000

        result = Yuma2(W=W, W_prev=W_prev, S=S, B_old=B_state, config=config)

        B_state = result['validator_ema_bond']
        W_prev = result['weight']
        D_normalized = result['validator_reward_normalized']

        E_i = config.validator_emission_ratio * D_normalized
        validator_emission = E_i * config.total_epoch_emission

        for i, validator in enumerate(validators):
            stake_unit = stakes_units[i].item()
            validator_emission_i = validator_emission[i].item()

            if stake_unit > 1e-6:
                dividend_per_1000_tao = validator_emission_i / stake_unit
            else:
                dividend_per_1000_tao = 0.0  # No stake means no dividend per 1000 Tao

            dividends_per_validator[validator].append(dividend_per_1000_tao)


    return dividends_per_validator

def calculate_total_dividends(
    validators: list[str],
    dividends_per_validator: dict[str, list[float]],
    base_validator: str,
    num_epochs: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Calculates the total dividends per validator and computes the percentage difference
    relative to the provided base validator.

    Returns:
        total_dividends: Dict mapping validator names to their total dividends.
        percentage_diff_vs_base: Dict mapping validator names to their percentage difference vs. base.
    """
    total_dividends = {}
    for validator in validators:
        dividends = dividends_per_validator.get(validator, [])
        dividends = dividends[:num_epochs]
        total_dividend = sum(dividends)
        total_dividends[validator] = total_dividend

    # Get base dividend
    base_dividend = total_dividends.get(base_validator, None)
    if base_dividend is None or base_dividend == 0.0:
        print(f"Warning: Base validator '{base_validator}' has zero or missing total dividends.")
        base_dividend = 1e-6  # Assign a small epsilon value to avoid division by zero

    # Compute percentage difference vs base for each validator
    percentage_diff_vs_base = {}
    for validator, total_dividend in total_dividends.items():
        if validator == base_validator:
            percentage_diff_vs_base[validator] = 0.0  # Base validator has 0% difference vs itself
        else:
            percentage_diff = ((total_dividend - base_dividend) / base_dividend) * 100.0
            percentage_diff_vs_base[validator] = percentage_diff

    return total_dividends, percentage_diff_vs_base
