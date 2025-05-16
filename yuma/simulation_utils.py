import torch

def _slice_tensors(
    *tensors: torch.Tensor,
    num_validators: int,
    num_servers: int,
) -> tuple[torch.Tensor]:
    """
    Applies a uniform slicing rule to each provided tensor:
    """
    sliced_tensors = []
    for tensor in tensors:
        if tensor.dim() == 1:
            sliced_tensors.append(tensor[-num_servers:])
        elif tensor.dim() == 2:
            sliced_tensors.append(tensor[:num_validators, -num_servers:])
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}. Only 1D or 2D allowed.")
    return tuple(sliced_tensors)

def full_matrices(func):
    def wrapper(
        W: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        alpha_sigmoid_steepness: float,
        alpha_low: float,
        alpha_high: float,
        num_validators: int,
        num_servers: int,
        use_full_matrices: bool,
        ):
        if use_full_matrices:
            W_slice, B_slice, C_slice = _slice_tensors(W, B, C,
                                                       num_validators=num_validators,
                                                       num_servers=num_servers)
        else:
            W_slice, B_slice, C_slice = W, B, C

        alpha_slice = func(W_slice, B_slice, C_slice, alpha_sigmoid_steepness, alpha_low, alpha_high)

        if use_full_matrices:
            alpha_full = torch.full_like(W, fill_value=0.0)
            alpha_full[:num_validators, -num_servers:] = alpha_slice
            return alpha_full
        return alpha_slice
    return wrapper

@full_matrices
def _compute_liquid_alpha(
    W: torch.tensor,
    B: torch.tensor,
    C: torch.tensor,
    alpha_sigmoid_steepness: float,
    alpha_low: float,
    alpha_high: float,
    ):
    """
    Liquid alpha is computed using a combination of previous epoch consensus weights, previous epoch bonds, and current epoch weights.

    Buying Bonds:
    When the current epoch weights exceed the previous epoch bonds, it indicates that the validator intends to purchase bonds.
    The greater the discrepancy between the current weights and the previous epoch consensus weights, the more Liquid Alpha 2.0 will shift toward the alpha low value, facilitating faster bond acquisition.

    Selling Bonds:
    When the current epoch weights are lower than the previous epoch bonds, it signals that the validator aims to sell bonds.
    The larger the difference between the current epoch weights and the previous epoch bonds, the more Liquid Alpha 2.0 will adjust toward the alpha low value, enabling faster bond liquidation.
    """
    buy_mask = (W >= B)
    sell_mask = (W < B)
    
    diff_buy = (W - C).clamp(min=0.0, max=1.0)
    diff_sell = (B - W).clamp(min=0.0, max=1.0)
    
    combined_diff = torch.where(buy_mask, diff_buy, diff_sell)
    
    combined_diff = 1.0 / (1.0 + torch.exp(-alpha_sigmoid_steepness * (combined_diff - 0.5)))
    
    alpha_slice = alpha_low + combined_diff * (alpha_high - alpha_low)
    return alpha_slice.clamp(alpha_low, alpha_high)