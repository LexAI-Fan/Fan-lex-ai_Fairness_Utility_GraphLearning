
from typing import Tuple
import torch

def demographic_parity_gap(preds: torch.Tensor, S: torch.Tensor, threshold: float = 0.5) -> float:
    yhat = (preds >= threshold).float()
    p1 = yhat[S == 1].mean() if (S == 1).any() else torch.tensor(0.0, device=preds.device)
    p0 = yhat[S == 0].mean() if (S == 0).any() else torch.tensor(0.0, device=preds.device)
    return float((p1 - p0).abs().item())

def equalized_odds_gap(preds: torch.Tensor, y_true: torch.Tensor, S: torch.Tensor, threshold: float = 0.5) -> float:
    yhat = (preds >= threshold).float()
    mask_pos = (y_true == 1)
    if not mask_pos.any():
        return 0.0
    p1 = (yhat[mask_pos & (S == 1)]).float().mean() if (mask_pos & (S == 1)).any() else torch.tensor(0.0, device=preds.device)
    p0 = (yhat[mask_pos & (S == 0)]).float().mean() if (mask_pos & (S == 0)).any() else torch.tensor(0.0, device=preds.device)
    return float((p1 - p0).abs().item())

def corr_with_sensitive(preds: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    x = preds.float()
    s = S.float()
    x = (x - x.mean()) / (x.std() + 1e-8)
    s = (s - s.mean()) / (s.std() + 1e-8)
    return (x * s).mean()
