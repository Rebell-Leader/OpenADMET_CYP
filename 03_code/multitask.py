"""Shared-trunk multitask network for the four CYP pIC50 endpoints.

Memo 3 §3.3. The four models we ship are independent fits; the organizers' own
baselines are multitask (`n_tasks: 4`). The case for sharing is the label matrix:
of 4,905 training compounds, 3,596 carry exactly ONE pIC50 label and only 41 carry
all four (33.3 % filled). Four independent fits therefore see 1,285-2,335 rows each,
while a masked-loss trunk sees all 4,905 and lets every row shape the shared
representation.

The auxiliary task is the reason this is worth trying after pseudo-labelling returned
only -0.008. The single-concentration screen fills 89.2 % of the same matrix
(4,375 compounds x 4 enzymes). Supervising extra HEADS on it is a third mechanism,
distinct from the two already tested in Memo 1 §5b:

  * as a FEATURE (distillation)  -> failed, +0.036: a structure-derived surrogate adds
    no information the model did not already have.
  * as extra ROWS (pseudo-labels) -> worked modestly, -0.008, but only on the two
    isoforms an a-priori floor-shift rule selects.
  * as an auxiliary TASK (here)   -> the gradient signal shapes the shared trunk during
    training but never enters at inference, so it cannot be a lossy re-encoding of the
    input the way the distilled feature was.

CYP2D6 is held out of the shared trunk by default (`d2d6_separate`). Memo 2 §2: its
mean cross-isoform potency correlation is -0.002 and basic-nitrogen count is the only
simple descriptor positively correlated with it while lipophilicity is inert - its SAR
is orthogonal, so forcing it through a trunk shaped by the other three is expected to
hurt. That expectation is tested rather than assumed: `variant="mt_all4"` shares all
four, and the ablation is reported.

Every scaler and every target standardisation is fitted inside the training fold only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class MTConfig:
    """Hyperparameters for :class:`MultitaskNet` and :func:`fit_multitask`."""

    hidden: tuple[int, ...] = (512, 256)
    head_hidden: int = 64
    dropout: float = 0.25
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 300
    batch_size: int = 256
    patience: int = 40
    aux_weight: float = 0.0
    seed: int = 0
    device: str = "cuda"
    tasks: tuple[str, ...] = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
    aux_tasks: tuple[str, ...] = field(default_factory=tuple)


class MultitaskNet(nn.Module):
    """Shared trunk with one small head per task.

    The trunk is a plain MLP; capacity lives mostly there, with deliberately small
    heads (``head_hidden`` defaults to 64) so that the shared layers are forced to
    carry the transferable chemistry rather than each head re-learning it privately.

    Args:
        d_in: Input feature width.
        n_main: Number of primary (pIC50) heads.
        n_aux: Number of auxiliary (log2fc) heads; 0 disables them.
        cfg: Hyperparameter bundle.
    """

    def __init__(self, d_in: int, n_main: int, n_aux: int, cfg: MTConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for h in cfg.hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.main_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(prev, cfg.head_hidden), nn.ReLU(), nn.Linear(cfg.head_hidden, 1))
            for _ in range(n_main)
        )
        self.aux_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(prev, cfg.head_hidden), nn.ReLU(), nn.Linear(cfg.head_hidden, 1))
            for _ in range(n_aux)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(main_predictions, aux_predictions_or_None)``, each ``[n, n_heads]``."""
        z = self.trunk(x)
        main = torch.cat([h(z) for h in self.main_heads], dim=1)
        aux = torch.cat([h(z) for h in self.aux_heads], dim=1) if len(self.aux_heads) else None
        return main, aux


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over observed entries only.

    The label matrix is 33 % filled, so an unmasked loss would train every head on
    every row and drive the missing entries toward zero. Averaging over the mask keeps
    each head's gradient scale independent of how sparse that particular task is.

    Returns a zero scalar (with grad) when the batch has no observed entry at all.
    """
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return (((pred - target) ** 2) * mask).sum() / mask.sum()


def fit_multitask(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_va: np.ndarray,
    Y_va: np.ndarray,
    cfg: MTConfig,
    A_tr: np.ndarray | None = None,
    A_va: np.ndarray | None = None,
) -> tuple[MultitaskNet, dict[str, float | int]]:
    """Train one multitask network with early stopping on masked validation loss.

    Targets are standardised per task using training-fold statistics only; predictions
    are returned on the original scale by :func:`predict_multitask`, which reads the
    statistics stashed on the returned module.

    Args:
        X_tr, X_va: Feature matrices.
        Y_tr, Y_va: Primary targets, ``[n, n_tasks]``, NaN where unobserved.
        cfg: Hyperparameters.
        A_tr, A_va: Optional auxiliary targets, same convention.

    Returns:
        The module (with ``y_mu_``/``y_sd_`` attached) and a history dict.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    m_tr, m_va = np.isfinite(Y_tr), np.isfinite(Y_va)
    y_mu = np.array([Y_tr[m_tr[:, j], j].mean() for j in range(Y_tr.shape[1])])
    y_sd = np.array([Y_tr[m_tr[:, j], j].std() + 1e-8 for j in range(Y_tr.shape[1])])
    Yz_tr = np.nan_to_num((Y_tr - y_mu) / y_sd)
    Yz_va = np.nan_to_num((Y_va - y_mu) / y_sd)

    use_aux = A_tr is not None and cfg.aux_weight > 0
    if use_aux:
        ma_tr = np.isfinite(A_tr)
        a_mu = np.array([A_tr[ma_tr[:, j], j].mean() for j in range(A_tr.shape[1])])
        a_sd = np.array([A_tr[ma_tr[:, j], j].std() + 1e-8 for j in range(A_tr.shape[1])])
        Az_tr = np.nan_to_num((A_tr - a_mu) / a_sd)
        Ma_tr = ma_tr.astype(np.float32)

    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    Xt, Yt, Mt = t(X_tr), t(Yz_tr), t(m_tr.astype(np.float32))
    Xv, Yv, Mv = t(X_va), t(Yz_va), t(m_va.astype(np.float32))
    At = t(Az_tr) if use_aux else None
    MAt = t(Ma_tr) if use_aux else None

    net = MultitaskNet(
        X_tr.shape[1], Y_tr.shape[1], A_tr.shape[1] if use_aux else 0, cfg
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best, best_state, bad = np.inf, None, 0
    n = len(Xt)
    for ep in range(cfg.epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, cfg.batch_size):
            b = perm[s : s + cfg.batch_size]
            if len(b) < 2:  # BatchNorm needs >1 row
                continue
            opt.zero_grad()
            pm, pa = net(Xt[b])
            loss = masked_mse(pm, Yt[b], Mt[b])
            if use_aux:
                loss = loss + cfg.aux_weight * masked_mse(pa, At[b], MAt[b])
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            pm, _ = net(Xv)
            vl = float(masked_mse(pm, Yv, Mv))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.y_mu_, net.y_sd_ = y_mu, y_sd
    return net, {"best_val": best, "epochs_run": ep + 1}


def predict_multitask(net: MultitaskNet, X: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Predict primary targets on the original (un-standardised) scale."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    net.eval()
    with torch.no_grad():
        pm, _ = net(torch.tensor(X, dtype=torch.float32, device=dev))
    return pm.cpu().numpy() * net.y_sd_ + net.y_mu_
