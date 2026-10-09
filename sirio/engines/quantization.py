"""Quantizzazione int8 "per riga" dei livelli lineari (solo CPU, motore locale).

La quantizzazione dinamica standard di PyTorch (``quantize_dynamic``) usa una
sola scala per tutto il tensore d'ingresso: il risultato di una riga dipende
quindi dalle *altre* righe del lotto (con le letture a lotti del motore locale
le probabilita' cambierebbero a seconda dei ritagli letti insieme).
``Int8Linear`` quantizza invece ogni riga d'ingresso con la propria scala e i
pesi con una scala per canale d'uscita, poi moltiplica in interi
(``torch._int_mm``, istruzioni VNNI/AMX sulle CPU recenti): il risultato di una
riga non dipende dal lotto ed e' piu' vicino a quello in virgola mobile.

Il modulo importa ``torch`` solo quando serve: il resto dell'applicazione
funziona anche senza i componenti offline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

_INT8_MAX = 127.0


def int8_matmul_available() -> bool:
    """True se ``torch._int_mm`` funziona su CPU con questa versione di PyTorch."""
    try:
        import torch  # noqa: PLC0415

        a = torch.ones((2, 8), dtype=torch.int8)
        b = torch.ones((8, 3), dtype=torch.int8)
        out = torch._int_mm(a, b)
    except Exception:  # noqa: BLE001 - versione di PyTorch senza moltiplicazione intera su CPU
        log.info("Moltiplicazione int8 non disponibile su CPU", exc_info=True)
        return False
    return bool(out.dtype == torch.int32 and int(out[0, 0]) == 8)


def _make_class() -> type:
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    class Int8Linear(nn.Module):
        """``nn.Linear`` con pesi int8 (scala per canale) e ingresso int8 (scala per riga)."""

        def __init__(self, linear: nn.Linear) -> None:
            super().__init__()
            w = linear.weight.detach().float()
            scale = w.abs().amax(dim=1).clamp(min=1e-12) / _INT8_MAX
            q = torch.round(w / scale[:, None]).clamp(-_INT8_MAX, _INT8_MAX).to(torch.int8)
            self.in_features = int(linear.in_features)
            self.out_features = int(linear.out_features)
            self.register_buffer("weight_q", q.t().contiguous())
            self.register_buffer("weight_scale", scale.view(1, -1).contiguous())
            bias = linear.bias.detach().float().clone() if linear.bias is not None else None
            self.register_buffer("bias", bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            shape = x.shape
            x2 = x.reshape(-1, shape[-1]).float()
            sx = x2.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / _INT8_MAX
            xq = torch.round(x2 / sx).clamp(-_INT8_MAX, _INT8_MAX).to(torch.int8)
            y = torch._int_mm(xq, self.weight_q).float()
            y = y * sx * self.weight_scale
            if self.bias is not None:
                y = y + self.bias
            return y.reshape(*shape[:-1], self.out_features)

        def extra_repr(self) -> str:
            return f"in_features={self.in_features}, out_features={self.out_features}, int8 per riga"

    return Int8Linear


_CLASS: list[type] = []


def int8_linear_class() -> type:
    """Classe ``Int8Linear`` (creata alla prima richiesta, quando torch e' disponibile)."""
    if not _CLASS:
        _CLASS.append(_make_class())
    return _CLASS[0]


def quantize_linears(module: Any, keep: Callable[[str, Any], bool] | None = None) -> int:
    """Sostituisce i ``nn.Linear`` di ``module`` con ``Int8Linear`` (in place).

    ``keep(nome, livello)`` -> True per lasciare un livello in virgola mobile.
    Restituisce il numero di livelli convertiti."""
    from torch import nn  # noqa: PLC0415

    cls = int8_linear_class()
    targets = [(name, mod) for name, mod in module.named_modules()
               if isinstance(mod, nn.Linear) and not (keep is not None and keep(name, mod))]
    for name, mod in targets:
        parent = module
        *path, leaf = name.split(".")
        for part in path:
            parent = getattr(parent, part)
        setattr(parent, leaf, cls(mod))
    return len(targets)


__all__ = ["int8_linear_class", "int8_matmul_available", "quantize_linears"]
