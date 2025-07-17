
import math, torch
from torch import nn
PI2 = 2 * math.pi

_PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
           37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
           79, 83, 89, 97, 101, 103, 107, 109, 113]

def _choose_periods(max_value, target_dim=None):
    periods, prod = [], 1
    for p in _PRIMES:
        periods.append(p)
        prod *= p
        # stop only when *both* constraints are met
        if prod > max_value and (target_dim is None or len(periods)*2 >= target_dim):
            break
    else:
        raise ValueError("extend _PRIMES or relax n_dim")
    return periods
# -------------------------------------------------------------------------
class OrdinalEmbedding(nn.Module):
    """
    A plug‑and‑play layer for an ordinal integer.
        >>> emb = OrdinalEmbedding(max_value=2**32-1, n_dim=18)  # 18‑D sphere
        >>> y = emb(torch.tensor([123456789]))
        >>> x = emb.decode(y)    # tensor([123456789])
    """

    def __init__(self, max_value: int,
                 n_dim: int | None = None,
                 periods: list[int] | None = None):
        super().__init__()

        if periods is None:
            periods = _choose_periods(max_value, n_dim)
        self.register_buffer('mods', torch.tensor(periods, dtype=torch.long),
                             persistent=False)

        self.n_dim  = len(periods) * 2
        self.norm   = 1 / math.sqrt(len(periods))
        self._build_garner_helpers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : int32 / int64 tensor of any shape
        →  float32 tensor (..., n_dim)   (unit vectors)
        """
        mods  = self.mods.to(x.device)           # (q,)
        phase = (x.unsqueeze(-1).float() / mods) * PI2
        cs    = torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)
        vec   = cs.flatten(-2) * self.norm       # (..., 2q)
        return torch.nn.functional.normalize(vec, dim=-1)

    @torch.no_grad()
    def decode(self, vec: torch.Tensor) -> torch.Tensor:
        vec  = torch.nn.functional.normalize(vec, dim=-1) / self.norm
        q    = self.mods.size(0)
        v    = vec.view(*vec.shape[:-1], q, 2)
        phases   = torch.atan2(v[..., 1], v[..., 0]) % PI2
        residues = torch.round(phases * self.mods / PI2).long() % self.mods
        x = residues[..., 0].clone()
        for i in range(1, q):
            t  = ((residues[..., i] - x) * self._invs[i]) % self.mods[i]
            x += t * self._prefix[i]
        return (x % self._Mtot).to(vec.device)

    def _build_garner_helpers(self):
        mods = self.mods.cpu().tolist()
        prefix, invs = [1], []
        for m in mods[:-1]:
            prefix.append(prefix[-1] * m)
        for Mi, m in zip(prefix, mods):
            invs.append(pow(Mi % m, -1, m))
        self.register_buffer('_prefix', torch.tensor(prefix, dtype=torch.long),
                             persistent=False)
        self.register_buffer('_invs',   torch.tensor(invs, dtype=torch.long),
                             persistent=False)
        self._Mtot = math.prod(mods)

