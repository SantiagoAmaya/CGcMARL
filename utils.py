"""
utils.py
=========
Stateless helper functions shared by *all* other modules.

Functions
---------
• **to_tensors(obs_dict, device)**  
  Flattens and converts a dict of arbitrary-shaped NumPy arrays / tensors into
  1-D `torch.float32` tensors on the target device.

• **flat_index(actions, sizes)**  
  Mixed-radix encoder that turns a tuple of per-agent discrete actions into a
  single contiguous index `[0, Π_i |A_i|)`.  Used by `RegionCritic` for table
  look-ups and by `MaxSumPlanner` for compact message storage.

Relationship map
----------------
``utils`` **imports nothing** from the rest of the project (it must stay
dependency-free).  In contrast:

* `core.py` uses `flat_index`.  
* `train.py` uses `to_tensors` to pre-process observations before passing them
  to `Agent`.  

Because the helpers are tiny but performance-critical, keeping them in a
separate module avoids accidental circular imports and makes unit-testing easy.

"""


# ---------------------------------------------------------------
#  Standard libraries
# ---------------------------------------------------------------

from urllib.parse import quote
from typing import Dict, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------
#  Third‑party libraries
# ---------------------------------------------------------------
import torch
###############################################################################
# Helpers                                                                     #
###############################################################################

def to_tensors(d: Dict[str, "np.ndarray | torch.Tensor"], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Convert a dict of (possibly N-D) observations to **flat** float32 tensors on
    the target device.  Any obs with dim>1 (e.g. RGB board state) is flattened
    so downstream code always receives a 1-D feature vector.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            t = v.to(device=device, dtype=torch.float32)
        else:
            t = torch.as_tensor(v, dtype=torch.float32, device=device)
        if t.dim() > 1:
            t = t.flatten()          # make it 1-D
        out[k] = t
    return out

def flat_index(actions: Tuple[int, ...], sizes: Sequence[int]) -> int:
    """
    Mixed-radix encoding.  `sizes[i]` is the number of actions for agent‐i.
    """
    idx = 0
    for a, base in zip(actions, sizes):
        assert 0 <= a < base, f"Action {a} out of range 0..{base-1}"
        idx = idx * base + a
    return idx