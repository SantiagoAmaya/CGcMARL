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

"""
utils.py - Enhanced version
===========================
Stateless helper functions with additional utilities for production use.
"""

from typing import Dict, Sequence, Tuple, Union, Optional, Any
import numpy as np
import torch
import hashlib
import json


def to_tensors(
    d: Dict[str, Union[np.ndarray, torch.Tensor]], 
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    flatten: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Convert a dict of observations to tensors on the target device.
    
    Parameters
    ----------
    d : dict
        Observations indexed by agent ID
    device : torch.device
        Target device for tensors
    dtype : torch.dtype
        Target dtype (default: float32)
    flatten : bool
        Whether to flatten multi-dimensional inputs (default: True)
    
    Returns
    -------
    Dict[str, torch.Tensor]
        Processed observations
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            t = v.to(device=device, dtype=dtype)
        else:
            t = torch.as_tensor(v, dtype=dtype, device=device)
        
        if flatten and t.dim() > 1:
            t = t.flatten()
        
        out[k] = t
    return out


def flat_index(actions: Tuple[int, ...], sizes: Sequence[int]) -> int:
    """
    Mixed-radix encoding: converts multi-agent actions to single index.
    
    Parameters
    ----------
    actions : tuple of int
        Individual agent actions
    sizes : sequence of int
        Action space size for each agent
        
    Returns
    -------
    int
        Flat index in [0, prod(sizes))
        
    Examples
    --------
    >>> flat_index((1, 2), (3, 4))  # agent0: action 1 of 3, agent1: action 2 of 4
    6  # = 1 * 4 + 2
    """
    idx = 0
    for a, base in zip(actions, sizes):
        if not 0 <= a < base:
            raise ValueError(f"Action {a} out of range [0, {base})")
        idx = idx * base + a
    return idx


def unflatten_index(idx: int, sizes: Sequence[int]) -> Tuple[int, ...]:
    """
    Inverse of flat_index: converts flat index back to tuple of actions.
    
    Parameters
    ----------
    idx : int
        Flat index
    sizes : sequence of int
        Action space size for each agent
        
    Returns
    -------
    tuple of int
        Individual agent actions
        
    Examples
    --------
    >>> unflatten_index(6, (3, 4))
    (1, 2)
    """
    if idx < 0 or idx >= np.prod(sizes):
        raise ValueError(f"Index {idx} out of range [0, {np.prod(sizes)})")
    
    actions = []
    for size in reversed(sizes):
        actions.append(idx % size)
        idx //= size
    return tuple(reversed(actions))


def compute_returns(rewards: Sequence[float], gamma: float = 1.0) -> np.ndarray:
    """
    Compute discounted returns for a trajectory.
    
    Parameters
    ----------
    rewards : sequence of float
        Rewards for each timestep
    gamma : float
        Discount factor
        
    Returns
    -------
    np.ndarray
        Returns for each timestep
    """
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0
    
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    return returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize advantages to have zero mean and unit variance.
    
    Parameters
    ----------
    advantages : torch.Tensor
        Raw advantages
    eps : float
        Small constant for numerical stability
        
    Returns
    -------
    torch.Tensor
        Normalized advantages
    """
    mean = advantages.mean()
    std = advantages.std() + eps
    return (advantages - mean) / std


def get_state_hash(state: Any) -> str:
    """
    Generate a hash for a state (useful for tabular methods or caching).
    
    Parameters
    ----------
    state : any
        State to hash (must be JSON-serializable or tensor/array)
        
    Returns
    -------
    str
        Hexadecimal hash string
    """
    if isinstance(state, (torch.Tensor, np.ndarray)):
        # Convert to bytes for hashing
        state_bytes = state.cpu().numpy().tobytes() if isinstance(state, torch.Tensor) else state.tobytes()
    else:
        # Try JSON serialization for other types
        try:
            state_bytes = json.dumps(state, sort_keys=True).encode()
        except (TypeError, ValueError):
            # Fallback to string representation
            state_bytes = str(state).encode()
    
    return hashlib.md5(state_bytes).hexdigest()


def polyak_update(
    target_params: Dict[str, torch.nn.Parameter],
    source_params: Dict[str, torch.nn.Parameter],
    tau: float = 0.001
) -> None:
    """
    Polyak averaging update for target networks.
    
    Parameters
    ----------
    target_params : dict
        Target network parameters to update
    source_params : dict
        Source network parameters
    tau : float
        Polyak averaging coefficient (0 = no update, 1 = full copy)
    """
    with torch.no_grad():
        for key in target_params:
            target_params[key].data.mul_(1 - tau)
            target_params[key].data.add_(tau * source_params[key].data)


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Compute explained variance (useful for value function quality).
    
    Parameters
    ----------
    y_pred : torch.Tensor
        Predictions
    y_true : torch.Tensor
        True values
        
    Returns
    -------
    float
        Explained variance in [0, 1], where 1 is perfect prediction
    """
    var_y = torch.var(y_true)
    return float(1 - torch.var(y_true - y_pred) / (var_y + 1e-8))


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Parameters
    ----------
    prefer_cuda : bool
        Whether to prefer CUDA if available
        
    Returns
    -------
    torch.device
        Best available device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    else:
        return torch.device("cpu")


class RunningMeanStd:
    """
    Maintains running statistics for normalization.
    Useful for observation/reward normalization.
    """
    
    def __init__(self, shape: Tuple[int, ...] = (), eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps
        self.eps = eps
    
    def update(self, x: np.ndarray) -> None:
        """Update statistics with new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / tot_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        
        self.var = M2 / tot_count
        self.count = tot_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using running statistics."""
        return (x - self.mean) / np.sqrt(self.var + self.eps)