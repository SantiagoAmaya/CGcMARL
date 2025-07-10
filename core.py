
"""
core.py
========
Core MARL components: networks, critics, planner, and the `Agent` façade.

Overview
--------
This module hosts the *algorithmic heart* of the three-layer public-goods MARL
prototype.  All stateful learning logic lives here; everything else in the
package is either plumbing (`logger.py`, `utils.py`) or orchestration
(`train.py`).

Key classes
-----------
• **MLP** – generic fully-connected network used by every value function.  
• **RegionCritic** – Q-network that estimates joint action-values for a single
  *region* (subset of agents).  
• **MaxSumPlanner** – message-passing solver (Max-Sum / factor graph) that
  combines all region critics to pick a global joint action.  
• **Agent** – convenience wrapper that owns *one* `RegionCritic` per region, the
  shared `MaxSumPlanner`, and handles TD-λ updates.

Inter-module relations
----------------------
* `utils.flat_index` is imported for mixed-radix indexing of joint actions.  
* Each `RegionCritic` may emit diagnostics through a `(log_fn, …)` callable that
  `train.py` passes in after retrieving it from `logger.setup_logger`.  
* `Agent` is the *only* public symbol training code should touch:
  `from core import Agent`.

External dependencies
---------------------
PyTorch ≥1.12 (for nn.Module, optim), NumPy, and optionally CUDA.

"""

# ---------------------------------------------------------------
#  Standard libraries
# ---------------------------------------------------------------

from urllib.parse import quote
from typing import Dict,  List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------
#  Third‑party libraries
# ---------------------------------------------------------------
import torch
import torch.nn as nn
from torch.optim import SGD 
from utils import flat_index

###############################################################################
# MLP                                                                         #
###############################################################################

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

###############################################################################
# Region critic (joint)                                                       #
###############################################################################

class RegionCritic(nn.Module):
    def __init__(self,
                 obs_dim_per_agent: int,
                 lambda_dim: int,
                 action_sizes: Sequence[int]):        # ← pass list here
        super().__init__()
        # e.g. [9, 5, 9] → heterogeneous branch factors
        self.sizes = [int(s) for s in action_sizes]
        input_dim = len(self.sizes) * (obs_dim_per_agent + lambda_dim)
        output_dim = int(np.prod(self.sizes))
        self.q_net = MLP(input_dim, output_dim)
        # Baseline must persist in checkpoints **but** stay out of
        # .parameters().  Register it as a buffer instead.
        self.register_buffer("avg_reward", torch.zeros(1))

    def forward(self, obs_concat: torch.Tensor) -> torch.Tensor:
        return self.q_net(obs_concat)


###############################################################################
# Max–Sum planner                                   #
###############################################################################

class MaxSumPlanner:
    """Max–Sum belief‑propagation planner that now allows **heterogeneous**
    action sizes – each message vector length equals the recipient agent’s
    branch factor `A_i`.  Supply a mapping `action_sizes[agent_id] → A_i`.
    """

    def __init__(
        self,
        region_graph: Dict[str, List[str]],
        action_sizes: Dict[str, int],            # ← new
        n_iters: int = 3,
        epsilon: float = 0.05,
        device: torch.device | str = "cpu",
    ):
        self.region_graph = region_graph
        self.sizes = {k: int(v) for k, v in action_sizes.items()}
        self.n_iters = n_iters
        self.epsilon = epsilon
        self.device = torch.device(device)

        # message buffers   R→i  and  i→R
        self.msg_R_to_i: Dict[Tuple[str, str], torch.Tensor] = {}
        self.msg_i_to_R: Dict[Tuple[str, str], torch.Tensor] = {}
        self._init_buffers()

    # ---------------------------------------------------------------
    def _zeros(self, agent_id: str) -> torch.Tensor:
        return torch.zeros(self.sizes[agent_id], device=self.device)

    def _init_buffers(self):
        """(Re-)initialise message dictionaries on **current** device."""
        self.msg_R_to_i.clear()
        self.msg_i_to_R.clear()
        for R, agents in self.region_graph.items():
            for i in agents:
                self.msg_R_to_i[(R, i)] = self._zeros(i)
                self.msg_i_to_R[(i, R)] = self._zeros(i)

    def reset_messages(self):
        for buf in (*self.msg_R_to_i.values(), *self.msg_i_to_R.values()):
            buf.zero_()

    @staticmethod
    def _broadcast(vec: torch.Tensor, axis: int, rank: int) -> torch.Tensor:
        shape = [1] * rank
        shape[axis] = vec.shape[0]
        return vec.reshape(*shape)

    # ---------------------------------------------------------------
    def step(self, q_tables: Dict[str, torch.Tensor]):
        for _ in range(self.n_iters):
            # factor → variable
            for R, agents in self.region_graph.items():
                if R not in q_tables:
                    continue
                joint_q = q_tables[R]  # shape [A_i]*k (mixed‑radix)
                rank = joint_q.dim()
                for axis, i in enumerate(agents):
                    inc = [self._broadcast(self.msg_i_to_R[(j, R)], ax_j, rank)
                           for ax_j, j in enumerate(agents) if j != i]
                    # Vectorised accumulation: avoids allocating a zero-sized
                    # buffer every iteration.
                    if inc:
                        augmented = joint_q + torch.stack(inc).sum(dim=0)
                    else:
                        augmented = joint_q

                    red_dims = tuple(d for d in range(rank) if d != axis)
                    self.msg_R_to_i[(R, i)] = torch.amax(augmented, dim=red_dims)
            # variable → factor
            for i in {a for agents in self.region_graph.values() for a in agents}:
                incidents = [R for R in self.region_graph if (R in q_tables and i in self.region_graph[R])]
                if not incidents:
                    continue
                base = self._zeros(i)
                for R in incidents:
                    agg = base.clone()
                    for R_dash in incidents:
                        if R_dash != R:
                            agg += self.msg_R_to_i[(R_dash, i)]
                    self.msg_i_to_R[(i, R)] = agg

    # ---------------------------------------------------------------
    def select_actions(self) -> Dict[str, int]:
        actions: Dict[str, int] = {}
        for i in {a for agents in self.region_graph.values() for a in agents}:
            msgs = [self.msg_R_to_i[(R, i)]
                for R in self.region_graph
                if (R, i) in self.msg_R_to_i]
            score = torch.stack(msgs).sum(dim=0) if msgs else self._zeros(i)
            a_max = int(torch.argmax(score).item())
            if torch.rand((), device=self.device).item() < self.epsilon:
                actions[i] = int(torch.randint(self.sizes[i], (1,), device=self.device).item())
            else:
                actions[i] = a_max
        return actions

    def to(self, device: torch.device | str):
        new_dev = torch.device(device)
        for d in (self.msg_R_to_i, self.msg_i_to_R):
            for k, v in d.items():
                d[k] = v.to(new_dev)
        self.device = new_dev
        return self

###############################################################################
# Agent (owns traces and multipliers)                                         #
###############################################################################

class Agent:
    def __init__(
        self, agent_id: str, owned_regions: List[str], traces_refs: Dict[str, Dict[str, torch.Tensor]],
        lambda_dim: int, device: torch.device, alpha: float, alpha_rho: float, lambda_e: float,
        gamma: float, grad_clip: float, optimizers: Dict[str, SGD],
    ):
        self.id = agent_id
        self.owned_regions = owned_regions  # list of region names where this agent is **owner** (first in list)
        self.traces_refs = traces_refs      # region→param_name→trace tensor
        self.lambda_vec = torch.zeros(lambda_dim, device=device)
        self.device = device
        self.alpha = alpha
        self.alpha_rho = alpha_rho
        self.lambda_e = lambda_e
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.optims = optimizers                          # new
        self.prev_q: Dict[str, torch.Tensor] = {} 

    # ----------------------------------------------
    def td_update_region(self, region: str, critic: RegionCritic, joint_obs: torch.Tensor, joint_action: Tuple[int, ...], reward: float, next_joint_obs: torch.Tensor):
        sizes = critic.sizes
        idx = flat_index(joint_action, sizes)
        q_vec = critic(joint_obs)
        q_val = q_vec[idx]
        with torch.no_grad():
            next_q = critic(next_joint_obs).max()
        delta = reward - critic.avg_reward.item() + next_q - q_val

# ------------------ true-online differential TD(λ) ------------------
        critic.zero_grad()
        q_val.backward(retain_graph=False)

        traces = self.traces_refs[region]
        opt = self.optims[region]

        for name, p in critic.named_parameters():
            if p.grad is None:
                continue

            grad = p.grad
            e = traces[name]

            # true-online eligibility trace update
            dot = torch.dot(e.flatten(), grad.flatten())
            e.mul_(self.gamma * self.lambda_e).add_(
                grad * (1.0 - self.alpha * self.gamma * self.lambda_e * dot)
            )

            # place final gradient for optimiser
            p.grad = delta * e

        # optional clipping
        torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)

        # optimiser step
        opt.step()
        opt.zero_grad(set_to_none=True)

        # running average reward baseline
        with torch.no_grad():
            critic.avg_reward += self.alpha_rho * delta

        # store last Q for next step (needed by full TO formulation – kept for extension)
        self.prev_q[region] = next_q.detach()

        return float(delta)