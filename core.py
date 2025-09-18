
"""
core.py
========
Core MARL components: networks, critics, planner, and the `Agent`.

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

from typing import Dict,  List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------
#  Third‑party libraries
# ---------------------------------------------------------------
import torch
import torch.nn as nn
from torch.optim import SGD 
from utils import flat_index
from collections import deque

# -------------------------------------------------------------------
#  Reproducibility helpers
# -------------------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)

###############################################################################
# MLP                                                                         #
###############################################################################

class MLP(nn.Module):
    @torch.jit.export
    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Expect a flattened tensor of shape (..., input_dim); give a clear
        # error early if a caller passes un-flattened observations.
        if x.dim() == 1:
            x = x.unsqueeze(0)
        exp = self.net[0].in_features
        assert x.shape[-1] == exp, (
            f"RegionCritic expected last dim {exp}, got {x.shape[-1]}"
        )
        out = self.net(x)
        return out.squeeze(0)

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
        self.q_net = MLP(input_dim, output_dim, hidden=(64, 64))
               
        # Better initialization for stability
        for m in self.q_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)  
       
        # Initialize with small value and add bounds
        self.register_buffer("avg_reward", torch.tensor([0.5], dtype=torch.float32))


    def forward(self, obs_concat: torch.Tensor) -> torch.Tensor:
        return self.q_net(obs_concat)


###############################################################################
# Max–Sum planner                                   #
###############################################################################

class MaxSumPlanner:
    """Max–Sum belief‑propagation planner that allows **heterogeneous**
    action sizes – each message vector length equals the recipient agent’s
    branch factor `A_i`.  Supply a mapping `action_sizes[agent_id] → A_i`.
    """

    def __init__(
        self,
        region_graph: Dict[str, List[str]],
        action_sizes: Dict[str, int],            
        n_iters: int = 3,
        epsilon: float = 0.05,
        device: torch.device | str = "cpu",
    ):
        self.region_graph = region_graph
        self.sizes = {k: int(v) for k, v in action_sizes.items()}
        # ------------------------------------------------------------------
        # Sanity-check: every agent mentioned in the region graph must have
        # an entry in `action_sizes`, otherwise message buffers cannot be
        # initialised later on.
        # ------------------------------------------------------------------
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

        with torch.no_grad():
           
           # Store old messages for warm start (optional)
           old_msgs = {}
           
           for key in self.msg_R_to_i:
                old_msgs[key] = self.msg_R_to_i[key].clone()
           
           for key in self.msg_R_to_i:
               # Warm start: decay old messages instead of full reset
               self.msg_R_to_i[key] = old_msgs[key] * 0.1  # Keep 10% of old info
               self.msg_R_to_i[key] = self.msg_R_to_i[key].detach()
           for key in self.msg_i_to_R:
               self.msg_i_to_R[key].zero_()
               self.msg_i_to_R[key] = self.msg_i_to_R[key].detach()



    @staticmethod
    def _broadcast(vec: torch.Tensor, axis: int, rank: int) -> torch.Tensor:
        shape = [1] * rank
        shape[axis] = vec.shape[0]
        return vec.reshape(*shape)

    # ---------------------------------------------------------------
    def step(self, q_tables: Dict[str, torch.Tensor]):
        with torch.no_grad():
            for _ in range(self.n_iters):
                # factor → variable
                for R, agents in self.region_graph.items():
                    if R not in q_tables:
                        continue
                    joint_q = q_tables[R]  # shape [A_i]*k (mixed‑radix)
                    rank = joint_q.dim()

                    # ------- factor  →  variable (R → i) messages -------

                    for axis, i in enumerate(agents):
                            inc = [self._broadcast(self.msg_i_to_R[(j, R)], ax_j, rank)
                                for ax_j, j in enumerate(agents) if j != i]
                            # Vectorised accumulation: avoids allocating a zero-sized
                            # buffer every iteration.
                            if inc:
                                augmented = joint_q.clone()
                                for msg in inc:
                                    augmented += msg
                            else:
                                augmented = joint_q

                            red_dims = tuple(d for d in range(rank) if d != axis)
                            new_msg = torch.amax(augmented, dim=red_dims)
                            new_msg -= new_msg.max()  # Simple gauge fix

                            # Add damping for stability (0.5 is typical)
                            damping = 0.5
                            old_msg = self.msg_R_to_i[(R, i)]
                            self.msg_R_to_i[(R, i)] = (damping * old_msg + (1 - damping) * new_msg).detach()

                # variable → factor
                for i in {a for agents in self.region_graph.values() for a in agents}:
                    incidents = [
                        R for R in self.region_graph
                        if (R in q_tables and i in self.region_graph[R])
                    ]
                    if not incidents:
                        continue
                    # Pre-compute Σ m_{S→i} once (linear) instead of |N(i)|² additions
                    total_msg = sum(self.msg_R_to_i[(R_dash, i)] for R_dash in incidents)
                    for R in incidents:
                        msg = total_msg - self.msg_R_to_i[(R, i)]
                        # Optional centring – keeps growth in check
                        msg -= msg.max()
                        self.msg_i_to_R[(i, R)] = msg

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

#######################################
# Agent (owns traces and multipliers)  #
#######################################

class Agent:
    def __init__(
        self, agent_id: str, owned_regions: List[str], traces_refs: Dict[str, Dict[str, torch.Tensor]],
        lambda_dim: int, device: torch.device, alpha: float, alpha_rho: float, lambda_e: float,
        gamma: float, grad_clip: float, optimizers: Dict[str, SGD],
        eta: float = 1e-4,  # Learning rate for multipliers
        constraint_thresholds: List[float] = None,
        constraint_is_local: List[bool] = None
    ):
        self.id = agent_id
        self.owned_regions = owned_regions  # list of region names where this agent is **owner** (first in list)
        self.traces_refs = traces_refs      # region→param_name→trace tensor
        self.device = device
        self.alpha = alpha
        self.alpha_rho = alpha_rho
        self.lambda_e = lambda_e
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.optims = optimizers                         
        self.prev_q: Dict[str, torch.Tensor] = {}

        # Multiplier parameters
        self.eta = eta
        self.constraint_thresholds = constraint_thresholds or []
        self.num_constraints = len(self.constraint_thresholds) 
        self.constraint_is_local = constraint_is_local or [False] * self.num_constraints
        
        # Initialize multipliers
        self.lambda_vec = torch.zeros(lambda_dim, device=device, requires_grad=False)
        if lambda_dim != self.num_constraints:
            raise ValueError(f"lambda_dim ({lambda_dim}) must match num_constraints ({self.num_constraints})")

        
        # Track violations for CVaR-based updates
        self.safety_history = deque(maxlen=100)  # Track safety penalties
        self.border_history = deque(maxlen=100)  # Track border penalties
        self.warmup_steps = 20  # Need some history before using CVaR

        # Initialize traces properly for owned regions
        for region in self.owned_regions:
            if region not in self.traces_refs:
                self.traces_refs[region] = {}

    # ----------------------------------------------
    def td_update_region(self, region: str, critic: RegionCritic, joint_obs: torch.Tensor, joint_action: Tuple[int, ...], reward: float, next_joint_obs: torch.Tensor):
        sizes = critic.sizes
        idx = flat_index(joint_action, sizes)
        # Ensure inputs are finite
        if not (torch.isfinite(joint_obs).all() and torch.isfinite(next_joint_obs).all()):
            print(f"[WARNING] Non-finite observations detected, skipping update")
            return 0.0
        q_vec = critic(joint_obs)
        # Clamp Q-values to prevent explosion
        q_vec = torch.clamp(q_vec, min=-100, max=100)
               
        # Check for Q-value explosion
        if not torch.isfinite(q_vec).all() or q_vec.abs().max() > 1e6:
            print(f"[WARNING] Q-values exploding: {q_vec.abs().max():.2e}, resetting critic")
            # Reset the critic weights
            for m in critic.q_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
            return 0.0
        q_val = q_vec[idx]  # This is still a tensor with gradients

        with torch.no_grad():
            next_q = critic(next_joint_obs).max()
            # Clamp next Q to prevent explosion
            next_q = torch.clamp(next_q, min=-100, max=100)
            # Clip next Q to prevent explosion
            next_q = torch.clamp(next_q, min=-100, max=100)
            # Compute TD target
            td_target = reward - critic.avg_reward.item() + self.gamma * next_q.item()
            # Clip TD target to prevent explosion
            td_target = np.clip(td_target, -5.0, 5.0)
         
        # Compute delta using the actual Q-value
        delta = td_target - q_val.item()

        # Clipping for stability
        delta = float(np.clip(delta, -2.0, 2.0))

       

        # ------------------ true-online differential TD(λ) ------------------
        critic.zero_grad()
        q_val.backward(retain_graph=False, create_graph=False)

        traces = self.traces_refs[region]
        opt = self.optims[region]

        for name, p in critic.named_parameters():
            if p.grad is None:
                continue

            grad = p.grad.detach()
            # Clip gradients before applying to traces
            grad = torch.clamp(grad, min=-1.0, max=1.0)

            if name not in traces:
                traces[name] = torch.zeros_like(p, device=self.device)
            e = traces[name]

            # Online eligibility trace update
            with torch.no_grad():  # Ensure no graph creation
               dot = torch.dot(e.flatten(), grad.flatten()).item()
               # Decay traces more aggressively to prevent accumulation
               e.mul_(self.gamma * self.lambda_e * 0.9).add_(grad)
               # Clip traces to prevent explosion
               e = torch.clamp(e, min=-10.0, max=10.0)
               # Detach and reassign
               e = e.detach_()
               traces[name] = e

            # place final gradient for optimiser
            with torch.no_grad():
               p.grad = (e * delta).detach_()

        # Dual clipping for stability
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1) # actual run with 0.5
        
        # optimiser step
        opt.step()
        opt.zero_grad(set_to_none=True)

        # running average reward baseline
        with torch.no_grad():
            # Slower adaptation to prevent oscillation
            critic.avg_reward.mul_(0.995).add_(0.005 * reward)
            # Bound average reward
            critic.avg_reward.clamp_(min=0.0, max=5.0)



        # store last Q for next step (needed by full TO formulation)
        with torch.no_grad():
           self.prev_q[region] = next_q.item()  # Store as float, not tensor

        return float(delta)
    
    def update_multipliers(self, reward_components: List[float]) -> None:
        """Update multipliers using CVaR for distance-based constraints"""
        # Safety check for reward components
        expected_components = self.num_constraints + 1
        if len(reward_components) < expected_components:
            print(f"Warning: Expected {expected_components} reward components, got {len(reward_components)}")
            # Pad with zeros if needed
            reward_components = reward_components + [0.0] * (expected_components - len(reward_components))
        
        # Track history
        if self.num_constraints >= 1:
            self.safety_history.append(reward_components[1])  # r1: local distance constraint
        if self.num_constraints >= 2:
            self.border_history.append(reward_components[2])  # r2: global defense line
        
        # Use CVaR after warmup period
        if len(self.safety_history) >= self.warmup_steps:
            # Local constraint (index 0)
            if self.num_constraints >= 1:
                # For distance constraints (positive = good), lower percentile = worst outcomes
                safety_cvar = np.percentile(list(self.safety_history), 10)
                # Violation when CVaR < threshold (0.0)
                violation = self.constraint_thresholds[0] - safety_cvar
                self.lambda_vec[0] = torch.clamp(
                    self.lambda_vec[0] + self.eta * violation,
                    min=0.0,
                    max=10.0  
                )
            
            # Global constraint (index 1)
            if self.num_constraints >= 2:
                border_cvar = np.percentile(list(self.border_history), 10)
                violation = self.constraint_thresholds[1] - border_cvar
                self.lambda_vec[1] = torch.clamp(
                    self.lambda_vec[1] + self.eta * violation,
                    min=0.0,
                    max=10.0
                )
        else:
            # During warmup, use simple average
            for j in range(self.num_constraints):
                violation = self.constraint_thresholds[j] - reward_components[j + 1]
                self.lambda_vec[j] = torch.clamp(
                    self.lambda_vec[j] + self.eta * violation,
                    min=0.0,
                    max=10.0
                )
    
    def gossip_update(self, neighbor_lambdas: Dict[str, torch.Tensor], weights: Dict[str, float]):
        """Gossip consensus update following Eq. 15 from the paper"""
        new_lambda = self.lambda_vec.clone()
        
        for j in range(self.num_constraints):
            if not self.constraint_is_local[j]:  # Only gossip global constraints
                # Metropolis weights should sum to 1
                self_weight = weights.get(self.id, 1.0 - sum(
                    w for k, w in weights.items() if k != self.id
                ))
                weighted_sum = self_weight * self.lambda_vec[j]
    
                for neighbor_id, neighbor_lambda in neighbor_lambdas.items():
                    if neighbor_id != self.id:
                        weighted_sum += weights.get(neighbor_id, 0.25) * neighbor_lambda[j]
                new_lambda[j] = weighted_sum
            # Local constraints -> no gossip
            
        self.lambda_vec = new_lambda
    
    def get_augmented_reward(self, reward_components: List[float]) -> float:
        """Compute augmented reward for TD updates"""
        # If all multipliers are zero (unconstrained), just return primary reward
        if self.lambda_vec.norm().item() < 1e-8:
            return reward_components[0]
        
        r_augmented = reward_components[0]  # Primary
        for j in range(self.num_constraints):
            r_augmented += self.lambda_vec[j].item() * reward_components[j + 1]
        return r_augmented
    
