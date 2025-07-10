"""
Three‑Layer Primal–Dual MARL – **true joint‑critic version**
===========================================================
This patch completes the transition to *genuine* multi‑agent regions:

* **Single shared region by default** (all agents) – caller can still pass a
  custom ``region_graph``.
* **One RegionCritic per region** now outputs ``A**|R|`` Q‑values – a full
  joint‑action table in a single forward pass (eliminates all enumeration).
* **Vectorised Max–Sum planner** unchanged – it just receives the reshaped
  tensor.
* **Differential TD(λ) update** operates on the *joint* Q‑value selected by
  encoding the region’s joint action into a flat index.
* **Owners** – the first agent listed in each region updates that region’s
  critic to avoid duplicate gradient steps (simple ownership rule).
* **Tensor casting** – observations are converted to Tensors right after every
  env.reset/step so downstream code never sees NumPy.

Drop the file into any PettingZoo *parallel* env with discrete actions and it
will run coordinated learning.
"""

# ---------------------------------------------------------------
#  Standard libraries
# ---------------------------------------------------------------
import random
import os
import time
import importlib
import glob
from urllib.parse import quote
from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------
#  Third‑party libraries
# ---------------------------------------------------------------
import torch
import torch.nn as nn
from torch.optim import SGD 


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


# ----------------------------------------------------------------------------
# Optional loggers -----------------------------------------------------------
# ----------------------------------------------------------------------------

class _DummyLogger:
    def __init__(self):
        pass
    def log(self, metrics: Dict[str, float], step: int | None = None):
        pass
    def close(self):
        pass

def _setup_logger(logger: str = "wandb",project: str = "three_layer_pd_marl",run_id: str | None = None,**kwargs,):
    """Returns (log_fn, close_fn). If the requested backend is unavailable it
    transparently falls back to a no‑op logger so the training loop never
    crashes because of missing dependencies."""
    logger = logger.lower()

    if logger == "wandb":
        try:
            wandb = importlib.import_module("wandb")
            run = wandb.init(project=project, id=run_id, resume="allow", **kwargs)

            def _log(metrics: Dict[str, float], step: int | None = None):
                wandb.log(metrics, step=step)

            def _close():
                wandb.finish()

            return _log, _close
        except ModuleNotFoundError:
            print("wandb not available – falling back to dummy logger")
            dummy = _DummyLogger()
            return dummy.log, dummy.close

    elif logger in {"tb", "tensorboard"}:
        from torch.utils.tensorboard import SummaryWriter

        log_path = os.path.join("runs", project, run_id or time.strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir=log_path)

        def _log(metrics: Dict[str, float], step: int | None = None):
            for k, v in metrics.items():
                writer.add_scalar(k, v, global_step=step)

        def _close():
            writer.flush(); writer.close()

        return _log, _close

    else:
        print(f"Unknown logger '{logger}' – using dummy logger")
        dummy = _DummyLogger()
        return dummy.log, dummy.close
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
                        augmented = joint_q + sum(inc)
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
            score = sum(
                self.msg_R_to_i[(R, i)]
                for R in self.region_graph
                if (R, i) in self.msg_R_to_i
            )
            a_max = int(torch.argmax(score).item())
            if torch.rand((), device=self.device).item() < self.epsilon:
                actions[i] = int(torch.randint(self.sizes[i], (1,), device=self.device).item())
            else:
                actions[i] = a_max
        return actions

    def to(self, device: torch.device | str):
        self.device = torch.device(device)
        # recreate buffers on new device (simpler & safer than .to in-place)
        self._init_buffers()
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

###############################################################################
# Training                                                                    #
###############################################################################

def train(
    env,
    num_episodes: int = 5_000,
    *,
    lambda_dim: int = 1,
    maxsum_iters: int = 3,
    alpha: float = 1e-3,
    alpha_rho_ratio: float = 0.1,
    lambda_e: float = 0.95,
    region_graph: Dict[str, Iterable[str]] | None = None,
    device: str | torch.device = "cpu",
    logger: str = "wandb",              # "wandb", "tensorboard", or "none"
    project: str = "three_layer_pd_marl",
    seed: int | None = 0,
    grad_clip: float = 10.0,
    checkpoint_interval: int = 500,
    keep_last_ckpts: int = 5,
    run_id: str | None = None,
 ):
    """Run average‑reward TD(λ) on a **parallel** PettingZoo environment."""

    # -------------------------------------------------------
    #  Reproducible seeding 
    # -------------------------------------------------------
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if run_id is None:
        run_id = time.strftime("%Y%m%d-%H%M%S")

    device = torch.device(device)
    p_env = env                          # already parallel

    # ---------- environment seed ----------
    if seed is not None and hasattr(p_env, "seed"):
        try:
            p_env.seed(seed)
        except Exception:
            pass

    # ---------------- logger ----------------
    log_fn, close_logger = _setup_logger(logger, project, run_id=run_id, config={
        "env": getattr(env, "spec", "unknown"),
        "num_episodes": num_episodes,
        "lambda_dim": lambda_dim,
        "maxsum_iters": maxsum_iters,
        "alpha": alpha,
        "alpha_rho_ratio": alpha_rho_ratio,
        "lambda_e": lambda_e
    })

    # ---------- basic sizes ----------
    reset_out = p_env.reset()
    if isinstance(reset_out, tuple):
        obs_raw, _ = reset_out
    else:
        obs_raw = reset_out

    obs_dict = to_tensors(obs_raw, device)
    obs_dim = next(iter(obs_dict.values())).numel()
    assert all(o.numel() == obs_dim for o in obs_dict.values()), \
    "heterogeneous observation sizes are not supported (yet)"

    agents_list = list(obs_dict.keys())
    A_i = {ag: p_env.action_space(ag).n for ag in agents_list}

    # ---------- region graph ----------
    if region_graph is None:
        region_graph = {"global": agents_list}
    region_graph = {R: list(v) for R, v in region_graph.items()}

    # ---------- critics per region ----------
    region_critics: Dict[str, RegionCritic] = {}
    trace_refs: Dict[str, Dict[str, torch.Tensor]] = {}
    optimizers: Dict[str, SGD] = {}   # new

    for R, players in region_graph.items():
        action_sizes_R = [A_i[ag] for ag in players]
        crit = RegionCritic(obs_dim, lambda_dim, action_sizes_R).to(device)
        region_critics[R] = crit
        trace_refs[R] = {name: torch.zeros_like(p, device=device)
                        for name, p in crit.named_parameters() if p.requires_grad}
        optimizers[R] = SGD(crit.parameters(), lr=alpha)

    # ---------- agents ----------
    alpha_rho = alpha * alpha_rho_ratio
    agent_objs: Dict[str, Agent] = {}
    for ag in agents_list:
        owned = [R for R, players in region_graph.items() if players[0] == ag]
        agent_objs[ag] = Agent(ag, owned, trace_refs, lambda_dim, device, alpha, alpha_rho, lambda_e, gamma=1.0, grad_clip=grad_clip, optimizers=optimizers)

    # ---------- pre-allocated joint-obs buffer ----------
    joint_feat_dim = obs_dim + lambda_dim
    agent_feat_buf = torch.empty(len(agents_list), joint_feat_dim, device=device)
    # ---------- planner ----------
    planner = MaxSumPlanner(region_graph, A_i, n_iters=maxsum_iters, device=device)
    # ---------- unique checkpoint dir  ----------
    ckpt_dir = os.path.join("checkpoints", project, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---------- bookkeeping ----------
    episode_returns, episode_lengths = [], []
    td_errors: deque[float] = deque(maxlen=5_000)

    # ================== main loop ==================
    for ep in range(num_episodes):
        obs_raw, _ = p_env.reset()
        obs_dict = to_tensors(obs_raw, device)
        planner.reset_messages()
        # -------- eligibility-trace reset (true-online TD(λ)) --------
        for traces in trace_refs.values():
            for e in traces.values():
                e.zero_()
        ep_return = 0.0
        ep_steps = 0

        while p_env.agents:
            live = p_env.agents

            # ----- joint Q tables -----
            q_tables: Dict[str, torch.Tensor] = {}
            joint_obs_cache: Dict[str, torch.Tensor] = {}
            for R, players in region_graph.items():
                if not all(ag in live for ag in players):
                    continue
                # -------------------------------------------------
                # Fast in-place build of the joint feature tensor
                # -------------------------------------------------
                for pos, ag in enumerate(players):
                    agent_feat_buf[pos, :obs_dim].copy_(obs_dict[ag])
                    agent_feat_buf[pos, obs_dim:].copy_(agent_objs[ag].lambda_vec)
                joint_obs = agent_feat_buf[:len(players)].flatten()
                joint_obs_cache[R] = joint_obs.clone().detach()
                sizes_R = region_critics[R].sizes
                q_tables[R] = region_critics[R](joint_obs).view(tuple(sizes_R))

            # ----- planner & env step -----
            planner.step(q_tables)
            act_full = planner.select_actions()
            act_live = {ag: act_full[ag] for ag in live}

            next_obs_raw, rewards, terms, truncs, _ = p_env.step(act_live)

            # ----- bookkeeping -----
            ep_return += sum(rewards.values()) / max(len(rewards), 1)
            ep_steps += 1
            next_obs = to_tensors(next_obs_raw, device)

            # ----- TD updates by owners -----
            for R, players in region_graph.items():
                if not all(ag in act_full for ag in players):
                    continue
                owner = players[0]
                joint_act = tuple(act_full[ag] for ag in players)
                r = sum(rewards.get(ag, 0.0) for ag in players) / len(players)
                for pos, ag in enumerate(players):
                    agent_feat_buf[pos, :obs_dim].copy_(next_obs.get(ag, obs_dict[ag]))
                    agent_feat_buf[pos, obs_dim:].copy_(agent_objs[ag].lambda_vec)
                next_joint_obs = agent_feat_buf[: len(players)].flatten()
                delta = agent_objs[owner].td_update_region(
                    R, region_critics[R], joint_obs_cache[R], joint_act, r, next_joint_obs
                )
                td_errors.append(float(delta))

            obs_dict = next_obs

        # ----- end of episode -----
        episode_returns.append(ep_return)
        episode_lengths.append(ep_steps)

        if (ep + 1) % checkpoint_interval == 0:
            for name_R, critic in region_critics.items():
                fname = f"{quote(name_R, safe='')}_critic.pt"
                torch.save(critic.state_dict(), os.path.join(ckpt_dir, fname))

            # keep only the latest *keep_last_ckpts* checkpoint sets
            ckpts = sorted(
                glob.glob(os.path.join(ckpt_dir, "*_critic.pt")),
                key=os.path.getmtime,
            )
            num_to_keep = len(region_critics) * keep_last_ckpts
            if len(ckpts) > num_to_keep:
                for old in ckpts[:-num_to_keep]:
                    os.remove(old)

        log_fn({"episode/return": ep_return, "episode/length": ep_steps}, step=ep + 1)
        if (ep + 1) % 100 == 0:
            mean100 = float(np.mean(episode_returns[-100:]))
            print(f"Ep {ep + 1:>5} | R̄₁₀₀ {mean100:8.2f} | len {np.mean(episode_lengths[-100:]):5.1f}")
            log_fn({"stats/mean_return_100": mean100}, step=ep + 1)

    close_logger()

###############################################################################
# Evaluation (completed)                                                      #
###############################################################################

@torch.no_grad()
def evaluate_policy(
    env,
    region_graph: Dict[str, Iterable[str]] | None = None,
    episodes: int = 5,
    *,
    lambda_dim: int = 1,
    ckpt_dir: str = "checkpoints",
    render: bool = False,
    device: str | torch.device = "cpu",) -> None:
    device = torch.device(device)
    p_env = env                          # already parallel
    if region_graph is None:
        region_graph = {"global": list(p_env.possible_agents)}

    obs_dim = int(np.prod(p_env.observation_space(p_env.possible_agents[0]).shape))

    critics = {
        R: RegionCritic(
                obs_dim, lambda_dim,
                [p_env.action_space(a).n for a in agents]
            ).to(device)
        for R, agents in region_graph.items()
    }
    action_sizes_eval = {a: p_env.action_space(a).n for a in p_env.possible_agents}
    planner = MaxSumPlanner(region_graph, action_sizes_eval, n_iters=3, epsilon=0.0, device=device)

    for R, c in critics.items():
        fname = f"{quote(R, safe='')}_critic.pt"
        path  = os.path.join(ckpt_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected critic checkpoint at {path}. "
                                    "Run training or adjust --ckpt_dir.")
        c.load_state_dict(torch.load(path, map_location=device))

    returns = []
    for _ in range(episodes):
        obs_raw, _ = p_env.reset()
        obs = to_tensors(obs_raw, device)
        # pre-allocated buffer to avoid per-step concat
        agent_feat_buf = torch.empty(len(p_env.possible_agents), obs_dim + lambda_dim, device=device)
        planner.reset_messages()
        ep_ret = 0.0
        while p_env.agents:
            q_tables = {}
            for R, players in region_graph.items():
                if not all(a in p_env.agents for a in players):
                    continue
                for pos, ag in enumerate(players):
                    agent_feat_buf[pos, :obs_dim].copy_(obs[ag])
                    agent_feat_buf[pos, obs_dim:].zero_()           # λ = 0 during eval
                feats = agent_feat_buf[:len(players)].flatten()
                q_tables[R] = critics[R](feats).view(tuple(critics[R].sizes))
            planner.step(q_tables)
            act = planner.select_actions()
            next_obs_raw, rews, terms, truncs, _ = p_env.step(act)
            obs = to_tensors(next_obs_raw, device)
            ep_ret += sum(rews.values()) / max(len(rews), 1)
            if render:
                p_env.render()
        returns.append(ep_ret)
    print(f"Evaluation over {episodes} episodes: mean {np.mean(returns):.2f} ± {np.std(returns):.2f}")

###############################################################################
# Entry‑point                                                                 #
###############################################################################

if __name__ == "__main__":
    from pettingzoo.butterfly import knights_archers_zombies_v10 as kaz

    RUN_ID = time.strftime("%Y%m%d-%H%M%S")
    CKPT_DIR = os.path.join("checkpoints", "three_layer_pd_marl", RUN_ID)

    # Knights–Archers–Zombies is parallel by default and uses a Discrete(9) action space.
    env = kaz.parallel_env(
        render_mode="human",
        vector_state=True,
        use_typemasks=True,
    )

    train(
        env,
        num_episodes=10_000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger="wandb",
        project="three_layer_pd_marl",
        seed=0,
        grad_clip=10.0,
        run_id=RUN_ID,
    )

    env = kaz.parallel_env(
        render_mode="human",
        vector_state=True,
        use_typemasks=True,
    )

    evaluate_policy(
        env,
        episodes=3,
        lambda_dim=1,
        ckpt_dir=CKPT_DIR,
    )

    env.close()