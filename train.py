"""
train.py
=========
Experiment driver: environment setup, training loop, evaluation routine and
checkpoint I/O.

Responsibilities
----------------
1. **Create** a PettingZoo environment (or any compatible multi-agent env).  
2. **Instantiate** an `Agent` from `core.py`, wiring in hyper-parameters,
   region graph, and logging callables.  
3. **Drive** the main RL loop: collect joint actions via `Agent.act()`,
   perform TD-λ updates, clip gradients, and periodically call `log_fn`.  
4. **Checkpoint** the entire `Agent` (network weights + optimiser state) every
   *N* steps, keeping the last *k* snapshots.  
5. **Evaluate** the frozen policy in a deterministic test loop for reporting.

File interactions
-----------------
* Imports `to_tensors` …→ pre-process env observations.  
* Imports `Agent`, `RegionCritic`, `MaxSumPlanner` …→ core learning logic.  
* Calls `logger.setup_logger` once, then passes the resulting `log_fn` into
  `Agent` so lower-level modules never need to know about the backend.

Entry points & CLI
------------------
The script is designed to be runnable as

    python train.py --episodes 1_000_000 --logger wandb --project pd_marl

but its `train()` / `evaluate_policy()` functions are also import-friendly, so
you can embed them in a Jupyter notebook or a larger experiment manager.

Performance tips
----------------
* Use `device=\"cuda\"` when available; the helper `to_tensors` auto-casts to
  float32.  
* `grad_clip` default (10.0) prevents exploding updates in early stages.  
* `seed` can be `None` for true stochasticity; specifying it makes results
  deterministic across runs.

"""


# ──────────────────────────────────────────────────────────────
# Standard lib
# ──────────────────────────────────────────────────────────────
import random
import os
import time
from collections import deque
from typing import Dict, Iterable

# ──────────────────────────────────────────────────────────────
# Third-party
# ──────────────────────────────────────────────────────────────
import numpy as np
import torch
from torch.optim import SGD

# ──────────────────────────────────────────────────────────────
# Project modules
# ──────────────────────────────────────────────────────────────
from utils import to_tensors                    # tensor helpers
from core import Agent, RegionCritic, MaxSumPlanner
from logger import setup_logger                 # (log_fn, close_fn)


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
    log_fn, close_logger = setup_logger(logger, project, run_id=run_id, config={
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