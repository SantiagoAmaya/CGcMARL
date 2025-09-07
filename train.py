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
import random
from typing import Dict, Iterable, Any
import glob
from urllib.parse import quote
import traceback

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
from kaz_constraints import KAZRewardDecomposer, CommunicationGraph  # Your new modules


def save_full_checkpoint(
    filepath: str,
    episode: int,
    region_critics: Dict[str, RegionCritic],
    optimizers: Dict[str, torch.optim.SGD],
    schedulers: Dict[str, Any],
    agent_objs: Dict[str, Agent],
    trace_refs: Dict[str, Dict[str, torch.Tensor]],
    planner: MaxSumPlanner
):
    """Save complete training state for resumption"""
    checkpoint = {
        'episode': episode,
        'region_critics': {
            R: critic.state_dict() 
            for R, critic in region_critics.items()
        },
        'optimizers': {
            R: opt.state_dict() 
            for R, opt in optimizers.items()
        },
        'schedulers': {
            R: sched.state_dict()
            for R, sched in schedulers.items()
        },
        'agent_states': {
            ag: {
                'lambda_vec': agent.lambda_vec.cpu(),
                'prev_q': {
                    k: torch.tensor(v) if not torch.is_tensor(v) else v.cpu() 
                    for k, v in agent.prev_q.items()
                },
                'constraint_violations': list(agent.constraint_violations),  # Convert deque to list
                'eta': agent.eta  # Save current learning rate
            }
            for ag, agent in agent_objs.items()
        },
        'traces': {
            R: {name: trace.cpu() for name, trace in traces.items()}
            for R, traces in trace_refs.items()
        },
        'planner_epsilon': planner.epsilon
    }
    torch.save(checkpoint, filepath)
    print(f"Saved full checkpoint at episode {episode} to {filepath}")


###############################################################################
# Training                                                                    #
###############################################################################

def train(
    env,
    num_episodes: int = 5_000,
    constraint_thresholds: list[float] = None,
    eta: float = None, 
    eta_decay: float = 0.999,
    multiplier_update_freq: str = "step",
    communication_topology: str = "fully_connected",
    *,
    min_agent_zombie_dist: float = 0.15,  # Configurable safety distance
    min_zombie_border_dist: float = 0.2,   # Configurable border distance
    safety_penalty: float = -0.1,          # Penalty magnitude
    border_penalty: float = -0.2,          # Penalty magnitude
    maxsum_iters: int = 3,
    resume_from: str = None,
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
    memory_limit_mb: float = None,  # Add configurable memory limit
    enable_aggressive_gc: bool = False,  # Make GC optional
    gc_frequency: int = 100,  # Configurable GC frequency
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

    if constraint_thresholds is None:
        constraint_thresholds = [0.5, 0.5]  # No constraints

    # Ensure two-timescale separation: η_t = o(α_t)
    if eta is None:
        eta = alpha * 0.01  # Multiplier learning should be slower
    # Track effective eta for decay
    current_eta = eta
  
    num_constraints = len(constraint_thresholds)
    lambda_dim = num_constraints 
    constraint_is_local = [True, False]

    # ---------------- logger ----------------
    log_fn, close_logger = setup_logger(
    logger,                    # backend (positional)
    project=project,           # keyword-only argument
    run_id=run_id,            # keyword-only argument
    config={                  # wandb_kwargs
        "env": getattr(env, "spec", "unknown"),
        "num_episodes": num_episodes,
        "lambda_dim": lambda_dim,
        "maxsum_iters": maxsum_iters,
        "alpha": alpha,
        "alpha_rho_ratio": alpha_rho_ratio,
        "lambda_e": lambda_e
    }
)

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
    decomposer = KAZRewardDecomposer(
        danger_distance=0.02,       # Only death is penalized
        border_danger_rel_y=0.1   # Bottom 10% is critical
    )
    comm_graph = CommunicationGraph(agents_list, communication_topology)

    # Validate communication graph connectivity
    for ag in agents_list:
        neighbors = comm_graph.get_neighbors(ag)
        weights = comm_graph.get_weights(ag)
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Gossip weights for agent {ag} don't sum to 1: {weight_sum}")
    
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

    schedulers = {
        R: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_episodes+10, eta_min=alpha*0.01)
        for R, opt in optimizers.items()
    }
    # ---------- agents ----------
    alpha_rho = alpha * alpha_rho_ratio
    agent_objs: Dict[str, Agent] = {}
    for ag in agents_list:
        owned = [R for R, players in region_graph.items() if players[0] == ag]
        agent_objs[ag] = Agent(
            ag, owned, trace_refs, lambda_dim, device, alpha, alpha_rho, lambda_e, 
            gamma=1.0, grad_clip=grad_clip, optimizers=optimizers,
            eta=eta, constraint_thresholds=constraint_thresholds,
            constraint_is_local=constraint_is_local
        )
         # Add running average tracking for constraints
        agent_objs[ag].recent_constraints = deque(maxlen=10)

   # ---------- pre-allocated buffers for efficiency ----------
    max_region_size = max(len(players) for players in region_graph.values())
    agent_feat_buf = torch.zeros(max_region_size, obs_dim + lambda_dim, device=device)
    
    # ---------- planner ----------
    planner = MaxSumPlanner(region_graph, A_i, n_iters=maxsum_iters, device=device)
    # ---------- unique checkpoint dir  ----------
    ckpt_dir = os.path.join("checkpoints", project, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---------- bookkeeping ----------
    # Limit history to prevent unbounded growth
    episode_returns = deque(maxlen=1000)
    episode_lengths = deque(maxlen=1000)
    td_errors: deque[float] = deque(maxlen=5_000)

    start_episode = 0
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        
        # Restore critics
        for R, state_dict in checkpoint['region_critics'].items():
            region_critics[R].load_state_dict(state_dict)
        
        # Restore optimizers
        for R, state_dict in checkpoint['optimizers'].items():
            optimizers[R].load_state_dict(state_dict)

        # Restore schedulers
        if 'schedulers' in checkpoint:
            for R, state_dict in checkpoint['schedulers'].items():
                schedulers[R].load_state_dict(state_dict)
        
        
        # Restore agent states
        for ag, state in checkpoint['agent_states'].items():
            agent_objs[ag].lambda_vec = state['lambda_vec'].to(device)
            agent_objs[ag].prev_q = {
                k: v.item() if torch.is_tensor(v) else v for k, v in state['prev_q'].items()
            }
            # Restore constraint violations history
            if 'constraint_violations' in state:
                # Convert list back to deque
                agent_objs[ag].constraint_violations = deque(
                    state['constraint_violations'], 
                    maxlen=100
                )
            else:
                agent_objs[ag].constraint_violations = deque(maxlen=100)
         

        
        
        # Restore traces
        for R, traces in checkpoint['traces'].items():
            for name, trace in traces.items():
                trace_refs[R][name] = trace.to(device)
        
        start_episode = checkpoint['episode']
        print(f"Resumed from episode {start_episode}")

    # ================== main loop ==================
    for ep in range(start_episode, num_episodes):
        # Safety check at final episode
        if ep == num_episodes - 1:
            print(f"Final episode {ep+1} starting...")
            print(f"Current learning rates:")
            for R, opt in optimizers.items():
                for param_group in opt.param_groups:
                    print(f"  Region {R}: lr = {param_group['lr']}")
            print(f"Current avg_rewards:")
            for R, critic in region_critics.items():
                print(f"  Region {R}: avg_reward = {critic.avg_reward.item()}")
        # CONFIGURABLE MEMORY MANAGEMENT
        if enable_aggressive_gc and ep % gc_frequency == 0 and ep > 0:
            try:
                import gc
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"GC failed: {e}")
            
            # Check memory and warn
            if memory_limit_mb is not None:
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    if ep % 100 == 0:  # Log less frequently
                        print(f"Episode {ep}: Memory = {memory_mb:.1f} MB")
                    
                    if memory_mb > memory_limit_mb:
                        print(f"WARNING: Memory ({memory_mb:.1f} MB) exceeds limit ({memory_limit_mb} MB)")
                        # Only reset traces as last resort
                        print("Detaching traces to free memory...")
                        for traces in trace_refs.values():
                            for name, trace in traces.items():
                                traces[name] = trace.detach()
                except ImportError:
                    pass  # psutil not available

                        
           
        # Exploration decay - reduce exploration as training progresses 
        current_epsilon = max(0.05, 0.3 * (1 - ep / num_episodes))  # Linear decay to 0.1
        planner.epsilon = current_epsilon
        
        obs_raw, _ = p_env.reset()
        obs_dict = to_tensors(obs_raw, device)
        planner.reset_messages()

        # -------- eligibility-trace reset (true-online TD(λ)) --------
        for traces in trace_refs.values():
            for e in traces.values():
                e.zero_()

        # Only reset traces if memory is critically high
        if ep % 500 == 0 and ep > 0:
            try:
                import psutil
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                if memory_mb > 8000:  # Only if > 8GB
                    print(f"Episode {ep}: Performing full trace reset...")
                    for region, traces in trace_refs.items():
                        for name in list(traces.keys()):
                            del traces[name]
                        # Recreate traces
                        for name, p in region_critics[region].named_parameters():
                            if p.requires_grad:
                                traces[name] = torch.zeros_like(p, device=device).detach()
                else:
                   # Just detach without resetting
                   for traces in trace_refs.values():
                       for name, e in traces.items():
                           traces[name] = e.detach()
            except ImportError:
               pass

        ep_return = 0.0
        ep_steps = 0

        # Track per-step satisfaction instead of cumulative
        episode_stats = {
            ag: {
                'total_reward': 0.0,
                'steps': 0,
                'safety_satisfied': 0,  # Count of safe steps
                'border_satisfied': 0,   # Count of border-defended steps
            } for ag in agents_list
        }

        while p_env.agents:
            # Debug observation structure (first episode, first few steps)
            if ep == 0 and ep_steps < 3:
                print(f"\n=== Episode {ep}, Step {ep_steps} ===")
                for ag, obs in obs_dict.items():
                    print(f"\n{ag} observation debug:")
                    print(f"  Shape: {obs.shape}")
                    if obs.numel() >= 11:
                        # Reshape to see structure
                        obs_reshaped = obs.view(-1, 11) if obs.numel() % 11 == 0 else obs.view(-1, 5)
                        print(f"  First entity (current agent): {obs_reshaped[0].tolist()}")
                        # Check for zombies
                        for i in range(1, min(5, obs_reshaped.shape[0])):
                            if obs_reshaped.shape[1] == 11:  # Has typemask
                                if obs_reshaped[i, 0] > 0.5:  # Is zombie
                                    print(f"  Zombie at row {i}: typemask={obs_reshaped[i, :6].tolist()}, dist={obs_reshaped[i, 6]:.3f}, rel_pos=({obs_reshaped[i, 7]:.3f}, {obs_reshaped[i, 8]:.3f})")
            
            live = p_env.agents

            # ----- joint Q tables -----
            with torch.no_grad():
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
                    joint_obs_cache[R] = joint_obs.detach().clone()
                    sizes_R = region_critics[R].sizes
                    with torch.no_grad():
                       q_tables[R] = region_critics[R](joint_obs).view(tuple(sizes_R)).detach()


            # ----- planner & env step -----
            planner.step(q_tables)
            act_full = planner.select_actions()
            act_live = {ag: act_full[ag] for ag in live}

            # Environment step
            next_obs_raw, rewards, terms, truncs, _ = p_env.step(act_live)
            next_obs = to_tensors(next_obs_raw, device)
            reward_components = decomposer.decompose_rewards(next_obs, rewards)

            # Track constraint satisfaction properly
            for ag in live:
                components = reward_components.get(ag, [0.0, 1.0, 1.0])
                episode_stats[ag]['total_reward'] += components[0]  # Primary reward
                episode_stats[ag]['steps'] += 1
                episode_stats[ag]['safety_satisfied'] += (components[1] >= constraint_thresholds[0])
                episode_stats[ag]['border_satisfied'] += (components[2] >= constraint_thresholds[1])


            # Update multipliers (if per-step)
            if multiplier_update_freq == "step":
                warmup_episodes = 100
                if ep >= warmup_episodes:  
                    # First update multipliers
                    for ag in live:
                        # Track recent constraint satisfaction
                        if not hasattr(agent_objs[ag], 'recent_constraints'):
                            agent_objs[ag].recent_constraints = deque(maxlen=10)
                        
                        agent_objs[ag].recent_constraints.append(reward_components[ag])
                        
                        # Use average of recent steps for more stable updates
                        if len(agent_objs[ag].recent_constraints) > 0:
                            avg_components = [
                                np.mean([c[i] for c in agent_objs[ag].recent_constraints])
                                for i in range(num_constraints + 1)
                            ]
                        else:
                            avg_components = reward_components[ag]

                        # Use decaying learning rate for multipliers
                        agent_objs[ag].eta = current_eta
                        agent_objs[ag].update_multipliers(avg_components)
                    
                    # Then gossip (asynchronous simulation)
                    # Randomize gossip order to simulate asynchrony
                    gossip_order = list(live)
                    random.shuffle(gossip_order)
                    
                    # Create snapshot of current lambdas before gossip
                    lambda_snapshot = {
                        ag: agent_objs[ag].lambda_vec.clone() 
                        for ag in live
                    }
                    
                    for ag in gossip_order:
                        neighbors = comm_graph.get_neighbors(ag)
                        neighbor_lambdas = {}
                        for n in neighbors:
                            if n in lambda_snapshot:
                                neighbor_lambdas[n] = lambda_snapshot[n]
                        weights = comm_graph.get_weights(ag)
                        agent_objs[ag].gossip_update(neighbor_lambdas, weights)
                else:
                    # During warmup
                    for ag in live:
                        agent_objs[ag].lambda_vec.zero_()
                

            # ----- bookkeeping -----
            ep_return += sum(rewards.values()) / max(len(rewards), 1)
            ep_steps += 1
            next_obs = to_tensors(next_obs_raw, device)

            # ----- TD updates by owners -----
            for R, players in region_graph.items():
                # Check if this region was processed in the planning phase
                if R not in joint_obs_cache:
                    continue  # Skip regions that weren't cached

                # Original owner (first agent in region definition)
                original_owner = players[0]
                
                # Check if original owner is still alive
                if original_owner in rewards:
                    owner = original_owner
                else:
                    # Find backup owner (maintain consistency)
                    # Use deterministic selection based on agent ID to ensure consistency
                    alive_players = [ag for ag in players if ag in rewards]
                    if not alive_players:
                        continue  # No alive agents to perform update
                    
                    # Sort alive players to ensure deterministic selection
                    alive_players_sorted = sorted(alive_players)
                    owner = alive_players_sorted[0]
                    
                    # Log owner change for debugging
                    if ep % 100 == 0:  # Don't spam logs
                        print(f"Region {R}: Owner changed from {original_owner} to {owner}")
 

                # Ensure temporary owner has necessary structures
                if owner not in agent_objs:
                    continue  # Skip if owner doesn't exist
    
                # Also check if all agents in this region took actions
                if not all(ag in act_full for ag in players):
                    continue

                # Compute augmented reward for region
                r_components = [0.0] * (num_constraints + 1)
                for i in range(num_constraints + 1):
                    # Average the reward components across agents in region
                    r_components[i] = sum(
                        reward_components.get(ag, [0.0] * (num_constraints + 1))[i] 
                        for ag in players
                    ) / len(players)
                
                # Get augmented reward using owner's multipliers
                r_augmented = agent_objs[owner].get_augmented_reward(r_components)

                # Debug: Check reward magnitudes
                if ep % 100 == 0 and ep_steps == 1:  # Log once per 100 episodes
                    print(f"Episode {ep}, Region {R}:")
                    print(f"  Primary reward component: {r_components[0]:.4f}")
                    print(f"  Safety component: {r_components[1]:.4f}")
                    print(f"  Border component: {r_components[2]:.4f}")
                    print(f"  Augmented reward: {r_augmented:.4f}")
                    print(f"  Current avg_reward: {region_critics[R].avg_reward.item():.4f}")
                 
    
                
                joint_act = tuple(act_full[ag] for ag in players)
                
                # Build next joint observation
                for pos, ag in enumerate(players):
                    agent_feat_buf[pos, :obs_dim].copy_(next_obs.get(ag, obs_dict[ag]))
                    agent_feat_buf[pos, obs_dim:].copy_(agent_objs[ag].lambda_vec)
                next_joint_obs = agent_feat_buf[:len(players)].flatten()
                
                # Safe TD update with error handling
                delta = safe_train_step(
                    agent_objs[owner],
                    R, region_critics[R], joint_obs_cache[R], joint_act, 
                    r_augmented,  # Use augmented instead of raw reward
                    next_joint_obs
                )

                if not np.isfinite(delta):
                    print(f"NaN/Inf detected at episode {ep}, step {ep_steps}")
                    print(f"  Region: {R}")
                    print(f"  Reward: {r_augmented}")
                    print(f"  Avg reward: {region_critics[R].avg_reward.item()}")
                    print(f"  Joint obs: {joint_obs_cache[R][:5]}...")  # First 5 values
                    
                    # Check critic weights
                    for name, param in region_critics[R].named_parameters():
                        if torch.isnan(param).any():
                            print(f"  NaN in parameter: {name}")
                    
                    # Save emergency checkpoint
                    emergency_path = os.path.join(ckpt_dir, f"nan_debug_ep{ep}.pt")
                    save_full_checkpoint(...)
                    print(f"Saved debug checkpoint to {emergency_path}")
                    sys.exit(1)
                
                td_errors.append(float(delta))

            # Clear caches to free memory
            joint_obs_cache.clear()
            q_tables.clear()

            obs_dict = next_obs

        # Update multipliers (if per-episode)
        if multiplier_update_freq == "episode":
            # Apply decay to multiplier learning rate
            current_eta = max(current_eta * eta_decay, eta * 0.01)  # Floor at 1% of initial
            
            normalized_rewards = {}
            for ag in agents_list:
                stats = episode_stats[ag]
                steps = max(stats['steps'], 1)
                # Average satisfaction rates
                safety_rate = stats['safety_satisfied'] / steps
                border_rate = stats['border_satisfied'] / steps
                normalized_rewards[ag] = [
                    stats['total_reward'] / steps,  # Average primary reward
                    safety_rate,
                    border_rate
                ]
            
            warmup_episodes = 100
            if ep >= warmup_episodes:  # Only update after warmup
                for ag in agents_list:
                    agent_objs[ag].update_multipliers(normalized_rewards[ag])
                
                # Gossip
                for ag in agents_list:
                    neighbors = comm_graph.get_neighbors(ag)
                    neighbor_lambdas = {
                        n: agent_objs[n].lambda_vec 
                        for n in neighbors
                    }
                    weights = comm_graph.get_weights(ag)
                    agent_objs[ag].gossip_update(neighbor_lambdas, weights)
            else:
                # During warmup, keep multipliers at zero
                for ag in agents_list:
                    agent_objs[ag].lambda_vec.zero_()
        
        # Log constraint metrics
        # Calculate average constraint satisfaction across all agents
        avg_safety_rate = np.mean([
            episode_stats[ag]['safety_satisfied'] / max(episode_stats[ag]['steps'], 1)
            for ag in agents_list
        ])
        avg_border_rate = np.mean([
            episode_stats[ag]['border_satisfied'] / max(episode_stats[ag]['steps'], 1)
            for ag in agents_list
        ])
        
        constraints_satisfied = (
            avg_safety_rate >= constraint_thresholds[0] and
            avg_border_rate >= constraint_thresholds[1]
        )

        episode_returns.append(ep_return)
        episode_lengths.append(ep_steps)

        # Step learning rate schedulers (unless final episode)
        if ep < num_episodes - 1:
            for scheduler in schedulers.values():
                scheduler.step()
        
        # Additional decay for stability
        if ep > 2000 and ep % 500 == 0:
            for opt in optimizers.values():
                for param_group in opt.param_groups:
                    param_group['lr'] *= 0.9
            print(f"Reduced learning rate at episode {ep}")


        if (ep + 1) % checkpoint_interval == 0:
            # Save individual critic files (for backward compatibility)
            for name_R, critic in region_critics.items():
                fname = f"{quote(name_R, safe='')}_critic.pt"
                torch.save(critic.state_dict(), os.path.join(ckpt_dir, fname))
            
            # SAVE FULL CHECKPOINT
            full_ckpt_path = os.path.join(ckpt_dir, f"full_checkpoint_ep{ep+1}.pt")
            save_full_checkpoint(
                full_ckpt_path, ep + 1, region_critics, optimizers, 
                schedulers, agent_objs, trace_refs, planner
            )
            
            # Keep only the latest keep_last_ckpts full checkpoints
            full_ckpts = sorted(
                glob.glob(os.path.join(ckpt_dir, "full_checkpoint_*.pt")),
                key=os.path.getmtime
            )
            if len(full_ckpts) > keep_last_ckpts:
                for old in full_ckpts[:-keep_last_ckpts]:
                    os.remove(old)
                    print(f"Removed old checkpoint: {old}")

        log_fn({"episode/return": ep_return, "episode/length": ep_steps}, step=ep + 1)
        
        if (ep + 1) % 50 == 0:
            metrics = {
                "stats/mean_return_100": float(np.mean(list(episode_returns)[-100:])),
                "stats/mean_length_100": float(np.mean(list(episode_lengths)[-100:])),
                "constraints/satisfied": float(constraints_satisfied),
                "constraints/avg_safety_rate": avg_safety_rate,
                "constraints/avg_border_rate": avg_border_rate,
            }
            
            # Add TD error statistics
            if td_errors:
                metrics.update({
                    "td/error_mean": float(np.mean(td_errors)),
                    "td/error_std": float(np.std(td_errors)),
                    "td/error_min": float(np.min(td_errors)),
                    "td/error_max": float(np.max(td_errors)),
                })
            
            # Add per-region average rewards
            for R, critic in region_critics.items():
                metrics[f"avg_reward/{R}"] = float(critic.avg_reward.item())
            
            # Add lambda vector statistics
            lambda_norms = [agent.lambda_vec.norm().item() for agent in agent_objs.values()]
            metrics["lambda/mean_norm"] = float(np.mean(lambda_norms))

            # Log multiplier evolution
            for ag in agents_list:  # Remove the [:2] limitation
                lambda_vals = agent_objs[ag].lambda_vec
                for j in range(num_constraints):
                    constraint_type = "local" if constraint_is_local[j] else "global"
                    metrics[f"lambda/{ag}_constraint{j}_{constraint_type}"] = lambda_vals[j].item()
            
            # Log constraint violation rates
            # Calculate violation rates (1 - satisfaction rate)
            metrics["constraints/safety_violation_rate"] = 1.0 - avg_safety_rate
            metrics["constraints/border_violation_rate"] = 1.0 - avg_border_rate
            
            log_fn(metrics, step=ep + 1)

            # Force wandb to sync and clear buffer
            if logger == "wandb":
               try:
                   import wandb
                   if wandb.run is not None:
                       wandb.log({}, commit=True)  # Force commit
               except:
                   pass

    close_logger()

def safe_train_step(agent, region, critic, joint_obs, joint_action, reward, next_joint_obs):
    """Wrapper with error handling for TD updates"""
    try:
        delta = agent.td_update_region(
            region, critic, joint_obs, joint_action, reward, next_joint_obs
        )
        return delta
    except Exception as e:
        print(f"Warning: TD update failed for region {region}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        # Log error but continue training
        return 0.0  # Return zero TD error

###############################################################################
# Evaluation (completed)                                                      #
###############################################################################

@torch.no_grad()
def evaluate_policy(
    env,
    region_graph: Dict[str, Iterable[str]] | None = None,
    episodes: int = 5,
    *,
    lambda_dim: int = 2,
    ckpt_dir: str = "checkpoints",
    render: bool = False,
    device: str | torch.device = "cpu",) -> None:
    device = torch.device(device)
    p_env = env                          # already parallel
    if region_graph is None:
        region_graph = {"global": list(p_env.possible_agents)}

    # Get observation dimension from actual observation (same as training)
    reset_out = p_env.reset()
    if isinstance(reset_out, tuple):
        obs_raw, _ = reset_out
    else:
        obs_raw = reset_out
    
    obs_dict = to_tensors(obs_raw, device)
    obs_dim = next(iter(obs_dict.values())).numel()
    
    # Verify observation dimensions match across agents
    assert all(o.numel() == obs_dim for o in obs_dict.values()), \
        "Heterogeneous observation sizes detected"
    
    # Create critics with proper variable scoping
    critics = {}
    for R, agents in region_graph.items():
        action_sizes = [p_env.action_space(a).n for a in agents]
        critics[R] = RegionCritic(obs_dim, lambda_dim, action_sizes).to(device)
    

    action_sizes_eval = {a: p_env.action_space(a).n for a in p_env.possible_agents}
    planner = MaxSumPlanner(region_graph, action_sizes_eval, n_iters=3, epsilon=0.0, device=device)

    for R, c in critics.items():
        fname = f"{quote(R, safe='')}_critic.pt"
        path  = os.path.join(ckpt_dir, fname)
        if not os.path.exists(path):
            # Try loading from full checkpoint instead
            # Find the latest full checkpoint
            full_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "full_checkpoint_*.pt")))
            if full_ckpts:
                full_ckpt_path = full_ckpts[-1]  # Use the latest
            else:
                raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
            
            if os.path.exists(full_ckpt_path):
                checkpoint = torch.load(full_ckpt_path, map_location=device)
                c.load_state_dict(checkpoint['region_critics'][R])
                print(f"Loaded critic for region {R} from full checkpoint")
            else:
                raise FileNotFoundError(f"No checkpoint found for region {R}")
        else:
            c.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded critic for region {R} from {path}")

    # Debug: Print dimensions to verify
    print(f"Evaluation setup:")
    print(f"  Observation dimension: {obs_dim}")
    print(f"  Lambda dimension: {lambda_dim}")
    print(f"  Region graph: {list(region_graph.keys())}")
    for R, agents in region_graph.items():
        print(f"  Region {R}: {len(agents)} agents, input_dim={(len(agents) * (obs_dim + lambda_dim))}")


    returns = []
    for _ in range(episodes):
        reset_out = p_env.reset()
        if isinstance(reset_out, tuple):
            obs_raw, _ = reset_out
        else:
            obs_raw = reset_out
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
            # Only act for agents that are still alive
            act = {ag: act[ag] for ag in p_env.agents if ag in act}
            next_obs_raw, rews, terms, truncs, _ = p_env.step(act)
            obs = to_tensors(next_obs_raw, device)
            ep_ret += sum(rews.values()) / max(len(rews), 1)

            if render:
                p_env.render()
        returns.append(ep_ret)
        
    eval_metrics = {
        'eval/mean_return': float(np.mean(returns)),
        'eval/std_return': float(np.std(returns)),
        'eval/min_return': float(np.min(returns)),
        'eval/max_return': float(np.max(returns)),
    }
    
    print(f"Evaluation over {episodes} episodes:")
    print(f"  Mean: {eval_metrics['eval/mean_return']:.2f}")
    print(f"  Std:  {eval_metrics['eval/std_return']:.2f}")
    
    return eval_metrics

###############################################################################
# Entry‑point                                                                 #
###############################################################################

if __name__ == "__main__":
    from pettingzoo.butterfly import knights_archers_zombies_v10 as kaz

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  
        torch.backends.cuda.matmul.allow_tf32 = True  
        print(f"✅ CUDA optimizations enabled for {torch.cuda.get_device_name()}")


    RUN_ID = time.strftime("%Y%m%d-%H%M%S")
    CKPT_DIR = os.path.join("checkpoints", "three_layer_pd_marl", RUN_ID)

    # Knights–Archers–Zombies is parallel by default and uses a Discrete(9) action space.
    env = kaz.parallel_env(
        render_mode=None,
        vector_state=True,
        use_typemasks=True,  # Important for identifying zombies
    )

    train(
        env,
        num_episodes=500,
        # Constraint parameters
        # For safety: "agents must maintain safety 90% of the time"
        # For border: "zombies must be kept from border 90% of the time"  
        constraint_thresholds=[0.7, 0.7],  # Per-episode limits
        eta=1e-4,
        alpha=2e-3,
        multiplier_update_freq="step",
        communication_topology="fully_connected",
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger="wandb",
        project="three_layer_pd_marl",
        seed=0,
        grad_clip=5.0,
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
        lambda_dim=2,
        ckpt_dir=CKPT_DIR,
    )

    env.close()