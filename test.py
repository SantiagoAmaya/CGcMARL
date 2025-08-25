import torch
import numpy as np
from core import Agent, RegionCritic, MaxSumPlanner
from utils import to_tensors, flat_index, unflatten_index

def test_basic_functionality():
    """Test core components work together"""
    device = torch.device("cpu")
    
    # Test flat_index / unflatten_index
    actions = (1, 2, 0)
    sizes = (3, 4, 2)
    idx = flat_index(actions, sizes)
    recovered = unflatten_index(idx, sizes)
    assert recovered == actions, f"Index conversion failed: {actions} != {recovered}"
    print("✓ Index conversion test passed")
    
    # Test RegionCritic
    critic = RegionCritic(obs_dim_per_agent=4, lambda_dim=1, action_sizes=[3, 3])
    obs = torch.randn(2 * (4 + 1))  # 2 agents, 4 obs + 1 lambda dim
    q_values = critic(obs)
    assert q_values.shape == (9,), f"Q-values shape wrong: {q_values.shape}"
    print("✓ RegionCritic test passed")
    
    # Test MaxSumPlanner (with fixed indentation)
    region_graph = {"R1": ["a1", "a2"], "R2": ["a2", "a3"]}
    action_sizes = {"a1": 3, "a2": 3, "a3": 2}
    planner = MaxSumPlanner(region_graph, action_sizes, device=device)
    
    q_tables = {
        "R1": torch.randn(3, 3),
        "R2": torch.randn(3, 2)
    }
    
    planner.step(q_tables)
    actions = planner.select_actions()
    assert all(ag in actions for ag in ["a1", "a2", "a3"])
    print("✓ MaxSumPlanner test passed")
    
    print("\n✅ All basic tests passed!")

if __name__ == "__main__":
    test_basic_functionality()