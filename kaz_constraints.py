"""
Constraint System for Knights-Archers-Zombies (KAZ) Environment
===============================================================

1. REWARD DECOMPOSITION
-----------------------
We'll decompose the environment into multiple objectives:
"""

import numpy as np
import torch
from typing import Dict, List, Tuple

class KAZRewardDecomposer:
    """
    Decomposes KAZ rewards into multiple objectives for constrained MARL.
    
    Objectives:
    - r⁰: Primary reward (zombie kills) - MAXIMIZE
    - r¹: Safety distance (agent-zombie distance) - LOCAL CONSTRAINT
    - r²: Border defense (zombie-border distance) - GLOBAL CONSTRAINT
    """
    
    def __init__(self, 
                 danger_distance: float = 0.02,    # Only penalize extreme proximity
                 border_danger_rel_y: float = 0.1):  # Bottom 10% is critical
        """
        Args:
            danger_distance: Distance for imminent death
            border_danger_rel_y: Relative y threshold for border danger
                                (zombie this far below agent is near border)
        """
        self.danger_distance = danger_distance
        self.border_danger_rel_y = border_danger_rel_y

    
    def decompose_rewards(self, 
                         observations: Dict[str, torch.Tensor],
                         rewards: Dict[str, float],
                         info: Dict = None) -> Dict[str, List[float]]:
        """
        Decompose scalar rewards into multiple objectives.
        
        Returns:
            Dict mapping agent_id to [r⁰, r¹, r²]
            - r⁰: Primary reward (zombie kills) - unchanged from environment
            - r¹: Safety (1.0 if not about to die, 0.0 if death imminent)
            - r²: Border defense (fraction of zombies NOT at border)
        """
        decomposed = {}
        
        for agent_id, obs in observations.items():
            # r⁰: Primary reward (zombie kills from environment)
            r0 = rewards.get(agent_id, 0.0)
            
            # Extract distances from observation
            r1, r2 = self._compute_constraint_rewards(obs, agent_id)
            
            decomposed[agent_id] = [r0, r1, r2]
        
        return decomposed
    
    
    def _detect_observation_format(self, obs: torch.Tensor) -> Tuple[int, bool, int]:
        """
        Returns: (features_per_entity, has_typemask, num_entities)
        """
        total_features = obs.numel() if obs.dim() == 1 else obs.shape[1]

        # Simple format detection
        if total_features % 11 == 0:
            return (11, True, total_features // 11)
        elif total_features % 5 == 0:
            return (5, False, total_features // 5)
        else:
            raise ValueError(f"Non-standard observation size {total_features}")
    
    def _compute_constraint_rewards(self, obs: torch.Tensor, agent_id: str) -> Tuple[float, float]:
        
        if obs.dim() == 1:
            features_per_entity, has_typemask, num_entities = self._detect_observation_format(obs)
            obs = obs.view(num_entities, features_per_entity)
        else:
            features_per_entity, has_typemask, num_entities = self._detect_observation_format(obs)
        
        if not has_typemask:
            # This implementation requires typemasks to identify zombies
            raise ValueError("KAZRewardDecomposer requires use_typemasks=True")
        
        # Track zombie information
        min_zombie_distance = float('inf')
        total_zombies = 0
        zombies_in_danger_zone = 0
        
        # Iterate through all other entities (skip first row which is current agent)
        for i in range(1, obs.shape[0]):
            # Check if entity exists (entire row must be non-zero)
            if torch.abs(obs[i]).sum() < 1e-6:
                continue
            
            # Typemask: [zombie, archer, knight, sword, arrow, current]
            typemask = obs[i, :6]
            is_zombie = typemask[0].item() > 0.5
            
            if is_zombie:
                # After typemask (6 values), the structure is:
                # [distance, rel_x, rel_y, heading_x, heading_y]
                distance = obs[i, 6].item()
                zombie_rel_y = obs[i, 8].item()
                
                # Sanity check on distance
                if distance <= 0 or distance > 2.0:
                    continue
                
                total_zombies += 1
                min_zombie_distance = min(min_zombie_distance, distance)
                
                # Count zombies that are dangerously below the agent
                if zombie_rel_y > self.border_danger_rel_y:
                    zombies_in_danger_zone += 1
        
        # r¹: Safety constraint (binary with small smooth zone)
        r1 = 1.0 if min_zombie_distance > self.danger_distance else 0.0
        # r²: Border defense (based on relative positions)
        r2 = 1.0 - (zombies_in_danger_zone / total_zombies) if total_zombies > 0 else 1.0
        
        return r1, r2
    
    
"""
3. COMMUNICATION TOPOLOGY
-------------------------
Flexible communication graphs for gossip
"""

class CommunicationGraph:
    """Manages agent communication topology for multiplier gossip"""
    
    def __init__(self, agents: List[str], topology: str = "fully_connected"):
        self.agents = agents
        self.topology = topology
        self.graph = self._build_graph()
        self.weights = self._compute_weights()
    
    def _build_graph(self) -> Dict[str, List[str]]:
        """Build communication graph based on topology"""
        n = len(self.agents)
        
        if self.topology == "fully_connected":
            # Everyone talks to everyone
            return {ag: [a for a in self.agents if a != ag] for ag in self.agents}
        
        elif self.topology == "line":
            # Linear chain: agent_i talks to agent_{i-1} and agent_{i+1}
            graph = {}
            for i, ag in enumerate(self.agents):
                neighbors = []
                if i > 0:
                    neighbors.append(self.agents[i-1])
                if i < n - 1:
                    neighbors.append(self.agents[i+1])
                graph[ag] = neighbors
            return graph
        
        elif self.topology == "ring":
            # Circular: like line but with wraparound
            graph = {}
            for i, ag in enumerate(self.agents):
                prev_idx = (i - 1) % n
                next_idx = (i + 1) % n
                graph[ag] = [self.agents[prev_idx], self.agents[next_idx]]
            return graph
        
        elif self.topology == "star":
            # Central hub (first agent) connected to all others
            hub = self.agents[0]
            graph = {hub: self.agents[1:]}
            for ag in self.agents[1:]:
                graph[ag] = [hub]
            return graph
        
        else:
            raise ValueError(f"Unknown topology: {self.topology}")
    
    def _compute_weights(self) -> Dict[str, Dict[str, float]]:
        """Compute symmetric Metropolis weights for gossip"""
        weights = {}
        
        for ag in self.agents:
            neighbors = self.graph[ag]
            degree_i = len(neighbors)
            
            ag_weights = {}
            
            # Weights for neighbors
            for neighbor in neighbors:
                degree_j = len(self.graph[neighbor])
                # Metropolis weight
                w_ij = 1.0 / (1 + max(degree_i, degree_j))
                ag_weights[neighbor] = w_ij
            
            # Self-weight (ensures weights sum to 1)
            ag_weights[ag] = 1.0 - sum(ag_weights.values())
            
            weights[ag] = ag_weights
        
        return weights
    
    def get_neighbors(self, agent: str) -> List[str]:
        """Get communication neighbors for an agent"""
        return self.graph.get(agent, [])
    
    def get_weights(self, agent: str) -> Dict[str, float]:
        """Get gossip weights for an agent"""
        return self.weights.get(agent, {})