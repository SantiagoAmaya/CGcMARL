"""
Constraint System for Knights-Archers-Zombies (KAZ) Environment
===============================================================

1. REWARD DECOMPOSITION
-----------------------
We'll decompose the environment into multiple objectives:
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

class KAZRewardDecomposer:
    """
    Decomposes KAZ rewards into multiple objectives for constrained MARL.
    
    Objectives:
    - r⁰: Primary reward (zombie kills) - MAXIMIZE
    - r¹: Safety distance (agent-zombie distance) - LOCAL CONSTRAINT
    - r²: Border defense (zombie-border distance) - GLOBAL CONSTRAINT
    """
    
    def __init__(self, 
                 min_agent_zombie_dist: float = 0.1,  # Minimum safe distance (normalized)
                 min_zombie_border_dist: float = 0.2,  # Minimum border distance (normalized)
                 safety_penalty: float = -0.1,         # Penalty per violation
                 border_penalty: float = -0.2):        # Penalty per border violation
        self.min_agent_zombie_dist = min_agent_zombie_dist
        self.min_zombie_border_dist = min_zombie_border_dist
        self.safety_penalty = safety_penalty
        self.border_penalty = border_penalty
    
    def decompose_rewards(self, 
                         observations: Dict[str, torch.Tensor],
                         rewards: Dict[str, float],
                         info: Dict = None) -> Dict[str, List[float]]:
        """
        Decompose scalar rewards into multiple objectives.
        
        Returns:
            Dict mapping agent_id to [r⁰, r¹, r²]
        """
        decomposed = {}
        
        for agent_id, obs in observations.items():
            # r⁰: Primary reward (zombie kills from environment)
            r0 = rewards.get(agent_id, 0.0)
            
            # Extract distances from observation
            r1, r2 = self._compute_constraint_rewards(obs, agent_id)
            
            decomposed[agent_id] = [r0, r1, r2]
        
        return decomposed
    
    def _compute_constraint_rewards(self, obs: torch.Tensor, agent_id: str) -> Tuple[float, float]:
        """
        Compute constraint-based rewards from observation.
        
        Observation structure with typemasks (11 features per entity):
        - [0:6]: Typemask [zombie, archer, knight, sword, arrow, current]
        - [6]: Distance (for non-current entities)
        - [7:9]: Relative position
        - [9:11]: Heading unit vector
        """
        if obs.dim() == 1:
            # Determine feature size
            total_features = obs.numel()
            # With typemasks: 11 features per entity
            # Without: 5 features per entity
            features_per_entity = 11 if total_features % 11 == 0 else 5
            num_entities = total_features // features_per_entity
            obs = obs.view(num_entities, features_per_entity)
        
        r1 = 0.0  # Safety constraint (agent-zombie distance)
        r2 = 0.0  # Border constraint (zombie distance from bottom)
        
        has_typemask = obs.shape[1] == 11
        
        if has_typemask:
            # Current agent's y position (normalized, 0=top, 1=bottom)
            current_y = obs[0, 8].item()  # Absolute y position of current agent
            
            # Check each entity (skip first row which is current agent)
            for i in range(1, obs.shape[0]):
                typemask = obs[i, :6]
                
                # Check if this is a zombie (first element of typemask)
                if typemask[0] > 0.5:  # Is zombie
                    distance = obs[i, 6].item()
                    
                    # Skip if distance is 0 (entity doesn't exist)
                    if distance > 0:
                        # r1: Safety - penalize if zombie too close to agent
                        if distance < self.min_agent_zombie_dist:
                            r1 += self.safety_penalty
                        
                        # r2: Border defense - zombie's y position
                        zombie_rel_y = obs[i, 8].item()  # Relative y to agent
                        zombie_abs_y = current_y + zombie_rel_y  # Absolute y
                        
                        # Distance from bottom (1.0 = at top, 0.0 = at bottom)
                        dist_from_bottom = 1.0 - min(max(zombie_abs_y, 0.0), 1.0)
                        
                        if dist_from_bottom < self.min_zombie_border_dist:
                            r2 += self.border_penalty
        else:
            # Without typemasks - need heuristics to identify zombies
            # This is less reliable but a fallback
            r1 = -0.01  # Small penalty
            r2 = -0.01
        
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