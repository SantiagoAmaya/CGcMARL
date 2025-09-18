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
    - r¹: Local defense perimeter (avg zombie distance to agent's circle) - LOCAL CONSTRAINT
    - r²: Border defense line (avg zombie distance to defense line) - GLOBAL CONSTRAINT
    
    Dense reward shaping (added to r⁰):
   - Looking at zombies
   - Shooting arrows near zombies
   - Knights approaching zombies
   - Knights slashing near zombies
    """
    
    def __init__(self, use_dense_rewards: bool = True):
       """Initialize with defense perimeters and line positions."""
       self.use_dense_rewards = use_dense_rewards
       # Defense perimeter radii for different agent types
       self.defense_radii = {
           'knight': 0.12,  # Knights maintain smaller perimeter
           'archer': 0.25,  # Archers maintain larger perimeter
       }
       # Global defense line position 
       # Agents are at bottom (y≈0.9-1.0), zombies spawn at top (y≈0)
       # Defense line at y=0.2 keeps zombies in top 1/5 of screen
       self.defense_line_y = 0.2  # Defense line position (1/5 from top)
       
       # Approximate agent starting position (bottom of screen)
       self.agent_start_y = 0.9  # Agents start near bottom
       
       # Desired average distances for constraints
       self.min_avg_distance_local = {
           'knight': 0.08,  # Knights want zombies at ~0.08 avg distance
           'archer': 0.20,  # Archers want zombies at ~0.20 avg distance
       }
       # Distance from defense line (positive = zombie above line, negative = below)
       self.min_avg_distance_global = 0.1  # Want zombies at least 0.1 above defense line

       # Dense reward weights
       self.dense_reward_weights = {
            'looking_at_zombie': 0.01,      # Small reward for facing zombies
            'arrow_near_zombie': 0.05,       # Reward for arrows close to zombies
            'knight_approaching': 0.02,     # Reward knights for closing distance
            'knight_slash_near': 0.1,       # Larger reward for slashing near zombies
            'distance_penalty': -0.005,     # Small penalty for being far from zombies
            }
    
    def decompose_rewards(self, 
                         observations: Dict[str, torch.Tensor],
                         rewards: Dict[str, float],
                         info: Dict = None) -> Dict[str, List[float]]:
        """
        Decompose scalar rewards into multiple objectives.
        
        Returns:
            Dict mapping agent_id to [r⁰, r¹, r²]
            - r⁰: Primary reward (zombie kills) - unchanged from environment
                    + dense reward shaping if enabled
            - r¹: Safety (1.0 if not about to die, 0.0 if death imminent)
            - r²: Border defense (fraction of zombies NOT at border)
        """
        decomposed = {}
        
        for agent_id, obs in observations.items():
            # r⁰: Primary reward (zombie kills from environment)
            r0 = rewards.get(agent_id, 0.0)

            # Add dense rewards if enabled
            if self.use_dense_rewards:
                dense_reward = self._compute_dense_reward(obs, agent_id)
                r0 = r0 + dense_reward * 2.0
            
            # Extract distances from observation
            r1, r2 = self._compute_constraint_rewards(obs, agent_id)
            
            decomposed[agent_id] = [r0, r1, r2]
        
        return decomposed
    
    def _compute_dense_reward(self, obs: torch.Tensor, agent_id: str) -> float:
        """
        Compute dense reward shaping to encourage good behaviors.
        """
        if obs.dim() == 1:
            features_per_entity, has_typemask, num_entities = self._detect_observation_format(obs)
            obs = obs.view(num_entities, features_per_entity)
        else:
            features_per_entity, has_typemask, num_entities = self._detect_observation_format(obs)
        
        if not has_typemask:
            return 0.0
        
        dense_reward = 0.0
        # Base survival bonus - small reward for staying alive
        dense_reward += 0.005  # Small per-step bonus
        debug_info = {}
        
        # Get agent info from first row
        agent_abs_x = obs[0, 7].item()  # Absolute position
        agent_abs_y = obs[0, 8].item()  
        agent_heading_x = obs[0, 9].item()  # Heading unit vector
        agent_heading_y = obs[0, 10].item()
        
        # Normalize heading
        heading_norm = np.sqrt(agent_heading_x**2 + agent_heading_y**2)
        if heading_norm > 0:
            agent_heading_x /= heading_norm
            agent_heading_y /= heading_norm
        
        # Determine agent type
        is_knight = 'knight' in agent_id.lower()
        is_archer = 'archer' in agent_id.lower()
        
        # Track all entity positions and types
        zombies_info = []
        arrows_info = []
        swords_info = []
        
        # Analyze all entities
        for i in range(1, obs.shape[0]):
            if torch.abs(obs[i]).sum() < 1e-6:
                continue
            
            # Typemask: [zombie, archer, knight, sword, arrow, current]
            typemask = obs[i, :6]
            is_zombie = typemask[0].item() > 0.5
            is_arrow = typemask[4].item() > 0.5
            is_sword = typemask[3].item() > 0.5

            # Debug: Ensure we're detecting entities
            if is_zombie or is_arrow or is_sword:
                distance = obs[i, 6].item()
                # Sanity check distance
                if distance < 0 or distance > 2.0:
                    print(f"[WARNING] Unusual distance {distance} for entity type")
                    continue
            
            distance = obs[i, 6].item()
            rel_x = obs[i, 7].item()  # Position relative to agent
            rel_y = obs[i, 8].item()
            entity_heading_x = obs[i, 9].item()  # Entity's own heading (not used for most)
            entity_heading_y = obs[i, 10].item()
            
            if is_zombie and distance > 0:
                zombies_info.append({
                    'distance': distance,
                    'rel_x': rel_x,
                    'rel_y': rel_y
                })
            
            elif is_arrow:
                arrows_info.append({
                    'distance': distance,
                    'rel_x': rel_x,
                    'rel_y': rel_y
                })
            
            elif is_sword:
                swords_info.append({
                    'distance': distance,
                    'rel_x': rel_x,
                    'rel_y': rel_y
                })
        
        debug_info['n_zombies'] = len(zombies_info)
        debug_info['n_arrows'] = len(arrows_info)
        debug_info['n_swords'] = len(swords_info)

        # Compute rewards based on zombie positions
        if zombies_info:
            closest_zombie = min(zombies_info, key=lambda z: z['distance'])
            
            # 1. Reward for looking at zombies
            for zombie in zombies_info:
                # Direction vector to zombie (using relative position)
                direction_to_zombie_x = zombie['rel_x']
                direction_to_zombie_y = zombie['rel_y']
                
                # Normalize direction vector
                dist = np.sqrt(direction_to_zombie_x**2 + direction_to_zombie_y**2)
                if dist > 0:
                    direction_to_zombie_x /= dist
                    direction_to_zombie_y /= dist
                    
                    # Dot product with heading (cosine similarity)
                    # This tells us how aligned the agent's facing direction is with the direction to the zombie
                    look_score = agent_heading_x * direction_to_zombie_x + agent_heading_y * direction_to_zombie_y
                    
                    
                    # look_score ranges from -1 (opposite direction) to 1 (perfectly aligned)
                    if look_score > 0.5:  # Looking somewhat towards zombie (>60 degrees)
                        # Scale reward by alignment and inverse distance
                        dense_reward += self.dense_reward_weights['looking_at_zombie'] * look_score / max(zombie['distance'], 0.1)
            
            # 2. Reward for arrows near zombies (if archer)
            if is_archer and arrows_info:
                # For each arrow, check distance to zombies
                for arrow in arrows_info:
                    # Convert arrow relative position to absolute
                    arrow_abs_x = agent_abs_x + arrow['rel_x']
                    arrow_abs_y = agent_abs_y + arrow['rel_y']
                    
                    for zombie in zombies_info:
                        # Convert zombie relative position to absolute
                        zombie_abs_x = agent_abs_x + zombie['rel_x']
                        zombie_abs_y = agent_abs_y + zombie['rel_y']
                        
                        # Calculate actual world distance
                        arrow_zombie_dist = np.sqrt(
                            (arrow_abs_x - zombie_abs_x)**2 + 
                            (arrow_abs_y - zombie_abs_y)**2
                        )
                       
                        # Reward if arrow is close to zombie
                        if arrow_zombie_dist < 0.1:  # Very close
                            dense_reward += self.dense_reward_weights['arrow_near_zombie']
                        elif arrow_zombie_dist < 0.2:  # Somewhat close
                            dense_reward += self.dense_reward_weights['arrow_near_zombie'] * 0.5

            
            # 3. Reward knights for approaching zombies
            if is_knight:
                # Reward inversely proportional to distance to closest zombie
                if closest_zombie['distance'] < 0.3:  # Close range
                    approach_reward = (0.3 - closest_zombie['distance']) / 0.3
                    dense_reward += self.dense_reward_weights['knight_approaching'] * approach_reward
                
                # 4. Reward for slashing near zombies
                # Only check for swords very close to this knight (likely theirs)
                active_sword = any(s['distance'] < 0.05 for s in swords_info)
                
                if active_sword:
                    # This knight is currently slashing
                    for zombie in zombies_info:
                        if zombie['distance'] < 0.2:  # Within sword range
                            # Check if facing the zombie
                            dir_x = zombie['rel_x'] / max(zombie['distance'], 0.01)
                            dir_y = zombie['rel_y'] / max(zombie['distance'], 0.01)
                            look_score = agent_heading_x * dir_x + agent_heading_y * dir_y
                            
                            if look_score > 0.5:  # Facing somewhat toward zombie
                                # Reward based on how well-aimed the slash is
                                reward_mult = look_score * (0.2 - zombie['distance']) / 0.2
                                dense_reward += self.dense_reward_weights['knight_slash_near'] * reward_mult
                
            
            # Small penalty for being too far from all zombies (encourage engagement)
            avg_distance = sum(z['distance'] for z in zombies_info) / len(zombies_info)
            if avg_distance > 0.5:
                dense_reward += self.dense_reward_weights['distance_penalty'] * (avg_distance - 0.5)
        
        # Occasionally log debug info
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        
        if self._debug_counter % 50000 == 0:  # Every 1000 calls
            print(f"[DENSE DEBUG] {agent_id}: zombies={debug_info['n_zombies']}, "
                    f"arrows={debug_info['n_arrows']}, swords={debug_info['n_swords']}, "
                    f"dense_reward={dense_reward:.6f}")

        return dense_reward
    
    
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
        """
        Compute distance-based constraint rewards.
        Returns (r1, r2) where:
        - r1: Local defense perimeter (avg distance of zombies to agent's circle)
        - r2: Global defense line (avg distance of zombies to defense line)
        """
        # Debug: Print observation shape periodically
        if np.random.random() < 0.00001:  
            print(f"[DEBUG] Observation shape for {agent_id}: {obs.shape}")

        if obs.dim() == 1:
            features_per_entity, has_typemask, num_entities = self._detect_observation_format(obs)
            obs = obs.view(num_entities, features_per_entity)
        else:
            features_per_entity, has_typemask, num_entities = self._detect_observation_format(obs)
        
        if not has_typemask:
            # This implementation requires typemasks to identify zombies
            # raise ValueError("KAZRewardDecomposer requires use_typemasks=True")
            print(f"[WARNING] No typemasks available, returning neutral constraint rewards")
            return 0.0, 0.0
        
        # Get ACTUAL agent position from observation (row 0)
        # With typemask: [6 typemask values, 0, pos_x, pos_y, heading_x, heading_y]
        agent_x = obs[0, 7].item()  # Actual normalized x position [0, 1]
        agent_y = obs[0, 8].item()  # Actual normalized y position [0, 1]
        
        # Determine agent type from ID
        agent_type = 'knight' if 'knight' in agent_id.lower() else 'archer'
        defense_radius = self.defense_radii[agent_type]
        min_local_dist = self.min_avg_distance_local[agent_type]
        
        # Track zombie information
        zombie_distances = []
        zombie_y_positions = []
        
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
                rel_x = obs[i, 7].item()
                rel_y = obs[i, 8].item()
                
                # Sanity check on distance
                if distance <= 0 or distance > 2.0:
                    continue
                
                zombie_distances.append(distance)
                # Calculate absolute y position
                zombie_x_absolute = agent_x + rel_x
                zombie_y_absolute = agent_y + rel_y
                zombie_y_positions.append(zombie_y_absolute)
        
        # r¹: Local defense perimeter constraint
        if zombie_distances:
            # Calculate average distance to agent's defense perimeter
            # Distance to perimeter = max(0, defense_radius - actual_distance)
            avg_distance = sum(zombie_distances) / len(zombie_distances)
            
            # Constraint satisfaction: avg distance should be >= min_local_dist
            # Positive r1 = constraint satisfied, negative = violated
            r1 = avg_distance - min_local_dist
            
            # Clip to reasonable range
            r1 = max(-2.0, min(1.0, r1))
            
        else:
            # No zombies visible - constraint satisfied
            r1 = 0.5
        
        # r²: Global defense line constraint
        if zombie_y_positions:
            # Calculate average distance of zombies to defense line
            # Defense line is at y = self.defense_line_y (0.2)
            # We want zombies to stay ABOVE this line (lower y values)
            distances_to_line = []
            for zombie_y in zombie_y_positions:
                # Distance from zombie to defense line
                # Positive if zombie is above line (zombie_y < defense_line_y) - GOOD
                # Negative if zombie is below line (zombie_y > defense_line_y) - BAD
                dist_to_line = self.defense_line_y - zombie_y
                distances_to_line.append(dist_to_line)
            
            avg_distance_to_line = sum(distances_to_line) / len(distances_to_line)
            # Constraint: zombies should be above the line by at least min_avg_distance_global
            # r2 > 0 means constraint satisfied (zombies far enough above line)
            # r2 < 0 means constraint violated (zombies too close or past line)
            r2 = avg_distance_to_line - self.min_avg_distance_global
            
            # Clip to reasonable range
            r2 = max(-2.0, min(1.0, r2))         
        else:
            # No zombies = constraint satisfied
            r2 = 0.5

        # Debug: Log zombie detection
        if np.random.random() < 0.0001:
            print(f"[DEBUG] {agent_id}: Found {len(zombie_distances)} zombies")
        
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
    
    
"""
4. FCTOR GRAPH
-------------------------
Flexible factor graphs for coordination (MAX-SUM)
"""
class FactorGraph:
    """
    Generates factor graph structures for Max-Sum coordination.
    Each region becomes a factor node with its own Q-function critic.
    """
    
    def __init__(self, agents: List[str], topology: str = "pairwise", 
                 custom_regions: Dict[str, List[str]] = None):
        """
        Args:
            agents: List of agent names
            topology: Type of factor graph structure
                - "global": Single region with all agents (current default)
                - "individual": Each agent is its own region
                - "pairwise": Pairs of agents by type (archers, knights)
                - "local_pairs": Adjacent pairs (left/right teams)
                - "overlapping": Multiple overlapping coordination patterns
                - "hierarchical": Tree structure with local and global coordination
                - "fully_factored": All possible pairs (expensive!)
                - "custom": Use custom_regions parameter
            custom_regions: Dictionary defining custom regions (used when topology="custom")
        """
        self.agents = agents
        self.topology = topology
        self.custom_regions = custom_regions
        
        # Identify agent types
        self.archers = [a for a in agents if 'archer' in a.lower()]
        self.knights = [a for a in agents if 'knight' in a.lower()]
        
        # Build the region graph
        self.region_graph = self._build_graph()
        
        # Compute statistics
        self.num_regions = len(self.region_graph)
        self.max_region_size = max(len(agents) for agents in self.region_graph.values()) if self.region_graph else 0
        self.total_action_space = self._compute_total_action_space()
    
    def _build_graph(self) -> Dict[str, List[str]]:
        """Build factor graph based on topology."""
        
        if self.topology == "global":
            # Single region with all agents (current implementation)
            # Action space: 9^4 = 6561
            return {"global": self.agents}
        
        elif self.topology == "individual":
            # Each agent gets its own region (no coordination)
            # Action space: 4 regions × 9 actions = 36 total
            regions = {f"agent_{ag}": [ag] for ag in self.agents}
            return regions
        
        elif self.topology == "pairwise":
            # Agents coordinate by type
            # Action space: 2 regions × 9^2 = 162 total
            regions = {}
            if self.archers:
                regions["archers"] = self.archers
            if self.knights:
                regions["knights"] = self.knights
            return regions
        
        elif self.topology == "local_pairs":
            # Left/right team coordination
            # Action space: 2 regions × 9^2 = 162 total
            regions = {}
            if len(self.archers) >= 1 and len(self.knights) >= 1:
                regions["left_team"] = [self.archers[0], self.knights[0]]
                if len(self.archers) >= 2 and len(self.knights) >= 2:
                    regions["right_team"] = [self.archers[1], self.knights[1]]
            return regions
        
        elif self.topology == "overlapping":
            # Multiple overlapping regions for rich coordination
            # Action space: 4 regions × 9^2 = 324 total
            regions = {}
            
            # Type-based coordination
            if len(self.archers) >= 2:
                regions["archers"] = self.archers[:2]
            if len(self.knights) >= 2:
                regions["knights"] = self.knights[:2]
            
            # Spatial coordination (assuming 0=left, 1=right)
            if self.archers and self.knights:
                regions["left_team"] = [self.archers[0], self.knights[0]]
                if len(self.archers) > 1 and len(self.knights) > 1:
                    regions["right_team"] = [self.archers[1], self.knights[1]]
            
            return regions
        
        elif self.topology == "hierarchical":
            # Tree structure: individuals + pairs + global
            # Good balance of local and global coordination
            regions = {}
            
            # Level 1: Individual agents
            for ag in self.agents:
                regions[f"L1_{ag}"] = [ag]
            
            # Level 2: Type-based pairs
            if len(self.archers) >= 2:
                regions["L2_archers"] = self.archers[:2]
            if len(self.knights) >= 2:
                regions["L2_knights"] = self.knights[:2]
            
            # Level 3: Cross-type coordination
            if self.archers and self.knights:
                regions["L3_mixed"] = [self.archers[0], self.knights[0]]
            
            return regions
        
        elif self.topology == "fully_factored":
            # All possible pairs (expensive but maximum coordination)
            # Action space: 6 regions × 9^2 = 486 total
            regions = {}
            for i, a1 in enumerate(self.agents):
                for j, a2 in enumerate(self.agents):
                    if i < j:  # Avoid duplicates
                        regions[f"{a1}_{a2}"] = [a1, a2]
            return regions
        
        elif self.topology == "adaptive":
            # Adaptive regions based on agent count
            n = len(self.agents)
            if n <= 2:
                return {"global": self.agents}
            elif n <= 4:
                return self._build_overlapping_regions()
            else:
                return self._build_hierarchical_regions()
        
        elif self.topology == "custom":
            if self.custom_regions is None:
                raise ValueError("custom_regions must be provided when topology='custom'")
            return self.custom_regions
        
        else:
            raise ValueError(f"Unknown topology: {self.topology}")
    
    def _build_overlapping_regions(self) -> Dict[str, List[str]]:
        """Helper for overlapping regions."""
        regions = {}
        
        # Type-based
        if self.archers:
            for i in range(0, len(self.archers), 2):
                pair = self.archers[i:i+2]
                if len(pair) > 1:
                    regions[f"archers_{i//2}"] = pair
        
        if self.knights:
            for i in range(0, len(self.knights), 2):
                pair = self.knights[i:i+2]
                if len(pair) > 1:
                    regions[f"knights_{i//2}"] = pair
        
        # Cross-type
        for i, archer in enumerate(self.archers):
            for j, knight in enumerate(self.knights):
                if i == j:  # Same index = same team
                    regions[f"team_{i}"] = [archer, knight]
        
        return regions
    
    def _build_hierarchical_regions(self) -> Dict[str, List[str]]:
        """Helper for hierarchical regions."""
        regions = {}
        
        # Bottom level: individuals
        for ag in self.agents:
            regions[f"self_{ag}"] = [ag]
        
        # Middle level: pairs
        for i in range(0, len(self.agents), 2):
            pair = self.agents[i:i+2]
            if len(pair) == 2:
                regions[f"pair_{i//2}"] = pair
        
        # Top level: quads (if enough agents)
        for i in range(0, len(self.agents), 4):
            quad = self.agents[i:i+4]
            if len(quad) >= 3:  # At least 3 agents
                regions[f"quad_{i//4}"] = quad
        
        return regions
    
    def _compute_total_action_space(self) -> int:
        """Compute total action space size across all regions."""
        total = 0
        action_per_agent = 9  # KAZ environment has 9 actions
        for agents in self.region_graph.values():
            total += action_per_agent ** len(agents)
        return total
    
    def get_region_graph(self) -> Dict[str, List[str]]:
        """Get the region graph dictionary for use in training."""
        return self.region_graph
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the factor graph."""
        return {
            "num_regions": self.num_regions,
            "max_region_size": self.max_region_size,
            "total_action_space": self.total_action_space,
            "num_agents": len(self.agents),
        }
    
    def visualize(self) -> str:
        """Generate a text visualization of the factor graph."""
        lines = [f"Factor Graph Topology: {self.topology}"]
        lines.append(f"Agents: {self.agents}")
        lines.append(f"Number of regions: {self.num_regions}")
        lines.append(f"Total action space: {self.total_action_space}")
        lines.append("\nRegions:")
        
        for region_name, agents in self.region_graph.items():
            action_space = 9 ** len(agents)
            lines.append(f"  {region_name}: {agents} (action space: {action_space})")
        
        return "\n".join(lines)