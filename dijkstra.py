# dijkstra_solver.py
import numpy as np
import heapq
from helper import (
    BasePathfinder, distance_euclidean,
    get_valid_neighbors, is_valid_and_not_obstacle
)
from env import START_NODE_VAL, TARGET_NODE_VAL

class DijkstraSolver(BasePathfinder):
    def __init__(self, grid,
                 turn_penalty_factor=0.1, safety_penalty_factor=0.05, min_safe_distance=1.5,
                 allow_diagonal_moves=True,
                 restrict_diagonal_near_obstacle_policy=True, # Dijkstra cũng sẽ tuân thủ nghiêm ngặt
                 diagonal_obstacle_penalty_value=1000.0):

        start_nodes_found = np.argwhere(grid == START_NODE_VAL)
        target_nodes_found = np.argwhere(grid == TARGET_NODE_VAL)
        if not start_nodes_found.size > 0: raise ValueError("Dijkstra: Start node not found.")
        if not target_nodes_found.size > 0: raise ValueError("Dijkstra: Target node not found.")
        start_node = tuple(start_nodes_found[0])
        target_node = tuple(target_nodes_found[0])

        super().__init__(grid, start_node, target_node,
                         turn_penalty_factor, safety_penalty_factor, min_safe_distance,
                         allow_diagonal_moves,
                         restrict_diagonal_near_obstacle_policy,
                         diagonal_obstacle_penalty_value)
        
        self.dijkstra_strictly_restricts_corners = self.restrict_diagonal_near_obstacle_policy

    def solve(self, start_node_override=None, target_node_override=None, nodes_to_avoid=None):
        _start_node = start_node_override if start_node_override else self.start_node
        _target_node = target_node_override if target_node_override else self.target_node

        if not is_valid_and_not_obstacle(_start_node[0], _start_node[1], self.grid, self.rows, self.cols) or \
           not is_valid_and_not_obstacle(_target_node[0], _target_node[1], self.grid, self.rows, self.cols):
            return self._calculate_stats_for_path([])

        if _start_node == _target_node:
            return self._calculate_stats_for_path([_start_node])

        open_set = [] 
        heapq.heappush(open_set, (0, _start_node)) # (g_score, node)
        
        came_from = {}
        g_score = {_start_node: 0}

        closed_set = set()
        if nodes_to_avoid:
            closed_set.update(nodes_to_avoid)
            if _start_node in closed_set: closed_set.remove(_start_node)
            if _target_node in closed_set: closed_set.remove(_target_node)

        max_steps = self.rows * self.cols * 3
        steps = 0
        while open_set and steps < max_steps:
            steps += 1
            current_g, current_node = heapq.heappop(open_set)

            if current_node == _target_node:
                path = []
                temp = current_node
                while temp in came_from: path.append(temp); temp = came_from[temp]
                path.append(_start_node)
                path.reverse()
                self.convergence_curve.append(current_g)
                return self._calculate_stats_for_path(path)

            if current_node in closed_set: continue
            closed_set.add(current_node)

            neighbors = get_valid_neighbors(
                current_node[0], current_node[1], self.grid, self.rows, self.cols,
                allow_diagonal_moves=self.allow_diagonal_moves,
                restrict_diagonal_corner_cutting=self.dijkstra_strictly_restricts_corners, # QUAN TRỌNG
                exclude_nodes=closed_set
            )
            
            for neighbor in neighbors:
                cost_to_neighbor = distance_euclidean(current_node, neighbor)
                tentative_g_score = current_g + cost_to_neighbor

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    
                    in_open_set_flag = any(n_open == neighbor for _, n_open in open_set)
                    if not in_open_set_flag:
                        heapq.heappush(open_set, (tentative_g_score, neighbor))
                    else:
                         for i, (g_o, n_o) in enumerate(open_set):
                            if n_o == neighbor and tentative_g_score < g_o:
                                open_set[i] = (tentative_g_score, neighbor)
                                heapq.heapify(open_set)
                                break
        return self._calculate_stats_for_path([])