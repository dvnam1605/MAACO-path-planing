# MPA.py (Phiên bản cố gắng giống paper nhất cho path planning)
import numpy as np
import random
import math
from env import OBSTACLE, START_NODE_VAL, TARGET_NODE_VAL
from visualization import plot_convergence_curve as plot_convergence_static_mpa
import heapq

class MPA:
    def __init__(self, grid, num_predators, num_iterations,
                 FADs_rate=0.2, P_const=0.5, levy_beta=1.5,
                 turn_penalty_factor=0.1):
        self.grid = np.array(grid, dtype=int)
        self.rows, self.cols = self.grid.shape
        self.num_predators = num_predators
        self.num_iterations = num_iterations
        self.FADs_rate = FADs_rate # Tham số FADs trong paper là 0.2
        self.P_const = P_const     # P=0.5 trong paper
        self.levy_beta = levy_beta # alpha cho Lévy, thường là 1.5
        self.turn_penalty_factor_mpa = turn_penalty_factor

        start_nodes_found = np.argwhere(self.grid == START_NODE_VAL)
        target_nodes_found = np.argwhere(self.grid == TARGET_NODE_VAL)
        if not start_nodes_found.size > 0: raise ValueError("MPA: Start node not found.")
        if not target_nodes_found.size > 0: raise ValueError("MPA: Target node not found.")
        self.start_node = tuple(start_nodes_found[0])
        self.target_node = tuple(target_nodes_found[0])

        self.best_path_overall = []
        self.best_path_length_overall = float('inf')
        self.best_path_turns_overall = float('inf')
        self.convergence_curve_data = []
        self.population = self._initialize_population()

    def _distance(self, node1, node2):
        return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

    def _is_valid_and_not_obstacle(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r,c] != OBSTACLE

    def _get_valid_neighbors(self, r, c, exclude_nodes=None):
        neighbors = []
        if exclude_nodes is None: exclude_nodes = set()
        for dr_nn, dc_nn in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            nr, nc = r + dr_nn, c + dc_nn
            if self._is_valid_and_not_obstacle(nr, nc) and (nr, nc) not in exclude_nodes:
                neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _a_star(self, start_node, end_node, nodes_to_avoid_in_path=None):
        if start_node == end_node: return [start_node], 0
        if not self._is_valid_and_not_obstacle(start_node[0], start_node[1]) or \
           not self._is_valid_and_not_obstacle(end_node[0], end_node[1]):
            return [], float('inf')
        open_set = []
        heapq.heappush(open_set, (0 + self._heuristic(start_node, end_node), 0, start_node))
        came_from = {}; g_score = {start_node: 0}
        if nodes_to_avoid_in_path is None: nodes_to_avoid_in_path = set()
        max_astar_steps = self.rows * self.cols * 2; astar_steps = 0
        while open_set and astar_steps < max_astar_steps:
            astar_steps += 1
            current_f, current_g, current_node = heapq.heappop(open_set)
            if current_node == end_node:
                path = []; temp_node = current_node
                while temp_node in came_from: path.append(temp_node); temp_node = came_from[temp_node]
                path.append(start_node); path.reverse()
                return path, current_g
            for neighbor in self._get_valid_neighbors(current_node[0], current_node[1], nodes_to_avoid_in_path):
                tentative_g_score = g_score[current_node] + self._distance(current_node, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node; g_score[neighbor] = tentative_g_score
                    f_score_neighbor = tentative_g_score + self._heuristic(neighbor, end_node)
                    already_in_open = False
                    for i, (f, g, n) in enumerate(open_set):
                        if n == neighbor:
                            if tentative_g_score < g: open_set[i] = (f_score_neighbor, tentative_g_score, neighbor); heapq.heapify(open_set)
                            already_in_open = True; break
                    if not already_in_open: heapq.heappush(open_set, (f_score_neighbor, tentative_g_score, neighbor))
        return [], float('inf')

    def _generate_initial_path(self):
        return self._a_star(self.start_node, self.target_node)

    def _initialize_population(self):
        population = []
        for _ in range(self.num_predators):
            path, length = self._generate_initial_path()
            if not path or length == float('inf'):
                path = [self.start_node, self.target_node] if self._distance(self.start_node, self.target_node) > 0 else [self.start_node]
                length = self._distance(self.start_node, self.target_node) if len(path) > 1 else 0
            _, _, turns, fitness = self._calculate_path_stats(path)
            population.append({'path': path, 'length': length, 'turns': turns, 'fitness': fitness})
        return population

    def _count_turns(self, path):
        if not path or len(path) < 3: return 0
        turns = 0
        for i in range(len(path) - 2):
            dr1 = path[i+1][0] - path[i][0]; dc1 = path[i+1][1] - path[i][1]
            dr2 = path[i+2][0] - path[i+1][0]; dc2 = path[i+2][1] - path[i+1][1]
            if dr1 != dr2 or dc1 != dc2: turns += 1
        return turns

    def _calculate_path_stats(self, path):
        if not path or len(path) == 0: return [], float('inf'), float('inf'), float('inf')
        length = sum(self._distance(path[i], path[i+1]) for i in range(len(path)-1)) if len(path) > 1 else 0
        turns = self._count_turns(path)
        fitness = length + self.turn_penalty_factor_mpa * turns
        return path, length, turns, fitness

    def _get_random_node_on_path(self, path_nodes):
        if not path_nodes: return None
        return random.choice(path_nodes)

    def _get_levy_target_node(self, current_node, scale=1.0):
        """Generates a target node using a Lévy-like step."""
        u_rand = random.random()
        if u_rand < 1e-6: u_rand = 1e-6
        step_len_factor = 1.0 / (u_rand ** (1.0 / self.levy_beta))
        
        max_dim_step = max(self.rows, self.cols) * 0.33 * scale # Scale can be P_const or P_const*CF
        actual_step_length = min(step_len_factor, max_dim_step)
        actual_step_length = max(actual_step_length, 1.0)

        angle = random.uniform(0, 2 * math.pi)
        dr = int(round(actual_step_length * math.sin(angle)))
        dc = int(round(actual_step_length * math.cos(angle)))
        
        target_r = max(0, min(self.rows - 1, current_node[0] + dr))
        target_c = max(0, min(self.cols - 1, current_node[1] + dc))
        return (target_r, target_c)

    def _get_brownian_target_node(self, current_node, elite_node_on_path, scale=0.1):
        """Generates a target node using a Brownian-like step (small perturbation or towards elite)."""
        if random.random() < 0.5 and elite_node_on_path: # Move towards a point on elite path
            # Small step towards elite_node_on_path
            dr = elite_node_on_path[0] - current_node[0]
            dc = elite_node_on_path[1] - current_node[1]
            dist_to_elite_pt = math.sqrt(dr**2 + dc**2)
            if dist_to_elite_pt < 1e-6: return elite_node_on_path

            step_size = max(1, int(round(dist_to_elite_pt * scale * random.random()))) # scale by random proportion
            
            target_r = current_node[0] + int(round(dr/dist_to_elite_pt * step_size))
            target_c = current_node[1] + int(round(dc/dist_to_elite_pt * step_size))

        else: # Small random perturbation
            max_step = max(1, int(round(max(self.rows, self.cols) * 0.05 * scale))) # 5% of max dim, scaled
            dr = random.randint(-max_step, max_step)
            dc = random.randint(-max_step, max_step)
            target_r = current_node[0] + dr
            target_c = current_node[1] + dc
        
        target_r = max(0, min(self.rows - 1, target_r))
        target_c = max(0, min(self.cols - 1, target_c))
        return (target_r, target_c)

    def _reconstruct_path_segment(self, path_to_modify, elite_path, start_modify_idx, 
                                  is_levy_move, step_scale_factor):
        """
        Core path modification logic.
        step_scale_factor: Corresponds to P or P*CF in equations.
        """
        if not path_to_modify or start_modify_idx >= len(path_to_modify) -1: # Cannot modify if idx is last node or beyond
            return self._calculate_path_stats(path_to_modify)

        current_node_to_modify_from = path_to_modify[start_modify_idx]
        prefix_path = path_to_modify[:start_modify_idx+1]
        nodes_to_avoid = set(prefix_path[:-1]) # Avoid re-visiting nodes in the prefix

        # Determine target for A* based on Lévy/Brownian
        if is_levy_move:
            intermediate_target = self._get_levy_target_node(current_node_to_modify_from, scale=step_scale_factor * 5) # Amplify Lévy
        else: # Brownian
            elite_node_sample = self._get_random_node_on_path(elite_path) if elite_path else None
            intermediate_target = self._get_brownian_target_node(current_node_to_modify_from, elite_node_sample, scale=step_scale_factor * 2) # Amplify Brownian scale

        new_segment_part1 = []
        new_segment_part2 = []

        if self._is_valid_and_not_obstacle(intermediate_target[0], intermediate_target[1]):
            path_to_intermediate, _ = self._a_star(current_node_to_modify_from, intermediate_target, nodes_to_avoid.copy())
            if path_to_intermediate:
                new_segment_part1 = path_to_intermediate[1:] # Exclude start node (current_node_to_modify_from)
                
                nodes_to_avoid_for_final = nodes_to_avoid.copy()
                for node in new_segment_part1: nodes_to_avoid_for_final.add(node)
                
                path_from_intermediate_to_goal, _ = self._a_star(intermediate_target, self.target_node, nodes_to_avoid_for_final)
                if path_from_intermediate_to_goal:
                    new_segment_part2 = path_from_intermediate_to_goal[1:]

        if new_segment_part1 and new_segment_part2:
            full_new_path = prefix_path + new_segment_part1 + new_segment_part2
        elif new_segment_part1: # Reached intermediate, but not goal from there, try direct from intermediate
             nodes_to_avoid_for_final = nodes_to_avoid.copy()
             for node in new_segment_part1: nodes_to_avoid_for_final.add(node)
             path_from_intermediate_to_goal, _ = self._a_star(intermediate_target, self.target_node, nodes_to_avoid_for_final)
             if path_from_intermediate_to_goal:
                 full_new_path = prefix_path + new_segment_part1 + path_from_intermediate_to_goal[1:]
             else: # Fallback: try to A* from current_node_to_modify_from to target
                 path_to_target, _ = self._a_star(current_node_to_modify_from, self.target_node, nodes_to_avoid.copy())
                 full_new_path = prefix_path + (path_to_target[1:] if path_to_target else [])
        else: # Fallback if intermediate_target is bad or path to it fails: A* from current_node_to_modify_from to target
            path_to_target, _ = self._a_star(current_node_to_modify_from, self.target_node, nodes_to_avoid.copy())
            full_new_path = prefix_path + (path_to_target[1:] if path_to_target else [])
        
        # Clean up path (remove consecutive duplicates)
        unique_path = []
        if full_new_path:
            unique_path.append(full_new_path[0])
            for i in range(1, len(full_new_path)):
                if full_new_path[i] != full_new_path[i-1]:
                    unique_path.append(full_new_path[i])
        
        if not unique_path or unique_path[-1] != self.target_node: # If path is invalid or doesn't reach target
            # Return original path's stats as modification failed to produce a valid complete path
            return self._calculate_path_stats(path_to_modify)

        return self._calculate_path_stats(unique_path)

    def solve_path_planning(self):
        self.population.sort(key=lambda p: p['fitness'])
        elite_obj = self.population[0].copy()
        self.best_path_overall = list(elite_obj['path'])
        self.best_path_length_overall = elite_obj['length']
        self.best_path_turns_overall = elite_obj['turns']
        self.convergence_curve_data.append(elite_obj['length'] if elite_obj['length'] != float('inf') else None)

        for iter_num in range(1, self.num_iterations + 1):
            self.population.sort(key=lambda p: p['fitness'])
            elite_obj = self.population[0].copy() # Current best predator

            ratio = iter_num / self.num_iterations
            CF = 0.0 if ratio >= 1.0 else ( (1.0 - ratio)**(2.0 * ratio) if ratio > 0 else 1.0)

            new_population_data = []

            # --- MPA Phases ---
            # R is a vector of uniform random numbers in [0,1] in paper.
            # Here, we use random.random() for decisions.

            if iter_num <= self.num_iterations / 3: # Phase 1 (Eq 12)
                # Prey_i_new = Prey_i + P_const * R * R_B * (Elite_i - R_B * Prey_i)
                # Interpretation: Prey (current individual) updates based on Brownian (local search)
                # influenced by Elite and its own current state.
                for i in range(self.num_predators):
                    prey_i_path = self.population[i]['path']
                    if not prey_i_path or len(prey_i_path) <= 1: # Skip if path is too short
                        new_population_data.append(self.population[i].copy())
                        continue
                    
                    start_modify_idx = random.randint(0, len(prey_i_path) - 2) if len(prey_i_path) >1 else 0

                    if random.random() < self.P_const: # P_const * R (R is implicitly random.random())
                        p, l, t, f = self._reconstruct_path_segment(prey_i_path, elite_obj['path'], 
                                                                  start_modify_idx, 
                                                                  is_levy_move=False, # R_B
                                                                  step_scale_factor=self.P_const) # Scale by P
                    else:
                        p,l,t,f = self._calculate_path_stats(prey_i_path)
                    new_population_data.append({'path': p, 'length': l, 'turns': t, 'fitness': f})

            elif iter_num <= 2 * self.num_iterations / 3: # Phase 2 (Eq 13 & 14)
                for i in range(self.num_predators):
                    prey_i_path = self.population[i]['path']
                    if not prey_i_path or len(prey_i_path) <= 1:
                        new_population_data.append(self.population[i].copy())
                        continue
                    start_modify_idx = random.randint(0, len(prey_i_path) - 2) if len(prey_i_path) > 1 else 0

                    if i < self.num_predators // 2: # First half: Prey moves Lévy (Eq 13)
                        # Prey_i_new = Prey_i + P_const * R * R_L * (Elite_i - R_L * Prey_i)
                        if random.random() < self.P_const:
                            p, l, t, f = self._reconstruct_path_segment(prey_i_path, elite_obj['path'],
                                                                      start_modify_idx,
                                                                      is_levy_move=True, # R_L
                                                                      step_scale_factor=self.P_const)
                        else:
                            p,l,t,f = self._calculate_path_stats(prey_i_path)
                    else: # Second half: Predator moves Brownian (Eq 14)
                        # Prey_i_new = Elite_i + P_const * CF * R_B * (Elite_i - R_B * Prey_i_current)
                        # Base is Elite's path for modification
                        elite_path_to_modify = elite_obj['path']
                        if not elite_path_to_modify or len(elite_path_to_modify) <=1:
                             p,l,t,f = self._calculate_path_stats(elite_path_to_modify)
                        else:
                            start_modify_idx_elite = random.randint(0, len(elite_path_to_modify) - 2) if len(elite_path_to_modify) > 1 else 0
                            if random.random() < self.P_const * CF : # Scaled by P_const * CF
                                p, l, t, f = self._reconstruct_path_segment(elite_path_to_modify, 
                                                                          prey_i_path, # Prey_i_current for reference if needed by Brownian
                                                                          start_modify_idx_elite,
                                                                          is_levy_move=False, # R_B
                                                                          step_scale_factor=self.P_const * CF)
                            else:
                                p,l,t,f = self._calculate_path_stats(elite_path_to_modify)
                    new_population_data.append({'path': p, 'length': l, 'turns': t, 'fitness': f})
            else: # Phase 3 (Eq 15)
                # Prey_i_new = Elite_i + P_const * CF * R_L * (Elite_i - R_L * Prey_i_current)
                # Base is Elite's path, modified by Lévy
                for i in range(self.num_predators):
                    elite_path_to_modify = elite_obj['path']
                    prey_i_path_ref = self.population[i]['path'] # Prey_i_current for reference
                    if not elite_path_to_modify or len(elite_path_to_modify) <= 1:
                        p,l,t,f = self._calculate_path_stats(elite_path_to_modify)
                    else:
                        start_modify_idx_elite = random.randint(0, len(elite_path_to_modify) - 2) if len(elite_path_to_modify) > 1 else 0
                        if random.random() < self.P_const * CF :
                            p, l, t, f = self._reconstruct_path_segment(elite_path_to_modify, 
                                                                      prey_i_path_ref, 
                                                                      start_modify_idx_elite,
                                                                      is_levy_move=True, # R_L
                                                                      step_scale_factor=self.P_const * CF)
                        else:
                            p,l,t,f = self._calculate_path_stats(elite_path_to_modify)
                    new_population_data.append({'path': p, 'length': l, 'turns': t, 'fitness': f})

            # Memory Update
            temp_pop_after_phases = []
            for i in range(self.num_predators):
                if new_population_data[i]['fitness'] < self.population[i]['fitness']:
                    temp_pop_after_phases.append(new_population_data[i])
                else:
                    temp_pop_after_phases.append(self.population[i].copy())
            
            self.population = [] # Reset for FADs

            # FADs Effect (Eq 16) - Simplified for path planning
            for ind_obj in temp_pop_after_phases:
                if random.random() < self.FADs_rate:
                    # First part of FADs: Prey_i_new = Prey_i + CF * [X_min + R(X_max - X_min)] * U
                    # Interpret as: current path + CF * (path to random grid point then to target)
                    if random.random() < CF: # More likely when CF is high
                        current_path_for_fad = ind_obj['path']
                        if current_path_for_fad and len(current_path_for_fad) > 1:
                            # Pick a random node on grid as intermediate
                            rand_r, rand_c = random.randint(0, self.rows-1), random.randint(0, self.cols-1)
                            rand_intermediate_node = (rand_r, rand_c)
                            
                            if self._is_valid_and_not_obstacle(rand_r, rand_c):
                                # Try Start -> Rand_Intermediate -> Target
                                path1_fad, _ = self._a_star(self.start_node, rand_intermediate_node)
                                if path1_fad:
                                    path2_fad, _ = self._a_star(rand_intermediate_node, self.target_node, set(path1_fad[1:]))
                                    if path2_fad:
                                        fad_path_v1 = path1_fad + path2_fad[1:]
                                        # Clean path
                                        unique_fad_path_v1 = []
                                        if fad_path_v1:
                                            unique_fad_path_v1.append(fad_path_v1[0])
                                            for k_idx in range(1, len(fad_path_v1)):
                                                if fad_path_v1[k_idx] != fad_path_v1[k_idx-1]: unique_fad_path_v1.append(fad_path_v1[k_idx])
                                        
                                        if unique_fad_path_v1 and unique_fad_path_v1[-1] == self.target_node:
                                            _, l_fadv1, t_fadv1, f_fadv1 = self._calculate_path_stats(unique_fad_path_v1)
                                            if f_fadv1 < ind_obj['fitness']:
                                                self.population.append({'path': unique_fad_path_v1, 'length': l_fadv1, 'turns': t_fadv1, 'fitness': f_fadv1})
                                                continue # Skip adding ind_obj
                    else:
                        # Second part of FADs (less likely with high CF): Prey_i_new = Prey_i + [FADs(1-r)+r] * (Prey_r1 - Prey_r2)
                        # Interpret as: try to incorporate elements from two other random paths.
                        # This is complex. Simpler: full re-initialization as before.
                        fad_reinit_p, fad_reinit_l = self._generate_initial_path()
                        if fad_reinit_p and fad_reinit_l != float('inf'):
                            _, _, fad_reinit_t, fad_reinit_f = self._calculate_path_stats(fad_reinit_p)
                            if fad_reinit_f < ind_obj['fitness']:
                                self.population.append({'path': fad_reinit_p, 'length': fad_reinit_l, 'turns': fad_reinit_t, 'fitness': fad_reinit_f})
                                continue
                self.population.append(ind_obj.copy())


            self.population.sort(key=lambda p: p['fitness'])
            current_iter_best_obj = self.population[0]

            # Update overall best
            current_overall_best_fitness = self.best_path_length_overall + self.turn_penalty_factor_mpa * self.best_path_turns_overall \
                                           if self.best_path_length_overall != float('inf') else float('inf')
            
            if current_iter_best_obj['fitness'] < current_overall_best_fitness:
                self.best_path_overall = list(current_iter_best_obj['path'])
                self.best_path_length_overall = current_iter_best_obj['length']
                self.best_path_turns_overall = current_iter_best_obj['turns']
            elif abs(current_iter_best_obj['fitness'] - current_overall_best_fitness) < 1e-9:
                if current_iter_best_obj['length'] < self.best_path_length_overall:
                    self.best_path_overall = list(current_iter_best_obj['path'])
                    self.best_path_length_overall = current_iter_best_obj['length']
                    self.best_path_turns_overall = current_iter_best_obj['turns']
                elif abs(current_iter_best_obj['length'] - self.best_path_length_overall) < 1e-9 and \
                     current_iter_best_obj['turns'] < self.best_path_turns_overall:
                    self.best_path_overall = list(current_iter_best_obj['path']) # Only update path if turns are better
                    self.best_path_turns_overall = current_iter_best_obj['turns']


            self.convergence_curve_data.append(self.best_path_length_overall if self.best_path_length_overall != float('inf') else \
                                               (self.convergence_curve_data[-1] if self.convergence_curve_data and self.convergence_curve_data[-1] is not None else None))
            
            if iter_num % 10 == 0 or iter_num == 1 or iter_num == self.num_iterations:
                print(f"MPA Iter {iter_num}/{self.num_iterations}: "
                      f"Iter Best L={current_iter_best_obj['length']:.2f}, T={current_iter_best_obj['turns']}, Fit={current_iter_best_obj['fitness']:.2f}, "
                      f"Overall Best L={self.best_path_length_overall:.2f}, T={self.best_path_turns_overall}")

        if self.best_path_overall:
            print(f"\nMPA Solved: Length={self.best_path_length_overall:.2f}, Turns={self.best_path_turns_overall}")
        else:
            print("\nMPA: No solution found.")
        return self.best_path_overall, self.best_path_length_overall, self.best_path_turns_overall
    
    def plot_convergence_curve(self):
        plot_convergence_static_mpa(self.convergence_curve_data, "MPA", color='blue')