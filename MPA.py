# MPA_pathfinding_with_safety_and_diag_rules.py
import numpy as np
import random
import math
from env import OBSTACLE, START_NODE_VAL, TARGET_NODE_VAL

import heapq

class MPA:
    def __init__(self, grid, num_predators, num_iterations,
                 FADs_rate=0.2, P_const=0.5, levy_beta=1.5,
                 turn_penalty_factor=0.1,
                 safety_penalty_factor=0.05,
                 min_safe_distance=1.5,
                 allow_diagonal_moves=True, # Cho phép đi chéo nói chung
                 restrict_diagonal_near_obstacle=True, # Cấm/phạt đi chéo gần vật cản
                 diagonal_obstacle_penalty=1000.0 # Phạt lớn nếu đi chéo cắt góc (nếu không cấm hoàn toàn)
                ):
        
        self.grid = np.array(grid, dtype=int)
        self.rows, self.cols = self.grid.shape
        self.num_predators = num_predators
        self.num_iterations = num_iterations
        self.FADs_rate = FADs_rate
        self.P_const = P_const
        self.levy_beta = levy_beta
        self.turn_penalty_factor_mpa = turn_penalty_factor
        self.safety_penalty_factor_mpa = safety_penalty_factor
        self.min_safe_distance_mpa = min_safe_distance
        self.allow_diagonal_moves = allow_diagonal_moves
        self.restrict_diagonal_near_obstacle = restrict_diagonal_near_obstacle
        self.diagonal_obstacle_penalty_val = diagonal_obstacle_penalty # Lưu trữ giá trị phạt

        start_nodes_found = np.argwhere(self.grid == START_NODE_VAL)
        target_nodes_found = np.argwhere(self.grid == TARGET_NODE_VAL)
        if not start_nodes_found.size > 0:
            raise ValueError("MPA: Start node not found in grid.")
        if not target_nodes_found.size > 0:
            raise ValueError("MPA: Target node not found in grid.")
        self.start_node = tuple(start_nodes_found[0])
        self.target_node = tuple(target_nodes_found[0])
        
        self.obstacle_nodes = np.argwhere(self.grid == OBSTACLE)

        self.best_path_overall = []
        self.best_path_length_overall = float('inf')
        self.best_path_turns_overall = float('inf')
        self.best_safety_penalty_overall = float('inf')
        self.best_diag_penalty_overall = float('inf') 
        self.best_fitness_overall = float('inf')

        self.convergence_curve_data = []
        self.population = self._initialize_population_with_safety()


    def _distance(self, node1, node2):
        # Khoảng cách Euclidean cho cả đi thẳng và chéo
        dr = node1[0] - node2[0]
        dc = node1[1] - node2[1]
        return math.sqrt(dr**2 + dc**2)

    def _is_valid_and_not_obstacle(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r,c] != OBSTACLE

    def _get_valid_neighbors(self, r, c, exclude_nodes=None): 
        neighbors = []
        if exclude_nodes is None:
            exclude_nodes = set()
        
        # Các bước đi (dr, dc, is_diagonal)
        possible_moves = [
            (0, 1, False), (0, -1, False), (1, 0, False), (-1, 0, False), # Thẳng
        ]
        if self.allow_diagonal_moves:
            possible_moves.extend([
                (1, 1, True), (1, -1, True), (-1, 1, True), (-1, -1, True) # Chéo
            ])

        for dr_nn, dc_nn, is_diag_move in possible_moves:
            nr, nc = r + dr_nn, c + dc_nn

            if not self._is_valid_and_not_obstacle(nr, nc) or (nr, nc) in exclude_nodes:
                continue

            can_move_diagonally = True
            if is_diag_move and self.restrict_diagonal_near_obstacle:
                # Kiểm tra xem có vật cản ở các ô "góc" không
                # Ví dụ, đi từ (r,c) đến (r+1,c+1) (dr=1, dc=1)
                # thì kiểm tra (r+1,c) và (r,c+1)
                # Ô 1: (r + dr_nn, c)
                # Ô 2: (r, c + dc_nn)
                obstacle_at_corner1 = not self._is_valid_and_not_obstacle(r + dr_nn, c)
                obstacle_at_corner2 = not self._is_valid_and_not_obstacle(r, c + dc_nn)
                
                if obstacle_at_corner1 or obstacle_at_corner2:
                    can_move_diagonally = False # Cấm đi chéo nếu có vật cản ở góc

            if can_move_diagonally:
                neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, a, b):
        return self._distance(a,b) # Ưu tiên Euclidean cho heuristic khi có đi chéo
        # return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _a_star(self, start_node, end_node, nodes_to_avoid_in_path=None):       
        if start_node == end_node:
            return [start_node], 0
        if not self._is_valid_and_not_obstacle(start_node[0], start_node[1]) or \
           not self._is_valid_and_not_obstacle(end_node[0], end_node[1]):
            return [], float('inf')
        open_set = []
        heapq.heappush(open_set, (0 + self._heuristic(start_node, end_node), 0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        if nodes_to_avoid_in_path is None:
            nodes_to_avoid_in_path = set()
        max_astar_steps = self.rows * self.cols * 2
        astar_steps = 0
        while open_set and astar_steps < max_astar_steps:
            astar_steps += 1
            _, current_g, current_node = heapq.heappop(open_set)
            if current_node == end_node:
                path = []
                temp_node = current_node
                while temp_node in came_from:
                    path.append(temp_node)
                    temp_node = came_from[temp_node]
                path.append(start_node)
                path.reverse()
                return path, current_g
            for neighbor in self._get_valid_neighbors(current_node[0], current_node[1], nodes_to_avoid_in_path):
                # Chi phí di chuyển (g) là khoảng cách thực tế
                cost_to_neighbor = self._distance(current_node, neighbor)
                tentative_g_score = g_score[current_node] + cost_to_neighbor
            
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score_neighbor = tentative_g_score + self._heuristic(neighbor, end_node)
                    already_in_open = False
                    for i_astar, (f_astar, _, n_astar) in enumerate(open_set):
                        if n_astar == neighbor:
                            if tentative_g_score < g_score[neighbor]:
                                open_set[i_astar] = (f_score_neighbor, tentative_g_score, neighbor)
                                heapq.heapify(open_set)
                            already_in_open = True
                            break
                    if not already_in_open:
                        heapq.heappush(open_set, (f_score_neighbor, tentative_g_score, neighbor))
        return [], float('inf')


    def _generate_initial_path(self):
        return self._a_star(self.start_node, self.target_node)

    def _get_min_distance_to_obstacle(self, node_r, node_c):       
        if not self.obstacle_nodes.size > 0:
            return float('inf')
        distances_sq = np.sum((self.obstacle_nodes - np.array([node_r, node_c]))**2, axis=1)
        return math.sqrt(np.min(distances_sq))


    def _calculate_path_safety_penalty(self, path):        
        # if not path or not self.obstacle_nodes.size > 0:
        #     return 0.0
        # total_safety_penalty = 0.0
        # for r_mpa_sp, c_mpa_sp in path:
        #     min_dist_mpa_sp = self._get_min_distance_to_obstacle(r_mpa_sp, c_mpa_sp)
        #     if min_dist_mpa_sp < self.min_safe_distance_mpa:
        #         penalty_for_point_mpa_sp = (self.min_safe_distance_mpa - min_dist_mpa_sp)**2
        #         total_safety_penalty += penalty_for_point_mpa_sp
        return 0.0


    def _calculate_diagonal_proximity_penalty(self, path): 
        if not path or len(path) < 2 or not self.restrict_diagonal_near_obstacle:
            return 0.0
        
        diag_penalty = 0.0
        for i in range(len(path) - 1):
            curr_r, curr_c = path[i]
            next_r, next_c = path[i+1]
            
            dr_diag = next_r - curr_r
            dc_diag = next_c - curr_c

            # Kiểm tra nếu là bước đi chéo
            if abs(dr_diag) == 1 and abs(dc_diag) == 1:
                # Ô góc 1: (curr_r + dr_diag, curr_c) hay chính là (next_r, curr_c)
                # Ô góc 2: (curr_r, curr_c + dc_diag) hay chính là (curr_r, next_c)
                obstacle_at_corner1_check = not self._is_valid_and_not_obstacle(next_r, curr_c)
                obstacle_at_corner2_check = not self._is_valid_and_not_obstacle(curr_r, next_c)
                
                if obstacle_at_corner1_check or obstacle_at_corner2_check:
                    # Nếu A* đã cấm thì không bao giờ đến đây.
                    # Nếu A* cho phép nhưng ta muốn phạt thêm ở fitness cuối:
                    diag_penalty += self.diagonal_obstacle_penalty_val 
        return diag_penalty # Tổng phạt, không phải trung bình


    def _count_turns(self, path):
        # Đếm số lần rẽ trong đường đi
        if len(path) < 3: 
            return 0
        turns = 0
        for i_mpa_turn_count in range(len(path) - 2):
            dr1_ct, dc1_ct = path[i_mpa_turn_count+1][0]-path[i_mpa_turn_count][0], path[i_mpa_turn_count+1][1]-path[i_mpa_turn_count][1]
            dr2_ct, dc2_ct = path[i_mpa_turn_count+2][0]-path[i_mpa_turn_count+1][0], path[i_mpa_turn_count+2][1]-path[i_mpa_turn_count+1][1]
            if dr1_ct != dr2_ct or dc1_ct != dc2_ct: 
                turns += 1
        return turns


    def _calculate_path_stats(self, path): 
        if not path or len(path) == 0:
            return [], float('inf'), 0, 0.0, 0.0, float('inf') # path, len, turns, safety_p, diag_p, fitness

        length = sum(self._distance(path[i_len_stats], path[i_len_stats+1]) for i_len_stats in range(len(path)-1)) if len(path) > 1 else 0
        turns = self._count_turns(path)
        safety_penalty = self._calculate_path_safety_penalty(path)
        diag_prox_penalty = self._calculate_diagonal_proximity_penalty(path) # << TÍNH PHẠT ĐI CHÉO

        fitness = length + \
                  self.turn_penalty_factor_mpa * turns + \
                  self.safety_penalty_factor_mpa * safety_penalty + \
                  diag_prox_penalty # << THÊM PHẠT ĐI CHÉO VÀO FITNESS
        
        return path, length, turns, safety_penalty, diag_prox_penalty, fitness
    
    def _initialize_population_with_safety(self):
        population = []
        for _ in range(self.num_predators):
            initial_p_mpa_init, _ = self._generate_initial_path()
            if not initial_p_mpa_init:
                initial_p_mpa_init = [self.start_node, self.target_node] if self._is_valid_and_not_obstacle(self.target_node[0], self.target_node[1]) else [self.start_node]
            
            p_mpa_init, l_mpa_init, t_mpa_init, sp_mpa_init, dp_mpa_init, f_mpa_init = self._calculate_path_stats(initial_p_mpa_init)
            population.append({
                'path': p_mpa_init, 'length': l_mpa_init, 'turns': t_mpa_init, 
                'safety_penalty': sp_mpa_init, 
                'diag_penalty': dp_mpa_init,
                'fitness': f_mpa_init
            })
        return population

    def _get_random_node_on_path(self, path_nodes):
        return random.choice(path_nodes) if path_nodes else None

    def _get_levy_target_node(self, current_node, scale=1.0):
        sigma_numerator = math.gamma(1 + self.levy_beta) * math.sin(math.pi * self.levy_beta / 2)
        sigma_denominator = math.gamma((1 + self.levy_beta) / 2) * self.levy_beta * (2 ** ((self.levy_beta - 1) / 2))
        sigma = (sigma_numerator / sigma_denominator) ** (1 / self.levy_beta) if sigma_denominator > 1e-9 else 1.0
        u = random.normalvariate(0, sigma)
        v = random.normalvariate(0, 1)
        if abs(v) < 1e-9:
            v = 1e-9
        step_len = 0.05 * u / (abs(v) ** (1 / self.levy_beta)) * scale
        max_sensible_step_mpa_levy = max(self.rows, self.cols) * 0.5 
        step_len = np.clip(step_len, -max_sensible_step_mpa_levy, max_sensible_step_mpa_levy)
        angle = random.uniform(0, 2 * math.pi)
        dr_levy_target, dc_levy_target = int(round(step_len * math.sin(angle))), int(round(step_len * math.cos(angle)))
        target_r_levy_final, target_c_levy_final = max(0, min(self.rows - 1, current_node[0] + dr_levy_target)), max(0, min(self.cols - 1, current_node[1] + dc_levy_target))
        return (target_r_levy_final, target_c_levy_final)

    def _get_brownian_target_node(self, current_node, elite_node_on_path, scale=0.1):
        if random.random() < 0.7 and elite_node_on_path:
            dr_to_elite_brown, dc_to_elite_brown = elite_node_on_path[0] - current_node[0], elite_node_on_path[1] - current_node[1]
            dist_to_elite_pt_brown = math.sqrt(dr_to_elite_brown**2 + dc_to_elite_brown**2)
            if dist_to_elite_pt_brown > 1e-6:
                brownian_step_size_factor_brown = abs(random.normalvariate(0, 1))
                max_step_towards_elite_brown = min(dist_to_elite_pt_brown, max(1, int(round(scale * brownian_step_size_factor_brown * 5))))
                step_r_brown_final, step_c_brown_final = int(round(dr_to_elite_brown / dist_to_elite_pt_brown * max_step_towards_elite_brown)), int(round(dc_to_elite_brown / dist_to_elite_pt_brown * max_step_towards_elite_brown))
                target_r_brown_final, target_c_brown_final = current_node[0] + step_r_brown_final, current_node[1] + step_c_brown_final
            else: 
                return elite_node_on_path 
        else: 
            max_perturb_step_mpa_brown = max(1, int(round(max(self.rows, self.cols) * 0.1 * scale * abs(random.normalvariate(0,1)))))
            dr_brown_perturb_final, dc_brown_perturb_final = random.randint(-max_perturb_step_mpa_brown, max_perturb_step_mpa_brown), random.randint(-max_perturb_step_mpa_brown, max_perturb_step_mpa_brown)
            target_r_brown_final, target_c_brown_final = current_node[0] + dr_brown_perturb_final, current_node[1] + dc_brown_perturb_final
        target_r_brown_final, target_c_brown_final = max(0, min(self.rows - 1, target_r_brown_final)), max(0, min(self.cols - 1, target_c_brown_final))
        return (target_r_brown_final, target_c_brown_final)

    def _reconstruct_path_segment(self, path_to_modify_recon, elite_path_recon, start_modify_idx_recon, 
                                  is_levy_move_recon, step_scale_factor_recon):
        if not path_to_modify_recon or start_modify_idx_recon >= len(path_to_modify_recon) -1:
            return self._calculate_path_stats(path_to_modify_recon)
        current_node_to_modify_from_recon = path_to_modify_recon[start_modify_idx_recon]
        prefix_path_reconstruct = path_to_modify_recon[:start_modify_idx_recon+1]
        nodes_to_avoid_reconstruct = set(prefix_path_reconstruct[:-1])
        if is_levy_move_recon:
            intermediate_target_reconstruct = self._get_levy_target_node(current_node_to_modify_from_recon, scale=step_scale_factor_recon)
        else:
            elite_node_sample_reconstruct = self._get_random_node_on_path(elite_path_recon) if elite_path_recon else None
            intermediate_target_reconstruct = self._get_brownian_target_node(current_node_to_modify_from_recon, elite_node_sample_reconstruct, scale=step_scale_factor_recon)
        full_new_path_reconstruct = list(prefix_path_reconstruct)
        current_a_star_start_reconstruct = current_node_to_modify_from_recon
        if self._is_valid_and_not_obstacle(intermediate_target_reconstruct[0], intermediate_target_reconstruct[1]) and intermediate_target_reconstruct != current_a_star_start_reconstruct:
            path_to_intermediate_reconstruct, _ = self._a_star(current_a_star_start_reconstruct, intermediate_target_reconstruct, nodes_to_avoid_reconstruct.copy())
            if path_to_intermediate_reconstruct and len(path_to_intermediate_reconstruct) > 1:
                new_segment_part1_reconstruct = path_to_intermediate_reconstruct[1:]
                full_new_path_reconstruct.extend(new_segment_part1_reconstruct)
                current_a_star_start_reconstruct = intermediate_target_reconstruct
                for node_reconstruct in new_segment_part1_reconstruct: 
                    nodes_to_avoid_reconstruct.add(node_reconstruct)
        if current_a_star_start_reconstruct != self.target_node:
            path_to_target_final_reconstruct, _ = self._a_star(current_a_star_start_reconstruct, self.target_node, nodes_to_avoid_reconstruct.copy())
            if path_to_target_final_reconstruct and len(path_to_target_final_reconstruct) > 1:
                full_new_path_reconstruct.extend(path_to_target_final_reconstruct[1:])
        unique_path_reconstruct = []
        if full_new_path_reconstruct:
            unique_path_reconstruct.append(full_new_path_reconstruct[0])
            for i_up_mpa_recon in range(1, len(full_new_path_reconstruct)):
                if full_new_path_reconstruct[i_up_mpa_recon] != full_new_path_reconstruct[i_up_mpa_recon-1]:
                    unique_path_reconstruct.append(full_new_path_reconstruct[i_up_mpa_recon])
        if not unique_path_reconstruct or unique_path_reconstruct[0] != self.start_node or unique_path_reconstruct[-1] != self.target_node:
            return self._calculate_path_stats(path_to_modify_recon)
        return self._calculate_path_stats(unique_path_reconstruct) # Trả về 6 giá trị

    def solve_path_planning(self): 
        self.population.sort(key=lambda p_sort_solve: p_sort_solve['fitness'])
        elite_obj_solve_mpa = self.population[0].copy()
        
        self.best_path_overall = list(elite_obj_solve_mpa['path'])
        self.best_path_length_overall = elite_obj_solve_mpa['length']
        self.best_path_turns_overall = elite_obj_solve_mpa['turns']
        self.best_safety_penalty_overall = elite_obj_solve_mpa.get('safety_penalty', float('inf'))
        self.best_diag_penalty_overall = elite_obj_solve_mpa.get('diag_penalty', float('inf')) # << LƯU MỚI
        self.best_fitness_overall = elite_obj_solve_mpa['fitness']
        self.convergence_curve_data.append(self.best_fitness_overall if self.best_fitness_overall != float('inf') else None)

        for iter_num_solve_mpa_main in range(1, self.num_iterations + 1):
            self.population.sort(key=lambda p_sort_iter_main: p_sort_iter_main['fitness'])
            elite_obj_iter_mpa_main = self.population[0].copy() 
            ratio_iter_mpa_main = iter_num_solve_mpa_main / self.num_iterations
            CF_mpa_main = 0.0 if ratio_iter_mpa_main >= 1.0 else ( (1.0 - ratio_iter_mpa_main)**(2.0 * ratio_iter_mpa_main) if ratio_iter_mpa_main > 0 else 1.0)
            new_population_data_list_mpa_main = []

            if iter_num_solve_mpa_main <= self.num_iterations / 3:
                for i_p1_main in range(self.num_predators):
                    prey_i_p1_main = self.population[i_p1_main]
                    if not prey_i_p1_main['path'] or len(prey_i_p1_main['path']) <= 1: new_population_data_list_mpa_main.append(prey_i_p1_main.copy()); continue
                    start_modify_idx_p1_main = random.randint(0, len(prey_i_p1_main['path']) - 2) if len(prey_i_p1_main['path']) > 1 else 0
                    if random.random() < self.P_const:
                        p_n,l_n,t_n,sp_n,dp_n,f_n = self._reconstruct_path_segment(prey_i_p1_main['path'], elite_obj_iter_mpa_main['path'], start_modify_idx_p1_main, False, self.P_const)
                        new_population_data_list_mpa_main.append({'path':p_n,'length':l_n,'turns':t_n,'safety_penalty':sp_n,'diag_penalty':dp_n,'fitness':f_n})
                    else: new_population_data_list_mpa_main.append(prey_i_p1_main.copy())
            elif iter_num_solve_mpa_main <= 2 * self.num_iterations / 3:
                for i_p2_main in range(self.num_predators):
                    prey_i_p2_main = self.population[i_p2_main]
                    is_levy_p2_main = (i_p2_main < self.num_predators // 2)
                    path_to_mod_p2_main = prey_i_p2_main['path'] if is_levy_p2_main else elite_obj_iter_mpa_main['path']
                    ref_path_p2_main = elite_obj_iter_mpa_main['path'] if is_levy_p2_main else prey_i_p2_main['path']
                    scale_p2_main = self.P_const if is_levy_p2_main else self.P_const * CF_mpa_main
                    if not path_to_mod_p2_main or len(path_to_mod_p2_main) <=1: 
                        p_n,l_n,t_n,sp_n,dp_n,f_n = self._calculate_path_stats(path_to_mod_p2_main)
                        new_population_data_list_mpa_main.append({'path':p_n,'length':l_n,'turns':t_n,'safety_penalty':sp_n,'diag_penalty':dp_n,'fitness':f_n}); continue
                    start_idx_p2_main = random.randint(0, len(path_to_mod_p2_main) - 2) if len(path_to_mod_p2_main) > 1 else 0
                    if random.random() < (self.P_const if is_levy_p2_main else self.P_const * CF_mpa_main):
                        p_n,l_n,t_n,sp_n,dp_n,f_n = self._reconstruct_path_segment(path_to_mod_p2_main, ref_path_p2_main, start_idx_p2_main, is_levy_p2_main, scale_p2_main)
                        new_population_data_list_mpa_main.append({'path':p_n,'length':l_n,'turns':t_n,'safety_penalty':sp_n,'diag_penalty':dp_n,'fitness':f_n})
                    else: 
                        p_s,l_s,t_s,sp_s,dp_s,f_s = self._calculate_path_stats(path_to_mod_p2_main)
                        new_population_data_list_mpa_main.append({'path':p_s,'length':l_s,'turns':t_s,'safety_penalty':sp_s,'diag_penalty':dp_s,'fitness':f_s})
            else: # Phase 3
                for i_p3_main in range(self.num_predators):
                    elite_path_mod_p3_main = elite_obj_iter_mpa_main['path']; prey_ref_p3_main = self.population[i_p3_main]['path']
                    if not elite_path_mod_p3_main or len(elite_path_mod_p3_main) <= 1: 
                        p_n,l_n,t_n,sp_n,dp_n,f_n = self._calculate_path_stats(elite_path_mod_p3_main)
                        new_population_data_list_mpa_main.append({'path':p_n,'length':l_n,'turns':t_n,'safety_penalty':sp_n,'diag_penalty':dp_n,'fitness':f_n}); continue
                    start_idx_elite_p3_main = random.randint(0, len(elite_path_mod_p3_main) - 2) if len(elite_path_mod_p3_main) > 1 else 0
                    if random.random() < self.P_const * CF_mpa_main:
                        p_n,l_n,t_n,sp_n,dp_n,f_n = self._reconstruct_path_segment(elite_path_mod_p3_main, prey_ref_p3_main, start_idx_elite_p3_main, True, self.P_const * CF_mpa_main)
                        new_population_data_list_mpa_main.append({'path':p_n,'length':l_n,'turns':t_n,'safety_penalty':sp_n,'diag_penalty':dp_n,'fitness':f_n})
                    else: 
                        p_s,l_s,t_s,sp_s,dp_s,f_s = self._calculate_path_stats(elite_path_mod_p3_main)
                        new_population_data_list_mpa_main.append({'path':p_s,'length':l_s,'turns':t_s,'safety_penalty':sp_s,'diag_penalty':dp_s,'fitness':f_s})


            temp_pop_after_phases_mpa_main = []
            for i_mem_mpa_main in range(self.num_predators):
                if new_population_data_list_mpa_main[i_mem_mpa_main]['fitness'] < self.population[i_mem_mpa_main]['fitness']:
                    temp_pop_after_phases_mpa_main.append(new_population_data_list_mpa_main[i_mem_mpa_main])
                else: temp_pop_after_phases_mpa_main.append(self.population[i_mem_mpa_main].copy())
            
            self.population = []
            for ind_obj_current_mpa_main in temp_pop_after_phases_mpa_main:
                final_ind_next_iter_mpa_main = ind_obj_current_mpa_main.copy()
                if random.random() < self.FADs_rate:
                    if random.random() < CF_mpa_main:
                        rand_r_fad_mpa_main, rand_c_fad_mpa_main = random.randint(0,self.rows-1),random.randint(0,self.cols-1)
                        rand_inter_node_fad_mpa_main = (rand_r_fad_mpa_main,rand_c_fad_mpa_main)
                        if self._is_valid_and_not_obstacle(rand_r_fad_mpa_main,rand_c_fad_mpa_main):
                            p1_fad_s1_mpa_main,_=self._a_star(self.start_node,rand_inter_node_fad_mpa_main)
                            if p1_fad_s1_mpa_main:
                                p2_fad_s1_mpa_main,_=self._a_star(rand_inter_node_fad_mpa_main,self.target_node,set(p1_fad_s1_mpa_main[:-1]))
                                if p2_fad_s1_mpa_main:
                                    fad_p1_raw_mpa_main=p1_fad_s1_mpa_main+p2_fad_s1_mpa_main[1:]
                                    uniq_fad1_mpa_main=[fad_p1_raw_mpa_main[0]]+[fad_p1_raw_mpa_main[k_u1_main] for k_u1_main in range(1,len(fad_p1_raw_mpa_main)) if fad_p1_raw_mpa_main[k_u1_main]!=fad_p1_raw_mpa_main[k_u1_main-1]] if fad_p1_raw_mpa_main else []
                                    if uniq_fad1_mpa_main and uniq_fad1_mpa_main[-1]==self.target_node:
                                        _,l_fs1_m,t_fs1_m,sp_fs1_m,dp_fs1_m,f_fs1_m=self._calculate_path_stats(uniq_fad1_mpa_main)
                                        if f_fs1_m < final_ind_next_iter_mpa_main['fitness']:
                                            final_ind_next_iter_mpa_main={'path':uniq_fad1_mpa_main,'length':l_fs1_m,'turns':t_fs1_m,'safety_penalty':sp_fs1_m,'diag_penalty':dp_fs1_m,'fitness':f_fs1_m}
                    else:
                        fad_reinit_p_s2_mpa_main,_=self._generate_initial_path()
                        if fad_reinit_p_s2_mpa_main:
                            _,l_fs2_m,t_fs2_m,sp_fs2_m,dp_fs2_m,f_fs2_m=self._calculate_path_stats(fad_reinit_p_s2_mpa_main)
                            if f_fs2_m < final_ind_next_iter_mpa_main['fitness']:
                                final_ind_next_iter_mpa_main={'path':fad_reinit_p_s2_mpa_main,'length':l_fs2_m,'turns':t_fs2_m,'safety_penalty':sp_fs2_m,'diag_penalty':dp_fs2_m,'fitness':f_fs2_m}
                self.population.append(final_ind_next_iter_mpa_main)

            self.population.sort(key=lambda p_sort_final_mpa: p_sort_final_mpa['fitness'])
            current_iter_best_obj_mpa_main = self.population[0]

            if current_iter_best_obj_mpa_main['fitness'] < self.best_fitness_overall:
                self.best_fitness_overall = current_iter_best_obj_mpa_main['fitness']
                self.best_path_overall = list(current_iter_best_obj_mpa_main['path'])
                self.best_path_length_overall = current_iter_best_obj_mpa_main['length']
                self.best_path_turns_overall = current_iter_best_obj_mpa_main['turns']
                self.best_safety_penalty_overall = current_iter_best_obj_mpa_main.get('safety_penalty', float('inf'))
                self.best_diag_penalty_overall = current_iter_best_obj_mpa_main.get('diag_penalty', float('inf'))
            elif abs(current_iter_best_obj_mpa_main['fitness'] - self.best_fitness_overall) < 1e-9: # Tie-breaking
                # Ưu tiên length -> turns -> safety -> diag_penalty
                if current_iter_best_obj_mpa_main['length'] < self.best_path_length_overall:
                    self._update_best_overall(current_iter_best_obj_mpa_main)
                elif abs(current_iter_best_obj_mpa_main['length']-self.best_path_length_overall)<1e-9 and \
                     current_iter_best_obj_mpa_main['turns'] < self.best_path_turns_overall:
                    self._update_best_overall(current_iter_best_obj_mpa_main)
                elif abs(current_iter_best_obj_mpa_main['length']-self.best_path_length_overall)<1e-9 and \
                     abs(current_iter_best_obj_mpa_main['turns']-self.best_path_turns_overall)<1e-9 and \
                     current_iter_best_obj_mpa_main.get('safety_penalty',float('inf')) < self.best_safety_penalty_overall:
                    self._update_best_overall(current_iter_best_obj_mpa_main)
                elif abs(current_iter_best_obj_mpa_main['length']-self.best_path_length_overall)<1e-9 and \
                     abs(current_iter_best_obj_mpa_main['turns']-self.best_path_turns_overall)<1e-9 and \
                     abs(current_iter_best_obj_mpa_main.get('safety_penalty',float('inf')) - self.best_safety_penalty_overall) < 1e-9 and \
                     current_iter_best_obj_mpa_main.get('diag_penalty',float('inf')) < self.best_diag_penalty_overall:
                    self._update_best_overall(current_iter_best_obj_mpa_main)


            self.convergence_curve_data.append(self.best_fitness_overall if self.best_fitness_overall != float('inf') else (self.convergence_curve_data[-1] if self.convergence_curve_data and self.convergence_curve_data[-1] is not None else None))
            if iter_num_solve_mpa_main%10==0 or iter_num_solve_mpa_main==1 or iter_num_solve_mpa_main==self.num_iterations:
                print(f"MPA Iter {iter_num_solve_mpa_main}/{self.num_iterations}: "
                      f"IterBest L={current_iter_best_obj_mpa_main['length']:.2f},T={current_iter_best_obj_mpa_main['turns']},SP={current_iter_best_obj_mpa_main.get('safety_penalty',0.0):.2f},DP={current_iter_best_obj_mpa_main.get('diag_penalty',0.0):.2f},Fit={current_iter_best_obj_mpa_main['fitness']:.2f}; "
                      f"OverallBest L={self.best_path_length_overall:.2f},T={self.best_path_turns_overall},SP={self.best_safety_penalty_overall:.2f},DP={self.best_diag_penalty_overall:.2f},Fit={self.best_fitness_overall:.2f}")
        
        if self.best_path_overall: print(f"\nMPA Solved: Fitness={self.best_fitness_overall:.2f} (L={self.best_path_length_overall:.2f},T={self.best_path_turns_overall},SP={self.best_safety_penalty_overall:.2f},DP={self.best_diag_penalty_overall:.2f})")
        else: print("\nMPA: No solution found.")
        return self.best_path_overall, self.best_path_length_overall, self.best_path_turns_overall, self.best_safety_penalty_overall, self.best_diag_penalty_overall, self.best_fitness_overall
    
    def _update_best_overall(self, new_best_obj):
        """Hàm tiện ích để cập nhật các thuộc tính best_overall."""
        self.best_fitness_overall = new_best_obj['fitness']
        self.best_path_overall = list(new_best_obj['path'])
        self.best_path_length_overall = new_best_obj['length']
        self.best_path_turns_overall = new_best_obj['turns']
        self.best_safety_penalty_overall = new_best_obj.get('safety_penalty', float('inf'))
        self.best_diag_penalty_overall = new_best_obj.get('diag_penalty', float('inf'))

    def plot_convergence_curve(self):
        import matplotlib.pyplot as plt
        valid_conv_data_mpa_plot=[fit for fit in self.convergence_curve_data if fit is not None]
        if valid_conv_data_mpa_plot:
            plt.figure(); plt.plot(valid_conv_data_mpa_plot); plt.title("MPA Convergence (Path Planning w/ Safety & Diag Rule)"); plt.xlabel("Iteration"); plt.ylabel("Best Fitness"); plt.grid(True); plt.show()
        else: print("No MPA convergence data to plot.")