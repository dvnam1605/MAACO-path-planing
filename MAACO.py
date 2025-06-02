import numpy as np
import random
import math
from env import OBSTACLE, START_NODE_VAL, TARGET_NODE_VAL # Import constants
# Import visualization functions that will be called by MAACO's methods
from visualization import visualize_pheromone_matrix as visualize_pheromone_static
from visualization import plot_convergence_curve as plot_convergence_static


class MAACO:
    def __init__(self, grid, num_ants, num_iterations,
                 alpha, beta, rho, Q, # Standard ACO params
                 a_turn_coef, wh_max, wh_min, k_h_adaptive, q0_initial, # MAACO specific params
                 C0_initial_pheromone=0.1): # Initial pheromone concentration
        self.grid = np.array(grid, dtype=int)
        self.rows, self.cols = self.grid.shape
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.a_turn_coef = a_turn_coef
        self.wh_max = wh_max
        self.wh_min = wh_min
        self.k_h_adaptive = k_h_adaptive
        self.q0_initial = q0_initial
        self.k0_iter_threshold_factor = 0.7 # k0 = 0.7K

        self.C0_base = C0_initial_pheromone

        start_nodes_found = np.argwhere(self.grid == START_NODE_VAL)
        target_nodes_found = np.argwhere(self.grid == TARGET_NODE_VAL)

        if not start_nodes_found.size > 0:
            raise ValueError("MAACO: Start node not found.")
        if not target_nodes_found.size > 0:
            raise ValueError("MAACO: Target node not found.")

        self.start_node = tuple(start_nodes_found[0])
        self.target_node = tuple(target_nodes_found[0])

        self.dist_S_to_T_overall = self._distance(self.start_node, self.target_node) # dsT
        if self.dist_S_to_T_overall < 1e-9:
            self.dist_S_to_T_overall = 1e-9

        self.pheromone_matrix = self._initialize_pheromones_maaco()
        self.dist_to_target_matrix = self._precompute_dist_to_target()

        self.best_path_overall = []
        self.best_path_length_overall = float('inf')
        self.best_path_turns_overall = float('inf')
        self.convergence_curve_data = []

    def _distance(self, node1, node2):
        return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

    def _initialize_pheromones_maaco(self):
        pheromones = np.zeros_like(self.grid, dtype=float)
        dsT = self.dist_S_to_T_overall

        for r in range(self.rows):
            for c in range(self.cols):
                current_node = (r, c)
                if self.grid[r, c] == OBSTACLE:
                    pheromones[r, c] = 1e-9 # Very small for obstacles
                else:
                    dsi = self._distance(self.start_node, current_node)
                    diT = self._distance(current_node, self.target_node)

                    denominator = dsi + diT
                    if denominator < 1e-9:
                        if self._distance(current_node, self.start_node) < 1e-6 or \
                           self._distance(current_node, self.target_node) < 1e-6 :
                            factor = 1.0
                        else:
                            factor = 0.1
                    else:
                        factor = dsT / denominator

                    pheromones[r, c] = factor * self.C0_base
                    if pheromones[r, c] < 1e-9:
                        pheromones[r, c] = 1e-9
        return pheromones

    def _precompute_dist_to_target(self):
        dist_matrix = np.zeros_like(self.grid, dtype=float)
        for r in range(self.rows):
            for c in range(self.cols):
                dist_matrix[r,c] = self._distance((r,c), self.target_node)
        return dist_matrix

    def _is_valid_and_not_tabu(self, r, c, tabu_list):
        return (0 <= r < self.rows and 0 <= c < self.cols and
                self.grid[r,c] != OBSTACLE and (r,c) not in tabu_list)

    def _get_all_potential_neighbor_moves(self):
        return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def _is_diagonal_corner_cut(self, r_curr, c_curr, r_next, c_next):
        """
        Kiểm tra xem một bước đi chéo từ (r_curr, c_curr) đến (r_next, c_next)
        có "cắt góc" của một vật cản hay không.
        Điều này xảy ra nếu một trong hai ô (r_next, c_curr) hoặc (r_curr, c_next) là vật cản.
        Hàm này giả định (r_curr, c_curr) và (r_next, c_next) là các tọa độ hợp lệ
        và bước đi là chéo.
        """
        # Các ô trung gian tạo thành "góc"
        # Ô 1: (r_next, c_curr)
        # Ô 2: (r_curr, c_next)

        # Kiểm tra xem ô (r_next, c_curr) có phải là vật cản không
        if self.grid[r_next, c_curr] == OBSTACLE:
            return True # Cắt góc qua vật cản (r_next, c_curr)

        # Kiểm tra xem ô (r_curr, c_next) có phải là vật cản không
        if self.grid[r_curr, c_next] == OBSTACLE:
            return True # Cắt góc qua vật cản (r_curr, c_next)

        return False # Không cắt góc

    def _apply_orientation_heuristic_filter(self, current_node_P, tabu_list):
        cr, cc = current_node_P # Current row, current col
        delta_r_ST = self.target_node[0] - self.start_node[0]
        delta_c_ST = self.target_node[1] - self.start_node[1]
        oriented_neighbors = []
        all_moves = self._get_all_potential_neighbor_moves()

        # --- Helper function for filtering moves ---
        def filter_moves_logic(moves_to_check, use_ST_orientation_vector):
            local_neighbors_list = []
            for dr_move, dc_move in moves_to_check: # dr_move, dc_move are deltas
                nr, nc = cr + dr_move, cc + dc_move # next row, next col

                if not self._is_valid_and_not_tabu(nr, nc, tabu_list):
                    continue

                # <<< --- LOGIC CẤM ĐI CHÉO CẮT GÓC VẬT CẢN --- >>>
                is_diagonal = (abs(dr_move) == 1 and abs(dc_move) == 1)
                if is_diagonal:
                    if self._is_diagonal_corner_cut(cr, cc, nr, nc):
                        continue # Bỏ qua nước đi này vì nó cắt góc vật cản
                # <<< --- KẾT THÚC LOGIC CẤM ĐI CHÉO --- >>>

                passes_orientation_check = True
                if use_ST_orientation_vector: # Lọc dựa trên vector Start-Target
                    if delta_c_ST > 0 and dc_move < 0: passes_orientation_check = False
                    if delta_c_ST < 0 and dc_move > 0: passes_orientation_check = False
                    if delta_r_ST > 0 and dr_move < 0: passes_orientation_check = False
                    if delta_r_ST < 0 and dr_move > 0: passes_orientation_check = False
                else: # Lọc dựa trên vector Current-Target
                    delta_r_PT = self.target_node[0] - cr
                    delta_c_PT = self.target_node[1] - cc
                    if delta_c_PT > 0 and dc_move < 0: passes_orientation_check = False
                    if delta_c_PT < 0 and dc_move > 0: passes_orientation_check = False
                    if delta_r_PT > 0 and dr_move < 0: passes_orientation_check = False
                    if delta_r_PT < 0 and dr_move > 0: passes_orientation_check = False

                if passes_orientation_check:
                    local_neighbors_list.append((nr, nc))
            return local_neighbors_list
        # --- End of helper function ---

        # Chiến lược 1: Lọc theo hướng Start -> Target
        oriented_neighbors = filter_moves_logic(all_moves, use_ST_orientation_vector=True)

        # Chiến lược 2: Nếu không có, lọc theo hướng Current -> Target
        if not oriented_neighbors:
            oriented_neighbors = filter_moves_logic(all_moves, use_ST_orientation_vector=False)

        # Chiến lược 3: Nếu vẫn không có, lấy tất cả các nước đi hợp lệ (đã được lọc cấm cắt góc)
        if not oriented_neighbors:
             for dr, dc in all_moves:
                nr, nc = cr + dr, cc + dc
                if self._is_valid_and_not_tabu(nr, nc, tabu_list):
                    is_diag_fallback = (abs(dr) == 1 and abs(dc) == 1)
                    if is_diag_fallback:
                        if self._is_diagonal_corner_cut(cr, cc, nr, nc):
                            continue
                    oriented_neighbors.append((nr, nc))
        return oriented_neighbors


    def _calculate_turn_penalty_factor_c_i(self, ant_path_history, next_node_candidate):
        if len(ant_path_history) < 2:
            return 0
        prev_node = ant_path_history[-2]
        current_node = ant_path_history[-1]
        dr_prev = current_node[0] - prev_node[0]
        dc_prev = current_node[1] - prev_node[1]
        dr_next = next_node_candidate[0] - current_node[0]
        dc_next = next_node_candidate[1] - current_node[1]
        if dr_prev != dr_next or dc_prev != dc_next:
            return 1
        return 0

    def _calculate_improved_heuristic_eta_prime(self, current_node_for_heuristic, next_node_candidate, ant_path_history_for_turn_calc):
        dsj = self._distance(self.start_node, next_node_candidate)
        djT = self.dist_to_target_matrix[next_node_candidate]
        if self.dist_S_to_T_overall < 1e-9:
             h_adaptive = self.wh_min
        else:
            exp_term = math.exp(-self.k_h_adaptive * djT / self.dist_S_to_T_overall)
            h_adaptive = self.wh_max - (self.wh_max - self.wh_min) * exp_term
        g_adaptive = 1.0 - h_adaptive
        c_i_turn_factor = self._calculate_turn_penalty_factor_c_i(
            ant_path_history_for_turn_calc, next_node_candidate)
        denominator = g_adaptive * dsj + h_adaptive * djT + self.a_turn_coef * c_i_turn_factor
        denominator = max(denominator, 1e-9)
        return 1.0 / denominator

    def _calculate_adaptive_q0(self, current_iteration_num):
        K_total_iter = self.num_iterations
        k_curr_iter = current_iteration_num
        k0_thresh_iter = self.k0_iter_threshold_factor * K_total_iter
        if k_curr_iter < k0_thresh_iter:
            if abs(K_total_iter - k0_thresh_iter) < 1e-6 :
                 q0_val = self.q0_initial
            else:
                 q0_val = ((K_total_iter - k_curr_iter) / K_total_iter) * self.q0_initial
        else:
            q0_at_k0 = ((K_total_iter - k0_thresh_iter) / K_total_iter) * self.q0_initial
            q0_val = q0_at_k0 + \
                     ((k_curr_iter - k0_thresh_iter) / (K_total_iter - k0_thresh_iter + 1e-9)) * \
                     (self.q0_initial * (1 - (K_total_iter - k0_thresh_iter) / K_total_iter) / 2.0)
        return min(max(q0_val, 0.01), 0.99)

    def _select_next_node_with_MAACO_rules(self, current_node, available_neighbors, ant_path_history, current_iteration_num):
        if not available_neighbors:
            return None
        q0_adaptive_val = self._calculate_adaptive_q0(current_iteration_num)
        random_num_for_q = random.random()
        attractiveness_values = []
        for neighbor_node in available_neighbors:
            pheromone_val_tau = self.pheromone_matrix[neighbor_node]
            heuristic_val_eta_prime = self._calculate_improved_heuristic_eta_prime(
                current_node, neighbor_node, ant_path_history)
            attractiveness = (pheromone_val_tau ** self.alpha) * (heuristic_val_eta_prime ** self.beta)
            attractiveness_values.append(attractiveness)

        if random_num_for_q <= q0_adaptive_val:
            max_attractiveness = -1
            best_candidates = []
            for i, attr in enumerate(attractiveness_values):
                if attr > max_attractiveness:
                    max_attractiveness = attr
                    best_candidates = [available_neighbors[i]]
                elif abs(attr - max_attractiveness) < 1e-9:
                    best_candidates.append(available_neighbors[i])
            return random.choice(best_candidates) if best_candidates else None
        else:
            sum_attractiveness = sum(attractiveness_values)
            if sum_attractiveness < 1e-9:
                return random.choice(available_neighbors) if available_neighbors else None
            probabilities = [attr / sum_attractiveness for attr in attractiveness_values]
            try:
                if abs(sum(probabilities) - 1.0) > 1e-6:
                    probabilities = [p / sum(probabilities) for p in probabilities] # Normalize
                selected_idx = np.random.choice(len(available_neighbors), p=probabilities)
                return available_neighbors[selected_idx]
            except ValueError: # Fallback if probabilities are still problematic
                return random.choice(available_neighbors) if available_neighbors else None

    def _count_turns(self, path):
        if len(path) < 3:
            return 0
        turns = 0
        for i in range(len(path) - 2):
            p1, p2, p3 = path[i], path[i+1], path[i+2]
            dr1 = p2[0] - p1[0]
            dc1 = p2[1] - p1[1]
            dr2 = p3[0] - p2[0]
            dc2 = p3[1] - p2[1]
            if dr1 != dr2 or dc1 != dc2:
                turns += 1
        return turns

    def _construct_ant_solution_maaco(self, ant_id, current_iteration_num):
        current_pos = self.start_node
        ant_path = [self.start_node]
        ant_path_len = 0.0
        tabu_nodes = {current_pos}
        max_steps = self.rows * self.cols * 2
        num_steps = 0
        while current_pos != self.target_node and num_steps < max_steps:
            candidate_nodes = self._apply_orientation_heuristic_filter(current_pos, tabu_nodes)
            if not candidate_nodes:
                return [], float('inf'), float('inf')
            next_node = self._select_next_node_with_MAACO_rules(
                current_pos, candidate_nodes, ant_path, current_iteration_num)
            if next_node is None:
                return [], float('inf'), float('inf')
            ant_path_len += self._distance(current_pos, next_node)
            current_pos = next_node
            ant_path.append(current_pos)
            tabu_nodes.add(current_pos)
            num_steps += 1
        if current_pos == self.target_node:
            num_turns = self._count_turns(ant_path)
            return ant_path, ant_path_len, num_turns
        else:
            return [], float('inf'), float('inf')

    def _update_pheromone_trails_maaco(self, all_ant_paths_this_iter, best_len_this_iter_for_tau_calc):
        self.pheromone_matrix *= (1.0 - self.rho)
        for path, length, _ in all_ant_paths_this_iter:
            if length != float('inf') and path and length > 1e-6 :
                deposit_amount = self.Q / length
                for node in path:
                    if self.grid[node] != OBSTACLE:
                         self.pheromone_matrix[node] += deposit_amount
        current_best_len_for_tau = self.best_path_length_overall
        if current_best_len_for_tau == float('inf'):
            current_best_len_for_tau = float(self.rows + self.cols)
        if current_best_len_for_tau < 1e-6:
            current_best_len_for_tau = 1e-6
        tau_max_mmas = (1.0 / (1.0 - self.rho)) * (1.0 / current_best_len_for_tau)
        # tau_max_to_use = tau_max_mmas # Original line from your code
        # Consider C0_base as an upper bound as well if needed, or MAACO specific tau_max if paper defines one.
        # For MMAS-style, tau_max is usually not additionally capped by C0_base after calculation.
        tau_max_to_use = tau_max_mmas

        tau_min_to_use = tau_max_to_use / (2.0 * max(self.cols, self.rows, 1)) # Avoid division by zero for 1xN grids
        # tau_min_to_use = max(tau_min_to_use, 1e-10) # Ensure tau_min is a very small positive number

        mask_free_space = (self.grid != OBSTACLE)
        self.pheromone_matrix[mask_free_space] = np.clip(
            self.pheromone_matrix[mask_free_space],
            tau_min_to_use,
            tau_max_to_use
        )
        self.pheromone_matrix[self.grid == OBSTACLE] = 1e-9

    def solve_path_planning(self):
        for iter_num in range(1, self.num_iterations + 1):
            iter_paths_details = []
            iter_best_length = float('inf')
            iter_best_path = []
            iter_best_turns = float('inf')
            for ant_idx in range(self.num_ants):
                path, length, turns = self._construct_ant_solution_maaco(ant_idx, iter_num)
                iter_paths_details.append((path, length, turns))
                if length < iter_best_length:
                    iter_best_length = length
                    iter_best_path = path
                    iter_best_turns = turns
                elif abs(length - iter_best_length) < 1e-9 and turns < iter_best_turns:
                    iter_best_path = path
                    iter_best_turns = turns

            if iter_best_length < self.best_path_length_overall:
                self.best_path_length_overall = iter_best_length
                self.best_path_overall = iter_best_path
                self.best_path_turns_overall = iter_best_turns
            elif abs(iter_best_length - self.best_path_length_overall) < 1e-9:
                if iter_best_turns < self.best_path_turns_overall:
                    self.best_path_overall = iter_best_path
                    self.best_path_turns_overall = iter_best_turns
            self._update_pheromone_trails_maaco(iter_paths_details, iter_best_length)
            self.convergence_curve_data.append(
                self.best_path_length_overall if self.best_path_length_overall != float('inf') else None
            )
            if iter_num % 10 == 0 or iter_num == 1 or iter_num == self.num_iterations:
                print(f"MAACO Iter {iter_num}/{self.num_iterations}: "
                      f"Iter Best L={iter_best_length:.2f}, T={iter_best_turns}, "
                      f"Overall Best L={self.best_path_length_overall:.2f}, T={self.best_path_turns_overall}")
        if self.best_path_overall:
            print(f"\nMAACO Solved: Length={self.best_path_length_overall:.2f}, Turns={self.best_path_turns_overall}")
        else:
            print("\nMAACO: No solution found.")
        return self.best_path_overall, self.best_path_length_overall, self.best_path_turns_overall

    def visualize_pheromone_matrix(self, title="Mức Pheromone MAACO"):
        visualize_pheromone_static(self.grid, self.pheromone_matrix, title)

    def plot_convergence_curve(self):
        plot_convergence_static(self.convergence_curve_data, "MAACO", color='orangered')