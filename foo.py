import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math

# Constants for grid
FREE_SPACE = 0
OBSTACLE = 1
START_NODE_VAL = 2
TARGET_NODE_VAL = 3

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
                    if denominator < 1e-9: # Node is on S or T, or S and T are same
                        if self._distance(current_node, self.start_node) < 1e-6 or \
                           self._distance(current_node, self.target_node) < 1e-6 :
                            factor = 1.0 # Max pheromone at S and T
                        else: # Should not happen if dsT is not 0
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
        return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] # 8 directions

    def _apply_orientation_heuristic_filter(self, current_node_P, tabu_list):
        cr, cc = current_node_P

        delta_r_ST = self.target_node[0] - self.start_node[0]
        delta_c_ST = self.target_node[1] - self.start_node[1]

        oriented_neighbors = []
        all_moves = self._get_all_potential_neighbor_moves()

        for dr_PPn, dc_PPn in all_moves: # dr_PPn, dc_PPn là vector P->Pn
            nr, nc = cr + dr_PPn, cc + dc_PPn
            if not self._is_valid_and_not_tabu(nr, nc, tabu_list):
                continue


            valid_orientation = True
            # Kiểm tra hướng cột (x)
            if delta_c_ST > 0 and dc_PPn < 0: valid_orientation = False
            if delta_c_ST < 0 and dc_PPn > 0: valid_orientation = False
            # Kiểm tra hướng hàng (y) - nhớ rằng dr_numpy > 0 là đi xuống
            if delta_r_ST > 0 and dr_PPn < 0: valid_orientation = False # S->T đi xuống, P->Pn không được đi lên
            if delta_r_ST < 0 and dr_PPn > 0: valid_orientation = False # S->T đi lên, P->Pn không được đi xuống
            
            if valid_orientation:
                oriented_neighbors.append((nr, nc))

        if not oriented_neighbors: # Nếu không có theo định hướng S->T, thử hướng P->T
            delta_r_PT = self.target_node[0] - cr
            delta_c_PT = self.target_node[1] - cc
            for dr_PPn, dc_PPn in all_moves:
                nr, nc = cr + dr_PPn, cc + dc_PPn
                if not self._is_valid_and_not_tabu(nr, nc, tabu_list):
                    continue
                valid_orientation_pt = True
                if delta_c_PT > 0 and dc_PPn < 0: valid_orientation_pt = False
                if delta_c_PT < 0 and dc_PPn > 0: valid_orientation_pt = False
                if delta_r_PT > 0 and dr_PPn < 0: valid_orientation_pt = False
                if delta_r_PT < 0 and dr_PPn > 0: valid_orientation_pt = False
                if valid_orientation_pt:
                    oriented_neighbors.append((nr, nc))

        if not oriented_neighbors: # Fallback cuối cùng: tất cả lân cận hợp lệ
             for dr, dc in all_moves:
                nr, nc = cr + dr, cc + dc
                if self._is_valid_and_not_tabu(nr, nc, tabu_list):
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

        if self.dist_S_to_T_overall < 1e-9: # Tránh chia cho 0 nếu S trùng T
             h_adaptive = self.wh_min # Hoặc một giá trị mặc định khác
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
            if K_total_iter - k0_thresh_iter < 1e-6 and K_total_iter - k0_thresh_iter > -1e-6 : # K == k0
                 q0_val = self.q0_initial # Hoặc một xử lý khác
            else:  
                 q0_val = ((K_total_iter - k_curr_iter) / K_total_iter) * self.q0_initial # Giảm từ q0_initial về 0

        else: 
           
            q0_val = self.q0_initial * (k0_thresh_iter / K_total_iter) + \
                     ((k_curr_iter - k0_thresh_iter) / (K_total_iter - k0_thresh_iter + 1e-9)) * \
                     (self.q0_initial * (1 - k0_thresh_iter / K_total_iter) / 2.0)

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
                elif abs(attr - max_attractiveness) < 1e-9: # Gần bằng
                    best_candidates.append(available_neighbors[i])
            return random.choice(best_candidates) if best_candidates else None

        else:
            sum_attractiveness = sum(attractiveness_values)
            if sum_attractiveness < 1e-9:
                return random.choice(available_neighbors) if available_neighbors else None
            
            probabilities = [attr / sum_attractiveness for attr in attractiveness_values]
            try:
                # Đảm bảo tổng xác suất là 1
                if abs(sum(probabilities) - 1.0) > 1e-6:
                    probabilities = [p / sum(probabilities) for p in probabilities] # Chuẩn hóa lại
                selected_idx = np.random.choice(len(available_neighbors), p=probabilities)
                return available_neighbors[selected_idx]
            except ValueError as e:
                # print(f"Lỗi xác suất: {e}, P={probabilities}, sum={sum(probabilities)}")
                return random.choice(available_neighbors) if available_neighbors else None

    def _count_turns(self, path):
        if len(path) < 3:
            return 0
        turns = 0
        for i in range(len(path) - 2):
            p1, p2, p3 = path[i], path[i+1], path[i+2]
            # Vector P1->P2
            dr1 = p2[0] - p1[0]
            dc1 = p2[1] - p1[1]
            # Vector P2->P3
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
        max_steps = self.rows * self.cols * 2 # Tăng giới hạn bước
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
        # Bay hơi
        self.pheromone_matrix *= (1.0 - self.rho)
        
        # Lắng đọng
        for path, length, _ in all_ant_paths_this_iter: # Bỏ qua số lần rẽ ở đây
            if length != float('inf') and path and length > 1e-6 : # Thêm kiểm tra length > 0
                deposit_amount = self.Q / length
                for node in path:
                    # Chỉ lắng đọng trên các ô không phải vật cản
                    if self.grid[node] != OBSTACLE:
                         self.pheromone_matrix[node] += deposit_amount
        
        # Sử dụng độ dài tốt nhất toàn cục đã tìm thấy cho đến nay để ổn định τmax, τmin
        current_best_len_for_tau = self.best_path_length_overall
        if current_best_len_for_tau == float('inf'):
            # Nếu chưa có đường đi tốt, sử dụng một ước lượng
            current_best_len_for_tau = float(self.rows + self.cols) # Ước lượng sơ bộ

        if current_best_len_for_tau < 1e-6: # Tránh chia cho 0
            current_best_len_for_tau = 1e-6

        tau_max_mmas = (1.0 / (1.0 - self.rho)) * (1.0 / current_best_len_for_tau)
    
        tau_max_to_use = tau_max_mmas
        tau_min_to_use = tau_max_to_use / (2.0 * self.cols) # Tỷ lệ MMAS phổ biến

        # Giới hạn pheromone
        mask_free_space = (self.grid != OBSTACLE)
        self.pheromone_matrix[mask_free_space] = np.clip(
            self.pheromone_matrix[mask_free_space],
            tau_min_to_use,
            tau_max_to_use
        )
        self.pheromone_matrix[self.grid == OBSTACLE] = 1e-9 # Đảm bảo vật cản có pheromone rất thấp


    def solve_path_planning(self):
        for iter_num in range(1, self.num_iterations + 1):
            iter_paths_details = [] # (path, length, turns)
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
                elif abs(length - iter_best_length) < 1e-9 and turns < iter_best_turns: # Cùng độ dài, chọn ít rẽ hơn
                    iter_best_path = path
                    iter_best_turns = turns


            # Cập nhật tốt nhất toàn cục
            # Ưu tiên độ dài, sau đó là số lần rẽ
            if iter_best_length < self.best_path_length_overall:
                self.best_path_length_overall = iter_best_length
                self.best_path_overall = iter_best_path
                self.best_path_turns_overall = iter_best_turns
            elif abs(iter_best_length - self.best_path_length_overall) < 1e-9: # Độ dài bằng nhau
                if iter_best_turns < self.best_path_turns_overall:
                    self.best_path_overall = iter_best_path # Vẫn cập nhật path
                    self.best_path_turns_overall = iter_best_turns

            # Cập nhật pheromone
            # Sử dụng iter_best_length (độ dài tốt nhất của vòng lặp này) cho tau_max/min
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

    # ... (Các hàm visualize giữ nguyên) ...
    @staticmethod
    def _visualize_grid_and_multiple_paths(grid_data, start_node_pos, target_node_pos, paths_dict, title="So sánh Đường đi"):
        """Hiển thị lưới và các đường đi."""
        rows_viz, cols_viz = grid_data.shape
        display_grid_viz = np.copy(grid_data).astype(float)
        
        colors_viz_map = {
            FREE_SPACE: 'white', 
            OBSTACLE: 'black',
            START_NODE_VAL: '#FF4500',
            TARGET_NODE_VAL: '#1E90FF'
        }
        
        unique_vals_in_grid = np.unique(display_grid_viz)
        active_colors_list = []
        bounds_list = [-0.5]
        
        for val in sorted(unique_vals_in_grid):
            if val in colors_viz_map:
                active_colors_list.append(colors_viz_map[val])
                bounds_list.append(val + 0.5)
                
        bounds_list = sorted(list(set(bounds_list)))
        
        if not active_colors_list: active_colors_list = ['white']
        if not bounds_list or len(bounds_list) < 2: bounds_list = [-0.5, 0.5]
            
        cmap_viz = mcolors.ListedColormap(active_colors_list)
        norm_viz = mcolors.BoundaryNorm(bounds_list, cmap_viz.N)
        
        plt.figure(figsize=(max(8, cols_viz/1.8), max(8, rows_viz/1.8))) # Tăng kích thước
        plt.imshow(display_grid_viz, cmap=cmap_viz, norm=norm_viz, origin='upper', 
                  interpolation='nearest')
        plt.title(title, fontsize=16, fontweight='bold')
        
        plt.xticks(np.arange(cols_viz), fontsize=10)
        plt.yticks(np.arange(rows_viz), fontsize=10)
        plt.gca().set_xticks(np.arange(-.5, cols_viz, 1), minor=True)
        plt.gca().set_yticks(np.arange(-.5, rows_viz, 1), minor=True)
        plt.grid(True, which='minor', color='lightgray', linewidth=0.5) # Mỏng hơn
        plt.grid(True, which='major', color='darkgray', linewidth=0.5)
        
        legend_handles = []
        for algo_name, (path_coords, path_color, path_linestyle) in paths_dict.items():
            if path_coords and len(path_coords) > 1:
                path_rows_viz = [p[0] for p in path_coords]
                path_cols_viz = [p[1] for p in path_coords]
                line, = plt.plot(path_cols_viz, path_rows_viz, marker='o', color=path_color, # 'o' thay vì '.'
                                 linestyle=path_linestyle, markersize=5, linewidth=2.0, alpha=0.85, label=algo_name)
                legend_handles.append(line)
        if legend_handles:
            plt.legend(handles=legend_handles, fontsize=11, loc='best') # 'best' location
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _visualize_pheromone_matrix_static(grid_data, pheromone_data, title="Mức Pheromone"):
        rows_viz, cols_viz = grid_data.shape
        plt.figure(figsize=(max(6,cols_viz/2.0), max(6,rows_viz/2.0)))
        masked_pheromones_viz = np.ma.masked_where(grid_data == OBSTACLE, pheromone_data)
        
        # Chuẩn hóa pheromone để hiển thị tốt hơn nếu dải giá trị quá lớn
        if np.nanmax(masked_pheromones_viz) > 0: # Tránh lỗi nếu toàn NaN
            log_pheromones = np.log1p(masked_pheromones_viz - np.nanmin(masked_pheromones_viz)) # Log transform
            im = plt.imshow(log_pheromones, cmap='viridis', origin='upper', interpolation='nearest')
        else:
            im = plt.imshow(masked_pheromones_viz, cmap='viridis', origin='upper', interpolation='nearest')

        plt.colorbar(im, label="Cường độ Pheromone (Log Scale)")
        plt.title(title, fontsize=14)
        plt.xticks(fontsize=9); plt.yticks(fontsize=9)
        plt.show()
        
    def visualize_pheromone_matrix(self, title="Mức Pheromone MAACO"):
        MAACO._visualize_pheromone_matrix_static(self.grid, self.pheromone_matrix, title)

    @staticmethod
    def _plot_convergence_curve_static(convergence_data, algo_name="Thuật toán", color='dodgerblue'):
        plt.figure(figsize=(9,5.5)) # Tăng kích thước
        valid_iterations_viz = [i+1 for i, l_val in enumerate(convergence_data) if l_val is not None and l_val != float('inf')]
        valid_lengths_viz = [l_val for l_val in convergence_data if l_val is not None and l_val != float('inf')]
        if valid_lengths_viz:
            plt.plot(valid_iterations_viz, valid_lengths_viz, marker='o', linestyle='-', markersize=5, color=color, linewidth=1.8) # Dày hơn
            plt.xlabel("Số lần lặp", fontsize=12)
            plt.ylabel("Độ dài đường đi tốt nhất", fontsize=12)
            plt.title(f"Đường cong hội tụ của {algo_name}", fontsize=14, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.6) # Kiểu lưới khác
            plt.xticks(fontsize=10); plt.yticks(fontsize=10)
            # Đặt giới hạn trục y hợp lý
            if len(valid_lengths_viz) > 1:
                 min_y = min(valid_lengths_viz) * 0.98
                 max_y = max(valid_lengths_viz) * 1.02
                 if max_y > min_y: # Tránh lỗi khi tất cả giá trị bằng nhau
                     plt.ylim(min_y, max_y)

            plt.tight_layout()
            plt.show()
        else:
            print(f"{algo_name}: Không có dữ liệu độ dài đường đi hợp lệ để vẽ đồ thị hội tụ.")

    def plot_convergence_curve(self):
        MAACO._plot_convergence_curve_static(self.convergence_curve_data, "MAACO", color='orangered')


if __name__ == '__main__':
    grid_fig7_layout_data = [
        [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0],
        [0,1,0,0,0,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0], [0,1,1,0,0,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0], [0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0,0],
        [1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0], [0,0,0,0,0,0,0,1,1,0,1,1,1,1,0,0,0,0,0,0],
        [0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0], [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0], [0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0],
        [1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0], [1,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,0,0], [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,0], [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]
    current_test_grid_base_fig7 = np.array(grid_fig7_layout_data)
    start_row_fig7, start_col_fig7 = 0, 0      
    target_row_fig7, target_col_fig7 = 19, 19  
    
    current_test_grid_base_fig7[start_row_fig7, start_col_fig7] = START_NODE_VAL
    current_test_grid_base_fig7[target_row_fig7, target_col_fig7] = TARGET_NODE_VAL

    maaco_params_fig13 = {
        "num_ants": 50, "num_iterations": 100, "alpha": 1.0, "beta": 7.0,
        "rho": 0.2, "Q": 2.5, # Q=1 theo tối ưu hóa của Fig.13
        "a_turn_coef": 1.0, "wh_max": 0.9, "wh_min": 0.2, 
        "k_h_adaptive": 0.9, # Giả định giá trị hợp lý
        "q0_initial": 0.5,
        "C0_initial_pheromone": 0.1 # Giá trị cơ sở
    }
    
    print("--- Running MAACO (Params from Fig 13 for Fig 7 Environment) ---")
    maaco_solver_fig7 = MAACO(grid=np.copy(current_test_grid_base_fig7), **maaco_params_fig13)
    maaco_path_fig7, maaco_len_fig7, maaco_turns_fig7 = maaco_solver_fig7.solve_path_planning()

    if maaco_path_fig7:
        maaco_solver_fig7.plot_convergence_curve()
        MAACO._visualize_grid_and_multiple_paths(
            grid_data=current_test_grid_base_fig7,
            start_node_pos=maaco_solver_fig7.start_node,
            target_node_pos=maaco_solver_fig7.target_node,
            paths_dict={f"MAACO (L:{maaco_len_fig7:.2f}, T:{maaco_turns_fig7})": (maaco_path_fig7, 'orangered', '-')},
            title="MAACO Path on Fig 7 Environment (Params from Fig 13)"
        )


    grid_map_fig13_base = np.array([
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,3], # Hàng 0
    [0,1,1,0,1,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0], # Hàng 1
    [0,1,1,0,1,0,1,1,1,0,0,1,0,0,0,1,1,1,0,0], # Hàng 2
    [0,1,0,0,0,1,1,1,1,1,0,0,1,1,0,1,1,1,0,0], # Hàng 3
    [0,0,0,0,0,0,1,1,1,0,1,0,0,0,0,1,1,1,0,0], # Hàng 4
    [0,0,0,1,0,0,1,1,1,0,0,1,0,0,0,1,1,1,0,0], # Hàng 5
    [0,1,1,1,0,0,1,1,1,0,0,0,0,1,0,1,1,1,0,0], # Hàng 6
    [0,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0], # Hàng 7
    [0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0], # Hàng 8
    [0,1,1,1,0,0,1,1,0,0,1,1,1,1,0,0,1,0,0,0], # Hàng 9
    [0,1,1,1,0,0,0,1,1,0,1,1,1,1,1,0,0,0,0,0], # Hàng 10
    [0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0], # Hàng 11
    [0,0,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,0], # Hàng 12
    [0,0,0,1,0,0,0,0,0,1,0,1,1,1,0,1,1,1,1,0], # Hàng 13
    [0,0,0,1,1,0,0,1,0,0,0,1,1,1,0,1,1,1,1,0], # Hàng 14
    [0,0,1,1,0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,0], # Hàng 15
    [0,0,1,1,1,0,1,1,1,0,1,1,0,0,0,0,1,1,1,0], # Hàng 16
    [0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,1,1,0], # Hàng 17
    [0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,1,0,0,0,0], # Hàng 18
    [2,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0]  
])
    start_row_fig13, start_col_fig13 = 19, 0      # S=(0,0) như Fig 7
    target_row_fig13, target_col_fig13 = 0, 19  # T=(19,19) như Fig 7
    maaco_params_fig = {
        "num_ants": 50, "num_iterations": 100, "alpha": 1.0, "beta": 7.0,
        "rho": 0.2, "Q": 2.5, # Q=1 theo tối ưu hóa của Fig.13
        "a_turn_coef": 1.0, "wh_max": 0.9, "wh_min": 0.2, 
        "k_h_adaptive": 0.9, # Giả định giá trị hợp lý
        "q0_initial": 0.5,
        "C0_initial_pheromone": 0.1 # Giá trị cơ sở
    }
    
    grid_map_fig13_base[start_row_fig13, start_col_fig13] = START_NODE_VAL
    grid_map_fig13_base[target_row_fig13, target_col_fig13] = TARGET_NODE_VAL
    print("--- Running MAACO (Params from Fig 13 for Fig 7 Environment) ---")
    maaco_solver_fig13 = MAACO(grid=np.copy(grid_map_fig13_base), **maaco_params_fig)
    maaco_path_fig13, maaco_len_fig13, maaco_turns_fig13 = maaco_solver_fig13.solve_path_planning()
    # Fig 13 kết quả: Length = 32.133, Turns = 8, Iter = 13

    if maaco_path_fig13:
        maaco_solver_fig13.plot_convergence_curve()
        MAACO._visualize_grid_and_multiple_paths(
            grid_data=grid_map_fig13_base,
            start_node_pos=maaco_solver_fig13.start_node,
            target_node_pos=maaco_solver_fig13.target_node,
            paths_dict={f"MAACO (L:{maaco_len_fig13:.2f}, T:{maaco_turns_fig13})": (maaco_path_fig13, 'orangered', '-')},
            title="MAACO Path on Fig 13 Environment (Params from Fig 13)"
        )

    grid_map_from_image = np.array([
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
])
    start_row_fig18, start_col_fig18 = 0, 0      
    target_row_fig18, target_col_fig18 = 19, 19  
    maaco_params_fig18 = {
        "num_ants": 50, "num_iterations": 100, "alpha": 1.0, "beta": 7.0,
        "rho": 0.2, "Q": 2.5, 
        "a_turn_coef": 1.0, "wh_max": 0.9, "wh_min": 0.2, 
        "k_h_adaptive": 0.9,
        "q0_initial": 0.5,
        "C0_initial_pheromone": 0.1 
    }
    
    grid_map_from_image[start_row_fig18, start_col_fig18] = START_NODE_VAL
    grid_map_from_image[target_row_fig18, target_col_fig18] = TARGET_NODE_VAL
    print("--- Running MAACO  ---")
    maaco_solver_fig18 = MAACO(grid=np.copy(grid_map_from_image), **maaco_params_fig18)
    maaco_path_fig18, maaco_len_fig18, maaco_turns_fig18 = maaco_solver_fig18.solve_path_planning()

    if maaco_path_fig18:
        maaco_solver_fig18.plot_convergence_curve()
        MAACO._visualize_grid_and_multiple_paths(
            grid_data=grid_map_from_image,
            start_node_pos=maaco_solver_fig18.start_node,
            target_node_pos=maaco_solver_fig18.target_node,
            paths_dict={f"MAACO (L:{maaco_len_fig18:.2f}, T:{maaco_turns_fig18})": (maaco_path_fig18, 'orangered', '-')},
            title="MAACO Path"
        )
    