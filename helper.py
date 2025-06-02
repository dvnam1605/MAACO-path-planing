# helpers.py
import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
from env import OBSTACLE # Đảm bảo env.py có định nghĩa OBSTACLE

def distance_euclidean(node1, node2):
    """Tính khoảng cách Euclidean."""
    dr = node1[0] - node2[0]
    dc = node1[1] - node2[1]
    return math.sqrt(dr**2 + dc**2)

def is_valid_and_not_obstacle(r, c, grid, rows, cols):
    """Kiểm tra một nút có hợp lệ và không phải vật cản."""
    return 0 <= r < rows and 0 <= c < cols and grid[r, c] != OBSTACLE

def get_valid_neighbors(r, c, grid, rows, cols,
                        allow_diagonal_moves=True,
                        restrict_diagonal_corner_cutting=True, # Tham số quan trọng
                        exclude_nodes=None):
    """
    Lấy các hàng xóm hợp lệ.
    'restrict_diagonal_corner_cutting' = True: cấm tuyệt đối đi chéo cắt góc.
    """
    neighbors = []
    if exclude_nodes is None:
        exclude_nodes = set()

    possible_moves = [
        (0, 1, False), (0, -1, False), (1, 0, False), (-1, 0, False), # Thẳng
    ]
    if allow_diagonal_moves:
        possible_moves.extend([
            (1, 1, True), (1, -1, True), (-1, 1, True), (-1, -1, True) # Chéo
        ])

    for dr_nn, dc_nn, is_diag_move in possible_moves:
        nr, nc = r + dr_nn, c + dc_nn

        if not is_valid_and_not_obstacle(nr, nc, grid, rows, cols) or (nr, nc) in exclude_nodes:
            continue

        can_move = True
        if is_diag_move and restrict_diagonal_corner_cutting:
            obstacle_at_corner1 = not is_valid_and_not_obstacle(r + dr_nn, c, grid, rows, cols)
            obstacle_at_corner2 = not is_valid_and_not_obstacle(r, c + dc_nn, grid, rows, cols)
            if obstacle_at_corner1 or obstacle_at_corner2:
                can_move = False

        if can_move:
            neighbors.append((nr, nc))
    return neighbors

def heuristic_euclidean(a, b):
    return distance_euclidean(a,b)

def count_turns(path):
    if len(path) < 3: return 0
    turns = 0
    for i in range(len(path) - 2):
        dr1, dc1 = path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]
        dr2, dc2 = path[i+2][0] - path[i+1][0], path[i+2][1] - path[i+1][1]
        if dr1 != dr2 or dc1 != dc2: turns += 1
    return turns

def calculate_path_safety_penalty(path, min_safe_dist, obstacle_nodes, grid, rows, cols):
    if not path or not obstacle_nodes.size > 0: return 0.0
    total_safety_penalty = 0.0
    for r_node, c_node in path:
        min_dist_to_obs_sq = float('inf')
        if obstacle_nodes.size > 0:
            diffs = obstacle_nodes - np.array([r_node, c_node])
            distances_sq = np.sum(diffs**2, axis=1)
            min_dist_to_obs_sq = np.min(distances_sq)
        min_dist_to_obs = math.sqrt(min_dist_to_obs_sq) if min_dist_to_obs_sq != float('inf') else float('inf')
        if min_dist_to_obs < min_safe_dist:
            penalty_for_point = (min_safe_dist - min_dist_to_obs)**2
            total_safety_penalty += penalty_for_point
    return total_safety_penalty / len(path) if path else 0.0

def calculate_diagonal_obstacle_penalty(path, grid, rows, cols,
                                        algorithm_restricts_diag_near_obstacle_policy,
                                        diagonal_obstacle_penalty_value):
    if not path or len(path) < 2 or not algorithm_restricts_diag_near_obstacle_policy:
        return 0.0
    diag_penalty = 0.0
    for i in range(len(path) - 1):
        curr_r, curr_c = path[i]; next_r, next_c = path[i+1]
        dr, dc = next_r - curr_r, next_c - curr_c
        if abs(dr) == 1 and abs(dc) == 1: # Bước đi chéo
            obstacle_at_corner1 = not is_valid_and_not_obstacle(next_r, curr_c, grid, rows, cols)
            obstacle_at_corner2 = not is_valid_and_not_obstacle(curr_r, next_c, grid, rows, cols)
            if obstacle_at_corner1 or obstacle_at_corner2:
                diag_penalty += diagonal_obstacle_penalty_value
    return diag_penalty

def calculate_path_stats(path, grid, rows, cols, obstacle_nodes,
                         turn_penalty_factor,
                         safety_penalty_factor, min_safe_distance,
                         algorithm_wants_to_restrict_diag_corner_cutting, # Chính sách của thuật toán
                         diagonal_obstacle_penalty_value, # Giá trị phạt của thuật toán
                         allow_diagonal_moves_overall=True):
    if not path or len(path) == 0:
        return [], float('inf'), 0, 0.0, 0.0, float('inf')
    length = sum(distance_euclidean(path[i], path[i+1]) for i in range(len(path)-1)) if len(path) > 1 else 0
    turns = count_turns(path)
    safety_p = calculate_path_safety_penalty(path, min_safe_distance, obstacle_nodes, grid, rows, cols)
    diag_p = calculate_diagonal_obstacle_penalty(path, grid, rows, cols,
                                                 algorithm_wants_to_restrict_diag_corner_cutting,
                                                 diagonal_obstacle_penalty_value)
    fitness = length + turn_penalty_factor * turns + safety_penalty_factor * safety_p + diag_p
    return path, length, turns, safety_p, diag_p, fitness

class BasePathfinder:
    def __init__(self, grid, start_node, target_node,
                 turn_penalty_factor, safety_penalty_factor, min_safe_distance,
                 allow_diagonal_moves,
                 restrict_diagonal_near_obstacle_policy, # Chính sách của thuật toán này
                 diagonal_obstacle_penalty_value):       # Giá trị phạt của thuật toán này
        self.grid = np.array(grid, dtype=int)
        self.rows, self.cols = self.grid.shape
        self.start_node = start_node
        self.target_node = target_node
        self.obstacle_nodes = np.argwhere(self.grid == OBSTACLE)

        self.turn_penalty_factor = turn_penalty_factor
        self.safety_penalty_factor = safety_penalty_factor
        self.min_safe_distance = min_safe_distance
        self.allow_diagonal_moves = allow_diagonal_moves # Cho phép đi chéo nói chung
        
        # Chính sách cụ thể của thuật toán này đối với việc đi chéo và phạt
        self.restrict_diagonal_near_obstacle_policy = restrict_diagonal_near_obstacle_policy
        self.diagonal_obstacle_penalty_value = diagonal_obstacle_penalty_value
        
        self.convergence_curve = [] # Dữ liệu cho đường cong hội tụ

    def _calculate_stats_for_path(self, path):
        """Tính toán các chỉ số cho một đường đi dựa trên chính sách của thuật toán này."""
        return calculate_path_stats(
            path, self.grid, self.rows, self.cols, self.obstacle_nodes,
            self.turn_penalty_factor,
            self.safety_penalty_factor, self.min_safe_distance,
            self.restrict_diagonal_near_obstacle_policy, # Sử dụng chính sách của thuật toán
            self.diagonal_obstacle_penalty_value,        # Sử dụng giá trị phạt của thuật toán
            self.allow_diagonal_moves
        )

    def plot_convergence_curve(self, title_prefix="Algorithm"):
        """Vẽ đường cong hội tụ (nếu có)."""
        valid_data = [fit for fit in self.convergence_curve if fit is not None and fit != float('inf')]
        if valid_data:
            plt.figure()
            plt.plot(valid_data)
            plt.title(f"{title_prefix} Convergence Curve")
            plt.xlabel("Iteration / Evaluation")
            plt.ylabel("Best Fitness")
            plt.grid(True)
            # plt.show() # Sẽ được gọi ở main script
        else:
            print(f"No valid convergence data to plot for {title_prefix}.")