# pso_solver.py
import numpy as np
import random
from helper import BasePathfinder, is_valid_and_not_obstacle
from astar import AStarSolver # PSO cũng dùng A* để nối waypoints
from env import START_NODE_VAL, TARGET_NODE_VAL

class PSOSolver(BasePathfinder):
    def __init__(self, grid,
                 num_iterations, num_particles, num_waypoints_per_particle,
                 w, c1, c2, # Tham số PSO
                 turn_penalty_factor=0.1, safety_penalty_factor=0.05, min_safe_distance=1.5,
                 allow_diagonal_moves=True,
                 restrict_diagonal_near_obstacle_policy=True,
                 diagonal_obstacle_penalty_value=1000.0):

        start_nodes_found = np.argwhere(grid == START_NODE_VAL)
        target_nodes_found = np.argwhere(grid == TARGET_NODE_VAL)
        if not start_nodes_found.size > 0: raise ValueError("PSO: Start node not found.")
        if not target_nodes_found.size > 0: raise ValueError("PSO: Target node not found.")
        start_node = tuple(start_nodes_found[0])
        target_node = tuple(target_nodes_found[0])

        super().__init__(grid, start_node, target_node,
                         turn_penalty_factor, safety_penalty_factor, min_safe_distance,
                         allow_diagonal_moves,
                         restrict_diagonal_near_obstacle_policy, # Chính sách của PSO
                         diagonal_obstacle_penalty_value)

        self.num_iterations = num_iterations
        self.num_particles = num_particles
        self.num_waypoints = num_waypoints_per_particle
        self.w, self.c1, self.c2 = w, c1, c2
        self.max_vel = max(1.0, 0.15 * max(self.rows, self.cols)) # Vận tốc tối đa cho waypoint

        self.path_connector = AStarSolver(
            grid=self.grid,
            turn_penalty_factor=0, safety_penalty_factor=0, min_safe_distance=0,
            allow_diagonal_moves=self.allow_diagonal_moves,
            restrict_diagonal_near_obstacle_policy=self.restrict_diagonal_near_obstacle_policy, # QUAN TRỌNG
            diagonal_obstacle_penalty_value=0
        )
        
        self.particles = [] # list of dicts
        self.gbest_particle_data = {'fitness': float('inf'), 'path': [], 'position': []}


    def _generate_random_waypoint(self): # Giống GA
        while True:
            r = random.uniform(0, self.rows - 1) # Vị trí có thể là float ban đầu
            c = random.uniform(0, self.cols - 1)
            # Kiểm tra tính hợp lệ của ô (làm tròn) khi cần, hoặc để A* xử lý
            # if is_valid_and_not_obstacle(int(round(r)), int(round(c)), self.grid, self.rows, self.cols):
            return [r, c] # Trả về list [r,c] để dễ thao tác với vận tốc

    def _reconstruct_path_from_position(self, position_waypoints_float):
        if not position_waypoints_float : # Nối thẳng start -> target
            path_stats = self.path_connector.solve(self.start_node, self.target_node)
            return path_stats[0]

        position_waypoints_int = [(int(round(wp[0])), int(round(wp[1]))) for wp in position_waypoints_float]
        
        full_path = [self.start_node]
        current_start = self.start_node
        nodes_in_path_so_far = {self.start_node}

        for waypoint_int in position_waypoints_int:
            # Đảm bảo waypoint nằm trong lưới sau khi làm tròn
            wp_clamped = (max(0, min(self.rows - 1, waypoint_int[0])), 
                          max(0, min(self.cols - 1, waypoint_int[1])))

            path_segment_stats = self.path_connector.solve(
                start_node_override=current_start, target_node_override=wp_clamped,
                nodes_to_avoid=nodes_in_path_so_far - {current_start, wp_clamped}
            )
            segment = path_segment_stats[0]
            if not segment or (len(segment) == 1 and current_start != wp_clamped): return []
            full_path.extend(segment[1:])
            nodes_in_path_so_far.update(segment[1:])
            current_start = wp_clamped
        
        final_segment_stats = self.path_connector.solve(
            start_node_override=current_start, target_node_override=self.target_node,
            nodes_to_avoid=nodes_in_path_so_far - {current_start, self.target_node}
        )
        final_segment = final_segment_stats[0]
        if not final_segment or (len(final_segment) == 1 and current_start != self.target_node): return []
        full_path.extend(final_segment[1:])

        if not full_path: return []
        unique_path = [full_path[0]]
        for i in range(1, len(full_path)):
            if full_path[i] != full_path[i-1]: unique_path.append(full_path[i])
        return unique_path


    def _initialize_particles(self):
        self.particles = []
        attempts = 0
        max_total_attempts = self.num_particles * 20

        while len(self.particles) < self.num_particles and attempts < max_total_attempts:
            attempts += 1
            position = [self._generate_random_waypoint() for _ in range(self.num_waypoints)]
            velocity = [[random.uniform(-self.max_vel/5, self.max_vel/5) for _ in range(2)] for _ in range(self.num_waypoints)]

            path = self._reconstruct_path_from_position(position)
            if path and path[0] == self.start_node and path[-1] == self.target_node:
                _, l, t, sp, dp, fitness = self._calculate_stats_for_path(path)
                
                particle_data = {
                    'position': position, 'velocity': velocity,
                    'pbest_position': list(p_dim[:] for p_dim in position), # Deep copy
                    'pbest_fitness': fitness, 'pbest_path': list(path),
                    'pbest_stats': {'l':l, 't':t, 'sp':sp, 'dp':dp},
                    'current_fitness': fitness, 'current_path': list(path),
                    'current_stats': {'l':l, 't':t, 'sp':sp, 'dp':dp}
                }
                self.particles.append(particle_data)

                if fitness < self.gbest_particle_data['fitness']:
                    self.gbest_particle_data = {
                        'fitness': fitness, 'path': list(path), 
                        'position': list(p_dim[:] for p_dim in position),
                        'length':l, 'turns':t, 'safety_penalty':sp, 'diag_penalty':dp
                    }
        
        if not self.particles and self.num_waypoints > 0: # Thử fallback
            path_direct = self._reconstruct_path_from_position([])
            if path_direct and path_direct[0] == self.start_node and path_direct[-1] == self.target_node:
                _, l, t, sp, dp, fitness = self._calculate_stats_for_path(path_direct)
                dummy_pos = [[0.0,0.0]]*self.num_waypoints
                dummy_vel = [[0.0,0.0]]*self.num_waypoints
                particle_data = {
                    'position': dummy_pos, 'velocity': dummy_vel,
                    'pbest_position': list(p[:] for p in dummy_pos), 'pbest_fitness': fitness, 'pbest_path': list(path_direct),
                    'pbest_stats': {'l':l, 't':t, 'sp':sp, 'dp':dp},
                    'current_fitness': fitness, 'current_path': list(path_direct),
                    'current_stats': {'l':l, 't':t, 'sp':sp, 'dp':dp}
                }
                self.particles.append(particle_data)
                if fitness < self.gbest_particle_data['fitness']:
                     self.gbest_particle_data = {'fitness': fitness, 'path': list(path_direct), 'position': list(p[:] for p in dummy_pos),
                                                'length':l, 'turns':t, 'safety_penalty':sp, 'diag_penalty':dp}
                print("PSO Warning: Population init failed, used a direct A* path as one particle.")

        if not self.particles:
            print("PSO Error: Could not initialize any valid particles.")
            # Tạo hạt giả để tránh lỗi
            dummy_path_stats = self._calculate_stats_for_path([])
            self.gbest_particle_data = {'fitness': float('inf'), 'path': [], 'position': [], 
                                        'length': float('inf'), 'turns':0, 'safety_penalty':0, 'diag_penalty':0}
            self.particles = [{'position': [[0.0,0.0]]*self.num_waypoints, 'velocity': [[0.0,0.0]]*self.num_waypoints,
                               'pbest_position': [[0.0,0.0]]*self.num_waypoints, 'pbest_fitness': float('inf'), 
                               'pbest_path': [], 'pbest_stats': {},
                               'current_fitness': float('inf'), 'current_path': [], 'current_stats': {}}] * self.num_particles
            return False
        
        while len(self.particles) < self.num_particles and self.particles:
            self.particles.append(random.choice(self.particles).copy())
        return True

    def solve(self):
        if self.num_waypoints == 0:
            print("PSO running with 0 waypoints (effectively A*).")
            path = self._reconstruct_path_from_position([])
            stats = self._calculate_stats_for_path(path)
            self.gbest_particle_data = {'path': stats[0], 'fitness': stats[5], 'length': stats[1], 'turns': stats[2], 'safety_penalty':stats[3], 'diag_penalty':stats[4], 'position':[]}
            self.convergence_curve.append(stats[5])
            return stats

        if not self._initialize_particles():
            print("PSO: Particle initialization failed completely. Returning empty result.")
            return [], float('inf'), 0, 0.0, 0.0, float('inf')

        self.convergence_curve.append(self.gbest_particle_data['fitness'])

        for iteration in range(self.num_iterations):
            for p_idx, p_data in enumerate(self.particles):
                new_velocity_dims = []
                new_position_dims = []

                for dim_idx in range(self.num_waypoints): # Từng waypoint
                    # Cập nhật vận tốc cho từng thành phần (r, c) của waypoint
                    vel_r = self.w * p_data['velocity'][dim_idx][0] + \
                            self.c1 * random.random() * (p_data['pbest_position'][dim_idx][0] - p_data['position'][dim_idx][0]) + \
                            self.c2 * random.random() * (self.gbest_particle_data['position'][dim_idx][0] - p_data['position'][dim_idx][0])
                    vel_c = self.w * p_data['velocity'][dim_idx][1] + \
                            self.c1 * random.random() * (p_data['pbest_position'][dim_idx][1] - p_data['position'][dim_idx][1]) + \
                            self.c2 * random.random() * (self.gbest_particle_data['position'][dim_idx][1] - p_data['position'][dim_idx][1])
                    
                    vel_r = np.clip(vel_r, -self.max_vel, self.max_vel)
                    vel_c = np.clip(vel_c, -self.max_vel, self.max_vel)
                    new_velocity_dims.append([vel_r, vel_c])

                    # Cập nhật vị trí
                    pos_r = p_data['position'][dim_idx][0] + vel_r
                    pos_c = p_data['position'][dim_idx][1] + vel_c
                    
                    # Giữ waypoint trong biên (không làm tròn ở đây, A* sẽ làm tròn)
                    pos_r = np.clip(pos_r, 0, self.rows - 1)
                    pos_c = np.clip(pos_c, 0, self.cols - 1)
                    new_position_dims.append([pos_r, pos_c])
                
                self.particles[p_idx]['velocity'] = new_velocity_dims
                self.particles[p_idx]['position'] = new_position_dims

                # Đánh giá vị trí mới
                current_path = self._reconstruct_path_from_position(self.particles[p_idx]['position'])
                if current_path and current_path[0] == self.start_node and current_path[-1] == self.target_node:
                    _, l, t, sp, dp, fitness = self._calculate_stats_for_path(current_path)
                    self.particles[p_idx]['current_fitness'] = fitness
                    self.particles[p_idx]['current_path'] = current_path
                    self.particles[p_idx]['current_stats'] = {'l':l, 't':t, 'sp':sp, 'dp':dp}

                    if fitness < self.particles[p_idx]['pbest_fitness']:
                        self.particles[p_idx]['pbest_fitness'] = fitness
                        self.particles[p_idx]['pbest_position'] = list(dim_p[:] for dim_p in self.particles[p_idx]['position'])
                        self.particles[p_idx]['pbest_path'] = list(current_path)
                        self.particles[p_idx]['pbest_stats'] = {'l':l, 't':t, 'sp':sp, 'dp':dp}

                        if fitness < self.gbest_particle_data['fitness']:
                            self.gbest_particle_data['fitness'] = fitness
                            self.gbest_particle_data['position'] = list(dim_p[:] for dim_p in self.particles[p_idx]['position'])
                            self.gbest_particle_data['path'] = list(current_path)
                            self.gbest_particle_data['length'] = l
                            self.gbest_particle_data['turns'] = t
                            self.gbest_particle_data['safety_penalty'] = sp
                            self.gbest_particle_data['diag_penalty'] = dp
            
            self.convergence_curve.append(self.gbest_particle_data['fitness'])
            if (iteration + 1) % 10 == 0 or iteration == 0 or iteration == self.num_iterations - 1:
                 best = self.gbest_particle_data
                 print(f"PSO Iter {iteration+1}/{self.num_iterations}: GBestFit={best['fitness']:.2f} "
                      f"(L:{best.get('length',0):.1f}, T:{best.get('turns',0)}, "
                      f"SP:{best.get('safety_penalty',0):.2f}, DP:{best.get('diag_penalty',0):.2f})")

        res = self.gbest_particle_data
        return (res['path'], res.get('length', float('inf')), res.get('turns', float('inf')), 
                res.get('safety_penalty', float('inf')), res.get('diag_penalty', float('inf')), res['fitness'])