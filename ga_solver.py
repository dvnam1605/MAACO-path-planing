# ga_solver.py
import numpy as np
import random
from helper import BasePathfinder, is_valid_and_not_obstacle
from astar import AStarSolver # GA dùng A* để nối waypoints
from env import START_NODE_VAL, TARGET_NODE_VAL

class GASolver(BasePathfinder):
    def __init__(self, grid,
                 num_generations, population_size, num_waypoints_per_chromosome,
                 mutation_rate, crossover_rate, tournament_size=3,
                 turn_penalty_factor=0.1, safety_penalty_factor=0.05, min_safe_distance=1.5,
                 allow_diagonal_moves=True, 
                 restrict_diagonal_near_obstacle_policy=True,
                 diagonal_obstacle_penalty_value=1000.0):

        start_nodes_found = np.argwhere(grid == START_NODE_VAL)
        target_nodes_found = np.argwhere(grid == TARGET_NODE_VAL)
        if not start_nodes_found.size > 0: raise ValueError("GA: Start node not found.")
        if not target_nodes_found.size > 0: raise ValueError("GA: Target node not found.")
        start_node = tuple(start_nodes_found[0])
        target_node = tuple(target_nodes_found[0])
        
        super().__init__(grid, start_node, target_node,
                         turn_penalty_factor, safety_penalty_factor, min_safe_distance,
                         allow_diagonal_moves,
                         restrict_diagonal_near_obstacle_policy, # Chính sách của GA
                         diagonal_obstacle_penalty_value)

        self.num_generations = num_generations
        self.population_size = population_size
        self.num_waypoints = num_waypoints_per_chromosome
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

        # A* dùng để nối các waypoints PHẢI tuân thủ chính sách của GA
        self.path_connector = AStarSolver(
            grid=self.grid,
            turn_penalty_factor=0, safety_penalty_factor=0, min_safe_distance=0, # Không quan trọng khi chỉ nối điểm
            allow_diagonal_moves=self.allow_diagonal_moves,
            restrict_diagonal_near_obstacle_policy=self.restrict_diagonal_near_obstacle_policy, # QUAN TRỌNG
            diagonal_obstacle_penalty_value=0 # Không phạt ở bước này, A* phải tự tránh
        )
        self.population = [] 
        self.best_solution_overall = {'fitness': float('inf'), 'path': []}

    def _generate_random_waypoint(self):
        while True:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            if is_valid_and_not_obstacle(r, c, self.grid, self.rows, self.cols):
                return (r, c)

    def _create_chromosome(self):
        return [self._generate_random_waypoint() for _ in range(self.num_waypoints)]

    def _reconstruct_path_from_chromosome(self, chromosome):
        if not chromosome: # Không có waypoint, nối thẳng start -> target
            path_stats = self.path_connector.solve(self.start_node, self.target_node)
            return path_stats[0] # Trả về list các node

        full_path = [self.start_node]
        current_start = self.start_node
        nodes_in_path_so_far = {self.start_node}

        for waypoint in chromosome:
            path_segment_stats = self.path_connector.solve(
                start_node_override=current_start,
                target_node_override=waypoint,
                nodes_to_avoid=nodes_in_path_so_far - {current_start, waypoint} # Tránh lặp lại quá nhiều
            )
            segment = path_segment_stats[0]
            if not segment or (len(segment) == 1 and current_start != waypoint): return [] # Không nối được
            full_path.extend(segment[1:])
            nodes_in_path_so_far.update(segment[1:])
            current_start = waypoint
        
        final_segment_stats = self.path_connector.solve(
            start_node_override=current_start,
            target_node_override=self.target_node,
            nodes_to_avoid=nodes_in_path_so_far - {current_start, self.target_node}
        )
        final_segment = final_segment_stats[0]
        if not final_segment or (len(final_segment) == 1 and current_start != self.target_node): return []
        full_path.extend(final_segment[1:])
        
        # Loại bỏ các nút trùng lặp liên tiếp
        if not full_path: return []
        unique_path = [full_path[0]]
        for i in range(1, len(full_path)):
            if full_path[i] != full_path[i-1]: unique_path.append(full_path[i])
        return unique_path

    def _initialize_population(self):
        self.population = []
        attempts = 0
        max_total_attempts = self.population_size * 20 # Giới hạn tổng số lần thử
        
        while len(self.population) < self.population_size and attempts < max_total_attempts:
            attempts += 1
            chromosome = self._create_chromosome()
            path = self._reconstruct_path_from_chromosome(chromosome)
            
            if path and path[0] == self.start_node and path[-1] == self.target_node:
                # Tính stats dựa trên chính sách của GA
                _, l, t, sp, dp, fitness = self._calculate_stats_for_path(path)
                self.population.append({'chromosome': chromosome, 'path': path, 'fitness': fitness, 
                                        'length':l, 'turns':t, 'safety_penalty':sp, 'diag_penalty':dp})
        
        if not self.population and self.num_waypoints > 0 : # Thử tạo 1 cá thể không có waypoint nếu thất bại
             path_direct = self._reconstruct_path_from_chromosome([])
             if path_direct and path_direct[0] == self.start_node and path_direct[-1] == self.target_node:
                _, l, t, sp, dp, fitness = self._calculate_stats_for_path(path_direct)
                self.population.append({'chromosome': [], 'path': path_direct, 'fitness': fitness, 
                                        'length':l, 'turns':t, 'safety_penalty':sp, 'diag_penalty':dp})
                print("GA Warning: Population init failed, used a direct A* path as one individual.")


        if not self.population: # Nếu vẫn không có gì, đây là vấn đề
            print("GA Error: Could not initialize any valid individuals.")
            # Tạo cá thể giả để tránh lỗi, nhưng đây là dấu hiệu xấu
            dummy_path_stats = self._calculate_stats_for_path([])
            self.population = [{'chromosome': [], 'path': [], 'fitness': float('inf'), 
                                'length': float('inf'), 'turns':0, 'safety_penalty':0, 'diag_penalty':0}] * self.population_size
            return False
        
        # Điền đầy quần thể nếu cần bằng cách sao chép các cá thể tốt nhất
        while len(self.population) < self.population_size and self.population:
            self.population.append(random.choice(self.population).copy()) # Thêm bản sao ngẫu nhiên
        
        self.population.sort(key=lambda x: x['fitness'])
        return True


    def _selection(self): # Tournament
        selected_parents = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
            winner = min(tournament, key=lambda x: x['fitness'])
            selected_parents.append(winner)
        return selected_parents

    def _crossover(self, p1_chrom, p2_chrom):
        if random.random() < self.crossover_rate and self.num_waypoints > 0:
            # Single-point crossover
            point = random.randint(1, self.num_waypoints - 1) if self.num_waypoints > 1 else 0
            if point > 0 :
                c1_chrom = p1_chrom[:point] + p2_chrom[point:]
                c2_chrom = p2_chrom[:point] + p1_chrom[point:]
                return c1_chrom, c2_chrom
        return list(p1_chrom), list(p2_chrom)

    def _mutate(self, chromosome):
        if not chromosome: return [] # Không đột biến nếu không có waypoint
        mutated_chrom = list(chromosome)
        for i in range(len(mutated_chrom)):
            if random.random() < self.mutation_rate:
                mutated_chrom[i] = self._generate_random_waypoint()
        return mutated_chrom

    def solve(self):
        if self.num_waypoints == 0: # Chạy như A* nếu không có waypoint
            print("GA running with 0 waypoints (effectively A*).")
            path = self._reconstruct_path_from_chromosome([])
            stats = self._calculate_stats_for_path(path)
            self.best_solution_overall = {'path': stats[0], 'fitness': stats[5], 'length': stats[1], 'turns': stats[2], 'safety_penalty':stats[3], 'diag_penalty':stats[4]}
            self.convergence_curve.append(stats[5])
            return stats

        if not self._initialize_population():
             print("GA: Population initialization failed completely. Returning empty result.")
             return [], float('inf'), 0, 0.0, 0.0, float('inf')

        self.best_solution_overall = self.population[0].copy()
        self.convergence_curve.append(self.best_solution_overall['fitness'])

        for gen in range(self.num_generations):
            new_population_individuals = []
            
            # Elitism: giữ lại cá thể tốt nhất
            # new_population_individuals.append(self.best_solution_overall.copy()) 

            parents = self._selection()
            
            idx = 0
            while len(new_population_individuals) < self.population_size:
                p1 = parents[idx % len(parents)]
                p2 = parents[(idx + 1) % len(parents)] # Đảm bảo có p2
                idx += 2

                c1_chrom, c2_chrom = self._crossover(p1['chromosome'], p2['chromosome'])
                c1_mut_chrom = self._mutate(c1_chrom)
                c2_mut_chrom = self._mutate(c2_chrom)

                for child_chrom_set in [c1_mut_chrom, c2_mut_chrom]:
                    if len(new_population_individuals) >= self.population_size: break
                    child_path = self._reconstruct_path_from_chromosome(child_chrom_set)
                    if child_path and child_path[0] == self.start_node and child_path[-1] == self.target_node:
                        _, l, t, sp, dp, fitness = self._calculate_stats_for_path(child_path)
                        new_population_individuals.append({'chromosome': child_chrom_set, 'path': child_path, 
                                                           'fitness': fitness, 'length':l, 'turns':t, 
                                                           'safety_penalty':sp, 'diag_penalty':dp})
                    else: # Nếu con không hợp lệ, giữ lại cha mẹ (đơn giản)
                        new_population_individuals.append(p1 if len(new_population_individuals) % 2 == 0 else p2)


            self.population = new_population_individuals
            self.population.sort(key=lambda x: x['fitness'])
            
            current_gen_best = self.population[0]
            if current_gen_best['fitness'] < self.best_solution_overall['fitness']:
                self.best_solution_overall = current_gen_best.copy()

            self.convergence_curve.append(self.best_solution_overall['fitness'])
            if (gen + 1) % 10 == 0 or gen == 0 or gen == self.num_generations - 1:
                print(f"GA Gen {gen+1}/{self.num_generations}: BestFit={self.best_solution_overall['fitness']:.2f} "
                      f"(L:{self.best_solution_overall['length']:.1f}, T:{self.best_solution_overall['turns']}, "
                      f"SP:{self.best_solution_overall['safety_penalty']:.2f}, DP:{self.best_solution_overall['diag_penalty']:.2f})")
        
        res = self.best_solution_overall
        return (res['path'], res['length'], res['turns'], 
                res['safety_penalty'], res['diag_penalty'], res['fitness'])