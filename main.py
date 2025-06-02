import numpy as np
from MAACO import MAACO 
from MPA import MPA   
from astar import AStarSolver
from dijkstra import DijkstraSolver
from ga_solver import GASolver
from pso import PSOSolver

from env import (
    grid_fig7_layout_data,
    grid_map_fig13_base_data,
    grid_map_from_image_data,
    grid_map_from_image_data2,
    grid_map_from_image_data3,
    grid_map_from_image_data5,
    START_NODE_VAL, TARGET_NODE_VAL, OBSTACLE, FREE_SPACE
)
from visualization import visualize_grid_and_multiple_paths

if __name__ == '__main__':
    COMMON_TURN_PENALTY = 0.3                
    COMMON_PROXIMITY_PENALTY_FACTOR = 0.8    
    COMMON_MIN_SAFE_DISTANCE_PROXIMITY = 1.8 
    COMMON_DIAGONAL_OBSTACLE_PENALTY = 100.0 
    
    # --- Test Case 1: Fig 7 Environment ---
    current_test_grid_base_fig7 = np.array(grid_fig7_layout_data)
    start_row_fig7, start_col_fig7 = 0, 0 
    target_row_fig7, target_col_fig7 = 19, 19
    current_test_grid_fig7_processed = np.copy(current_test_grid_base_fig7)
    current_test_grid_fig7_processed[start_row_fig7, start_col_fig7] = START_NODE_VAL
    current_test_grid_fig7_processed[target_row_fig7, target_col_fig7] = TARGET_NODE_VAL

    maaco_params_fig7 = {
        "num_ants": 50, "num_iterations": 100, "alpha": 1.0, "beta": 7.0,
        "rho": 0.1, "Q": 2.5, "a_turn_coef": 1.0, 
        "wh_max": 0.9, "wh_min": 0.2, "k_h_adaptive": 0.9, "q0_initial": 0.5, 
        "C0_initial_pheromone": 0.1,
        # "safety_penalty_factor": COMMON_PROXIMITY_PENALTY_FACTOR,
        # "min_safe_distance": COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        # "allow_diagonal_moves": True, 
        # "restrict_diagonal_near_obstacle_implicitly": True, 
    }
    mpa_params_fig7 = {
        "num_predators": 50, "num_iterations": 100, "FADs_rate": 0.2, "P_const": 0.5,
        "levy_beta": 2.0, "turn_penalty_factor": 0.1, 
        "safety_penalty_factor": COMMON_PROXIMITY_PENALTY_FACTOR,
        "min_safe_distance": COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        "diagonal_obstacle_penalty": COMMON_DIAGONAL_OBSTACLE_PENALTY, 
        "allow_diagonal_moves": True,        
        "restrict_diagonal_near_obstacle": True 
    }

    print("--- Running MAACO (Fig 7 Env - Safety & Diag Rules) ---")
    maaco_solver_fig7 = MAACO(grid=np.copy(current_test_grid_fig7_processed), **maaco_params_fig7)
    (maaco_path_fig7, maaco_len_fig7, maaco_turns_fig7) = maaco_solver_fig7.solve_path_planning()

    print("\n--- Running MPA (Fig 7 Env - Safety & Diag Rules) ---")
    mpa_solver_fig7 = MPA(grid=np.copy(current_test_grid_fig7_processed), **mpa_params_fig7)
    (mpa_path_fig7, mpa_len_fig7, mpa_turns_fig7, 
     mpa_sp_fig7, mpa_dp_fig7, mpa_fit_fig7) = mpa_solver_fig7.solve_path_planning()

    # A* Algorithm
    print("\n--- Running A* (Fig 7 Env - Safety & Diag Rules) ---")
    astar_solver_fig7 = AStarSolver(
        grid=np.copy(current_test_grid_fig7_processed),
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY
    )
    (astar_path_fig7, astar_len_fig7, astar_turns_fig7, 
     astar_sp_fig7, astar_dp_fig7, astar_fit_fig7) = astar_solver_fig7.solve()

    # Dijkstra Algorithm
    print("\n--- Running Dijkstra (Fig 7 Env - Safety & Diag Rules) ---")
    dijkstra_solver_fig7 = DijkstraSolver(
        grid=np.copy(current_test_grid_fig7_processed),
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY
    )
    (dijkstra_path_fig7, dijkstra_len_fig7, dijkstra_turns_fig7, 
     dijkstra_sp_fig7, dijkstra_dp_fig7, dijkstra_fit_fig7) = dijkstra_solver_fig7.solve()

    # GA Algorithm
    print("\n--- Running GA (Fig 7 Env - Safety & Diag Rules) ---")
    ga_solver_fig7 = GASolver(
        grid=np.copy(current_test_grid_fig7_processed),
        num_generations=100, population_size=50, num_waypoints_per_chromosome=5,
        mutation_rate=0.1, crossover_rate=0.8, tournament_size=3,
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY
    )
    (ga_path_fig7, ga_len_fig7, ga_turns_fig7, 
     ga_sp_fig7, ga_dp_fig7, ga_fit_fig7) = ga_solver_fig7.solve()

    # PSO Algorithm
    print("\n--- Running PSO (Fig 7 Env - Safety & Diag Rules) ---")
    pso_solver_fig7 = PSOSolver(
        grid=np.copy(current_test_grid_fig7_processed),
        num_iterations=50, num_particles=100, num_waypoints_per_particle=5,
        w=0.7, c1=1.5, c2=1.5,
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY
    )
    (pso_path_fig7, pso_len_fig7, pso_turns_fig7, 
     pso_sp_fig7, pso_dp_fig7, pso_fit_fig7) = pso_solver_fig7.solve()    # First visualization: MPA, A*, Dijkstra
    paths_to_visualize_fig7_group1 = {}
    if mpa_path_fig7:
        mpa_solver_fig7.plot_convergence_curve()
        paths_to_visualize_fig7_group1[
            f"MPA (F:{mpa_fit_fig7:.2f}|L:{mpa_len_fig7:.1f},T:{mpa_turns_fig7},SP:{mpa_sp_fig7:.2f},DP:{mpa_dp_fig7:.2f})"
        ] = (mpa_path_fig7, 'blue', '--')
    
    if astar_path_fig7:
        paths_to_visualize_fig7_group1[
            f"A* (F:{astar_fit_fig7:.2f}|L:{astar_len_fig7:.1f},T:{astar_turns_fig7},SP:{astar_sp_fig7:.2f},DP:{astar_dp_fig7:.2f})"
        ] = (astar_path_fig7, 'green', ':')
    
    if dijkstra_path_fig7:
        paths_to_visualize_fig7_group1[
            f"Dijkstra (F:{dijkstra_fit_fig7:.2f}|L:{dijkstra_len_fig7:.1f},T:{dijkstra_turns_fig7},SP:{dijkstra_sp_fig7:.2f},DP:{dijkstra_dp_fig7:.2f})"
        ] = (dijkstra_path_fig7, 'purple', '-.')
    
    if paths_to_visualize_fig7_group1:
        visualize_grid_and_multiple_paths(
            grid_data=current_test_grid_fig7_processed,
            start_node_pos=(start_row_fig7, start_col_fig7), 
            target_node_pos=(target_row_fig7, target_col_fig7),
            paths_dict=paths_to_visualize_fig7_group1,
            title="Classical Algorithms Comparison - Fig 7 Environment (MPA, A*, Dijkstra)"
        )

    # Second visualization: MPA, MAACO, GA, PSO
    paths_to_visualize_fig7_group2 = {}
    if mpa_path_fig7:
        paths_to_visualize_fig7_group2[
            f"MPA (F:{mpa_fit_fig7:.2f}|L:{mpa_len_fig7:.1f},T:{mpa_turns_fig7},SP:{mpa_sp_fig7:.2f},DP:{mpa_dp_fig7:.2f})"
        ] = (mpa_path_fig7, 'blue', '--')
    if maaco_path_fig7:
        maaco_solver_fig7.plot_convergence_curve()
        paths_to_visualize_fig7_group2[
            f"MAACO (L:{maaco_len_fig7:.1f},T:{maaco_turns_fig7})"
        ] = (maaco_path_fig7, 'orangered', '-')
    if ga_path_fig7:
        ga_solver_fig7.plot_convergence_curve("GA")
        paths_to_visualize_fig7_group2[
            f"GA (F:{ga_fit_fig7:.2f}|L:{ga_len_fig7:.1f},T:{ga_turns_fig7},SP:{ga_sp_fig7:.2f},DP:{ga_dp_fig7:.2f})"
        ] = (ga_path_fig7, 'red', (0, (3, 1, 1, 1)))
    if pso_path_fig7:
        pso_solver_fig7.plot_convergence_curve("PSO")
        paths_to_visualize_fig7_group2[
            f"PSO (F:{pso_fit_fig7:.2f}|L:{pso_len_fig7:.1f},T:{pso_turns_fig7},SP:{pso_sp_fig7:.2f},DP:{pso_dp_fig7:.2f})"
        ] = (pso_path_fig7, 'brown', (0, (5, 2, 1, 2)))
    
    if paths_to_visualize_fig7_group2:
        visualize_grid_and_multiple_paths(
            grid_data=current_test_grid_fig7_processed,
            start_node_pos=(start_row_fig7, start_col_fig7), 
            target_node_pos=(target_row_fig7, target_col_fig7),
            paths_dict=paths_to_visualize_fig7_group2,
            title="Metaheuristic Algorithms Comparison - Fig 7 Environment (MPA, MAACO, GA, PSO)"
        )

    # --- Test Case 2: Fig 13 Environment ---    # ... (Cập nhật tương tự cho Fig 13 và Image-based Map) ...
    grid_map_fig13_processed = np.array(grid_map_fig13_base_data)
    try:
        start_nodes_f13 = np.argwhere(grid_map_fig13_processed == START_NODE_VAL)
        target_nodes_f13 = np.argwhere(grid_map_fig13_processed == TARGET_NODE_VAL)
        if start_nodes_f13.size > 0 and target_nodes_f13.size > 0:
            start_row_fig13, start_col_fig13 = start_nodes_f13[0]; target_row_fig13, target_col_fig13 = target_nodes_f13[0]
        else: raise IndexError
    except IndexError:
        print("Fig 13: Start/Target not in grid, using defaults (19,0)->(0,19).")
        start_row_fig13,start_col_fig13=19,0; target_row_fig13,target_col_fig13=0,19
        grid_map_fig13_processed[start_row_fig13,start_col_fig13]=START_NODE_VAL
        grid_map_fig13_processed[target_row_fig13,target_col_fig13]=TARGET_NODE_VAL
    
    maaco_params_fig13 = {**maaco_params_fig7} 
    mpa_params_fig13 = {**mpa_params_fig7}
    
    print("\n--- Running MAACO (Fig 13 Env - Safety & Diag Rules) ---")
    maaco_solver_fig13 = MAACO(grid=np.copy(grid_map_fig13_processed), **maaco_params_fig13)
    (maaco_path_f13, maaco_len_f13, maaco_turns_f13) = maaco_solver_fig13.solve_path_planning()
    
    print("\n--- Running MPA (Fig 13 Env - Safety & Diag Rules) ---")
    mpa_solver_fig13 = MPA(grid=np.copy(grid_map_fig13_processed), **mpa_params_fig13)
    (mpa_path_f13, mpa_len_f13, mpa_turns_f13, mpa_sp_f13, mpa_dp_f13, mpa_fit_f13) = mpa_solver_fig13.solve_path_planning()

    print("\n--- Running A* (Fig 13 Env - Safety & Diag Rules) ---")
    astar_solver_fig13 = AStarSolver(
        grid=np.copy(grid_map_fig13_processed),
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY
    )
    (astar_path_f13, astar_len_f13, astar_turns_f13, astar_sp_f13, astar_dp_f13, astar_fit_f13) = astar_solver_fig13.solve()

    print("\n--- Running Dijkstra (Fig 13 Env - Safety & Diag Rules) ---")
    dijkstra_solver_fig13 = DijkstraSolver(
        grid=np.copy(grid_map_fig13_processed),
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY
    )
    (dijkstra_path_f13, dijkstra_len_f13, dijkstra_turns_f13, dijkstra_sp_f13, dijkstra_dp_f13, dijkstra_fit_f13) = dijkstra_solver_fig13.solve()

    print("\n--- Running GA (Fig 13 Env - Safety & Diag Rules) ---")
    ga_solver_fig13 = GASolver(
        grid=np.copy(grid_map_fig13_processed),
        num_generations=50, population_size=30, num_waypoints_per_chromosome=5,
        mutation_rate=0.1, crossover_rate=0.8, tournament_size=3,
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY
    )
    (ga_path_f13, ga_len_f13, ga_turns_f13, ga_sp_f13, ga_dp_f13, ga_fit_f13) = ga_solver_fig13.solve()

    print("\n--- Running PSO (Fig 13 Env - Safety & Diag Rules) ---")
    pso_solver_fig13 = PSOSolver(
        grid=np.copy(grid_map_fig13_processed),
        num_iterations=50, num_particles=30, num_waypoints_per_particle=5,
        w=0.7, c1=1.5, c2=1.5,
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY    )
    (pso_path_f13, pso_len_f13, pso_turns_f13, pso_sp_f13, pso_dp_f13, pso_fit_f13) = pso_solver_fig13.solve()      # First visualization: MPA, A*, Dijkstra
    paths_to_visualize_fig13_group1 = {}
    if mpa_path_f13:
        mpa_solver_fig13.plot_convergence_curve()
        paths_to_visualize_fig13_group1[f"MPA (F:{mpa_fit_f13:.2f}|L:{mpa_len_f13:.1f},T:{mpa_turns_f13},SP:{mpa_sp_f13:.2f},DP:{mpa_dp_f13:.2f})"] = (mpa_path_f13, 'blue', '--')
    
    if astar_path_f13:
        paths_to_visualize_fig13_group1[f"A* (F:{astar_fit_f13:.2f}|L:{astar_len_f13:.1f},T:{astar_turns_f13},SP:{astar_sp_f13:.2f},DP:{astar_dp_f13:.2f})"] = (astar_path_f13, 'green', ':')
    if dijkstra_path_f13:
        paths_to_visualize_fig13_group1[f"Dijkstra (F:{dijkstra_fit_f13:.2f}|L:{dijkstra_len_f13:.1f},T:{dijkstra_turns_f13},SP:{dijkstra_sp_f13:.2f},DP:{dijkstra_dp_f13:.2f})"] = (dijkstra_path_f13, 'purple', '-.')
    
    if paths_to_visualize_fig13_group1:
        visualize_grid_and_multiple_paths(
            grid_map_fig13_processed, 
            (start_row_fig13,start_col_fig13), 
            (target_row_fig13,target_col_fig13), 
            paths_to_visualize_fig13_group1, 
            "Classical Algorithms Comparison - Fig 13 Environment (MPA, A*, Dijkstra)"
        )

    # Second visualization: MPA, MAACO, GA, PSO
    paths_to_visualize_fig13_group2 = {}
    if mpa_path_f13:
        paths_to_visualize_fig13_group2[f"MPA (F:{mpa_fit_f13:.2f}|L:{mpa_len_f13:.1f},T:{mpa_turns_f13},SP:{mpa_sp_f13:.2f},DP:{mpa_dp_f13:.2f})"] = (mpa_path_f13, 'blue', '--')
    if maaco_path_f13:
        maaco_solver_fig13.plot_convergence_curve()
        paths_to_visualize_fig13_group2[f"MAACO (L:{maaco_len_f13:.1f},T:{maaco_turns_f13})"] = (maaco_path_f13, 'orangered', '-')
    if ga_path_f13:
        ga_solver_fig13.plot_convergence_curve("GA")
        paths_to_visualize_fig13_group2[f"GA (F:{ga_fit_f13:.2f}|L:{ga_len_f13:.1f},T:{ga_turns_f13},SP:{ga_sp_f13:.2f},DP:{ga_dp_f13:.2f})"] = (ga_path_f13, 'red', (0, (3, 1, 1, 1)))
    if pso_path_f13:
        pso_solver_fig13.plot_convergence_curve("PSO")
        paths_to_visualize_fig13_group2[f"PSO (F:{pso_fit_f13:.2f}|L:{pso_len_f13:.1f},T:{pso_turns_f13},SP:{pso_sp_f13:.2f},DP:{pso_dp_f13:.2f})"] = (pso_path_f13, 'brown', (0, (5, 2, 1, 2)))
    
    if paths_to_visualize_fig13_group2:
        visualize_grid_and_multiple_paths(
            grid_map_fig13_processed, 
            (start_row_fig13,start_col_fig13), 
            (target_row_fig13,target_col_fig13), 
            paths_to_visualize_fig13_group2, 
            "Metaheuristic Algorithms Comparison - Fig 13 Environment (MPA, MAACO, GA, PSO)"
        )# --- Test Case 3: Image-based Map ---
    grid_map_img_processed_main = np.array(grid_map_from_image_data)
    start_row_img_main, start_col_img_main = 0,0
    target_row_img_main, target_col_img_main = 19,19
    if grid_map_img_processed_main[start_row_img_main, start_col_img_main] == OBSTACLE:
        print(f"ImgMap: Start ({start_row_img_main},{start_col_img_main}) is obstacle. Finding free start."); free_s_img_m = np.argwhere(grid_map_img_processed_main != OBSTACLE)
        if free_s_img_m.size > 0: start_row_img_main, start_col_img_main = free_s_img_m[0]
        else: raise ValueError("No free start in image map.")
    if grid_map_img_processed_main[target_row_img_main, target_col_img_main] == OBSTACLE:
        print(f"ImgMap: Target ({target_row_img_main},{target_col_img_main}) is obstacle. Finding free target."); free_t_img_m = np.argwhere(grid_map_img_processed_main != OBSTACLE)
        if free_t_img_m.size > 0: target_row_img_main, target_col_img_main = free_t_img_m[-1]
        else: raise ValueError("No free target in image map.")
    grid_map_img_processed_main[start_row_img_main, start_col_img_main] = START_NODE_VAL
    grid_map_img_processed_main[target_row_img_main, target_col_img_main] = TARGET_NODE_VAL
    
    maaco_params_img_main = {**maaco_params_fig7}
    mpa_params_img_main = {**mpa_params_fig7}

    print("\n--- Running MAACO (Image-based Map with Safety & Diag Rules) ---")
    maaco_solver_img_main = MAACO(grid=np.copy(grid_map_img_processed_main), **maaco_params_img_main)
    (maaco_path_img_m, maaco_len_img_m, maaco_turns_img_m) = maaco_solver_img_main.solve_path_planning()
    
    print("\n--- Running MPA (Image-based Map with Safety & Diag Rules) ---")
    mpa_solver_img_main = MPA(grid=np.copy(grid_map_img_processed_main), **mpa_params_img_main)
    (mpa_path_img_m, mpa_len_img_m, mpa_turns_img_m, mpa_sp_img_m, mpa_dp_img_m, mpa_fit_img_m) = mpa_solver_img_main.solve_path_planning()

    print("\n--- Running A* (Image-based Map with Safety & Diag Rules) ---")
    astar_solver_img_main = AStarSolver(
        grid=np.copy(grid_map_img_processed_main),
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY
    )
    (astar_path_img_m, astar_len_img_m, astar_turns_img_m, astar_sp_img_m, astar_dp_img_m, astar_fit_img_m) = astar_solver_img_main.solve()

    print("\n--- Running Dijkstra (Image-based Map with Safety & Diag Rules) ---")
    dijkstra_solver_img_main = DijkstraSolver(
        grid=np.copy(grid_map_img_processed_main),
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY
    )
    (dijkstra_path_img_m, dijkstra_len_img_m, dijkstra_turns_img_m, dijkstra_sp_img_m, dijkstra_dp_img_m, dijkstra_fit_img_m) = dijkstra_solver_img_main.solve()

    print("\n--- Running GA (Image-based Map with Safety & Diag Rules) ---")
    ga_solver_img_main = GASolver(
        grid=np.copy(grid_map_img_processed_main),
        num_generations=30, population_size=20, num_waypoints_per_chromosome=3,
        mutation_rate=0.15, crossover_rate=0.8, tournament_size=3,
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY
    )
    (ga_path_img_m, ga_len_img_m, ga_turns_img_m, ga_sp_img_m, ga_dp_img_m, ga_fit_img_m) = ga_solver_img_main.solve()

    print("\n--- Running PSO (Image-based Map with Safety & Diag Rules) ---")
    pso_solver_img_main = PSOSolver(
        grid=np.copy(grid_map_img_processed_main),
        num_iterations=30, num_particles=20, num_waypoints_per_particle=3,
        w=0.7, c1=1.5, c2=1.5,
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY
    )
    (pso_path_img_m, pso_len_img_m, pso_turns_img_m, pso_sp_img_m, pso_dp_img_m, pso_fit_img_m) = pso_solver_img_main.solve()    
    # First visualization: MPA, A*, Dijkstra
    paths_to_visualize_img_main_group1 = {}
    if mpa_path_img_m:
        mpa_solver_img_main.plot_convergence_curve()
        paths_to_visualize_img_main_group1[f"MPA (F:{mpa_fit_img_m:.2f}|L:{mpa_len_img_m:.1f},T:{mpa_turns_img_m},SP:{mpa_sp_img_m:.2f},DP:{mpa_dp_img_m:.2f})"] = (mpa_path_img_m, 'blue', '--')
    if astar_path_img_m:
        astar_solver_img_main.plot_convergence_curve("A*")
        paths_to_visualize_img_main_group1[f"A* (F:{astar_fit_img_m:.2f}|L:{astar_len_img_m:.1f},T:{astar_turns_img_m},SP:{astar_sp_img_m:.2f},DP:{astar_dp_img_m:.2f})"] = (astar_path_img_m, 'green', ':')
    if dijkstra_path_img_m:
        dijkstra_solver_img_main.plot_convergence_curve("Dijkstra")
        paths_to_visualize_img_main_group1[f"Dijkstra (F:{dijkstra_fit_img_m:.2f}|L:{dijkstra_len_img_m:.1f},T:{dijkstra_turns_img_m},SP:{dijkstra_sp_img_m:.2f},DP:{dijkstra_dp_img_m:.2f})"] = (dijkstra_path_img_m, 'purple', '-.')
    
    if paths_to_visualize_img_main_group1:
        visualize_grid_and_multiple_paths(
            grid_map_img_processed_main, 
            (start_row_img_main,start_col_img_main), 
            (target_row_img_main,target_col_img_main), 
            paths_to_visualize_img_main_group1, 
            "Classical Algorithms Comparison - Image-based Map 1 (MPA, A*, Dijkstra)"
        )

    # Second visualization: MPA, MAACO, GA, PSO
    paths_to_visualize_img_main_group2 = {}
    if mpa_path_img_m:
        paths_to_visualize_img_main_group2[f"MPA (F:{mpa_fit_img_m:.2f}|L:{mpa_len_img_m:.1f},T:{mpa_turns_img_m},SP:{mpa_sp_img_m:.2f},DP:{mpa_dp_img_m:.2f})"] = (mpa_path_img_m, 'blue', '--')
    if maaco_path_img_m:
        maaco_solver_img_main.plot_convergence_curve()
        paths_to_visualize_img_main_group2[f"MAACO (L:{maaco_len_img_m:.1f},T:{maaco_turns_img_m})"] = (maaco_path_img_m, 'orangered', '-')
    if ga_path_img_m:
        ga_solver_img_main.plot_convergence_curve("GA")
        paths_to_visualize_img_main_group2[f"GA (F:{ga_fit_img_m:.2f}|L:{ga_len_img_m:.1f},T:{ga_turns_img_m},SP:{ga_sp_img_m:.2f},DP:{ga_dp_img_m:.2f})"] = (ga_path_img_m, 'red', (0, (3, 1, 1, 1)))
    if pso_path_img_m:
        pso_solver_img_main.plot_convergence_curve("PSO")
        paths_to_visualize_img_main_group2[f"PSO (F:{pso_fit_img_m:.2f}|L:{pso_len_img_m:.1f},T:{pso_turns_img_m},SP:{pso_sp_img_m:.2f},DP:{pso_dp_img_m:.2f})"] = (pso_path_img_m, 'brown', (0, (5, 2, 1, 2)))
    
    if paths_to_visualize_img_main_group2:
        visualize_grid_and_multiple_paths(
            grid_map_img_processed_main, 
            (start_row_img_main,start_col_img_main), 
            (target_row_img_main,target_col_img_main), 
            paths_to_visualize_img_main_group2, 
            "Metaheuristic Algorithms Comparison - Image-based Map 1 (MPA, MAACO, GA, PSO)"
        )
    grid_map_img_processed_main = np.array(grid_map_from_image_data2)
    start_row_img_main, start_col_img_main = 0,0
    target_row_img_main, target_col_img_main = 19,19
    if grid_map_img_processed_main[start_row_img_main, start_col_img_main] == OBSTACLE:
        print(f"ImgMap: Start ({start_row_img_main},{start_col_img_main}) is obstacle. Finding free start."); free_s_img_m = np.argwhere(grid_map_img_processed_main != OBSTACLE)
        if free_s_img_m.size > 0: start_row_img_main, start_col_img_main = free_s_img_m[0]
        else: raise ValueError("No free start in image map.")
    if grid_map_img_processed_main[target_row_img_main, target_col_img_main] == OBSTACLE:
        print(f"ImgMap: Target ({target_row_img_main},{target_col_img_main}) is obstacle. Finding free target."); free_t_img_m = np.argwhere(grid_map_img_processed_main != OBSTACLE)
        if free_t_img_m.size > 0: target_row_img_main, target_col_img_main = free_t_img_m[-1]
        else: raise ValueError("No free target in image map.")
    grid_map_img_processed_main[start_row_img_main, start_col_img_main] = START_NODE_VAL
    grid_map_img_processed_main[target_row_img_main, target_col_img_main] = TARGET_NODE_VAL
    
    maaco_params_img_main = {**maaco_params_fig7}
    mpa_params_img_main = {**mpa_params_fig7}

    print("\n--- Running MAACO (Image-based Map 2 with Safety & Diag Rules) ---")
    maaco_solver_img_main = MAACO(grid=np.copy(grid_map_img_processed_main), **maaco_params_img_main)
    (maaco_path_img_m, maaco_len_img_m, maaco_turns_img_m) = maaco_solver_img_main.solve_path_planning()
    
    print("\n--- Running MPA (Image-based Map 2 with Safety & Diag Rules) ---")
    mpa_solver_img_main = MPA(grid=np.copy(grid_map_img_processed_main), **mpa_params_img_main)
    (mpa_path_img_m, mpa_len_img_m, mpa_turns_img_m, mpa_sp_img_m, mpa_dp_img_m, mpa_fit_img_m) = mpa_solver_img_main.solve_path_planning()

    print("\n--- Running A* (Image-based Map 2 with Safety & Diag Rules) ---")
    astar_solver_img_main = AStarSolver(
        grid=np.copy(grid_map_img_processed_main),
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY
    )
    (astar_path_img_m, astar_len_img_m, astar_turns_img_m, astar_sp_img_m, astar_dp_img_m, astar_fit_img_m) = astar_solver_img_main.solve()

    print("\n--- Running Dijkstra (Image-based Map 2 with Safety & Diag Rules) ---")
    dijkstra_solver_img_main = DijkstraSolver(
        grid=np.copy(grid_map_img_processed_main),
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY
    )
    (dijkstra_path_img_m, dijkstra_len_img_m, dijkstra_turns_img_m, dijkstra_sp_img_m, dijkstra_dp_img_m, dijkstra_fit_img_m) = dijkstra_solver_img_main.solve()

    print("\n--- Running GA (Image-based Map 2 with Safety & Diag Rules) ---")
    ga_solver_img_main = GASolver(
        grid=np.copy(grid_map_img_processed_main),
        num_generations=30, population_size=20, num_waypoints_per_chromosome=3,
        mutation_rate=0.15, crossover_rate=0.8, tournament_size=3,
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY
    )
    (ga_path_img_m, ga_len_img_m, ga_turns_img_m, ga_sp_img_m, ga_dp_img_m, ga_fit_img_m) = ga_solver_img_main.solve()

    print("\n--- Running PSO (Image-based Map 2 with Safety & Diag Rules) ---")
    pso_solver_img_main = PSOSolver(
        grid=np.copy(grid_map_img_processed_main),
        num_iterations=30, num_particles=20, num_waypoints_per_particle=3,
        w=0.7, c1=1.5, c2=1.5,
        turn_penalty_factor=COMMON_TURN_PENALTY,
        safety_penalty_factor=COMMON_PROXIMITY_PENALTY_FACTOR,
        min_safe_distance=COMMON_MIN_SAFE_DISTANCE_PROXIMITY,
        allow_diagonal_moves=True,
        restrict_diagonal_near_obstacle_policy=True,
        diagonal_obstacle_penalty_value=COMMON_DIAGONAL_OBSTACLE_PENALTY
    )
    (pso_path_img_m, pso_len_img_m, pso_turns_img_m, pso_sp_img_m, pso_dp_img_m, pso_fit_img_m) = pso_solver_img_main.solve()    
    # First visualization: MPA, A*, Dijkstra
    paths_to_visualize_img_main_group1 = {}
    if mpa_path_img_m:
        mpa_solver_img_main.plot_convergence_curve()
        paths_to_visualize_img_main_group1[f"MPA (F:{mpa_fit_img_m:.2f}|L:{mpa_len_img_m:.1f},T:{mpa_turns_img_m},SP:{mpa_sp_img_m:.2f},DP:{mpa_dp_img_m:.2f})"] = (mpa_path_img_m, 'blue', '--')
    if astar_path_img_m:
        astar_solver_img_main.plot_convergence_curve("A*")
        paths_to_visualize_img_main_group1[f"A* (F:{astar_fit_img_m:.2f}|L:{astar_len_img_m:.1f},T:{astar_turns_img_m},SP:{astar_sp_img_m:.2f},DP:{astar_dp_img_m:.2f})"] = (astar_path_img_m, 'green', ':')
    if dijkstra_path_img_m:
        dijkstra_solver_img_main.plot_convergence_curve("Dijkstra")
        paths_to_visualize_img_main_group1[f"Dijkstra (F:{dijkstra_fit_img_m:.2f}|L:{dijkstra_len_img_m:.1f},T:{dijkstra_turns_img_m},SP:{dijkstra_sp_img_m:.2f},DP:{dijkstra_dp_img_m:.2f})"] = (dijkstra_path_img_m, 'purple', '-.')
    
    if paths_to_visualize_img_main_group1:
        visualize_grid_and_multiple_paths(
            grid_map_img_processed_main, 
            (start_row_img_main,start_col_img_main), 
            (target_row_img_main,target_col_img_main), 
            paths_to_visualize_img_main_group1, 
            "Classical Algorithms Comparison - Image-based Map 2 (MPA, A*, Dijkstra)"
        )

    # Second visualization: MPA, MAACO, GA, PSO
    paths_to_visualize_img_main_group2 = {}
    if mpa_path_img_m:
        paths_to_visualize_img_main_group2[f"MPA (F:{mpa_fit_img_m:.2f}|L:{mpa_len_img_m:.1f},T:{mpa_turns_img_m},SP:{mpa_sp_img_m:.2f},DP:{mpa_dp_img_m:.2f})"] = (mpa_path_img_m, 'blue', '--')
    if maaco_path_img_m:
        maaco_solver_img_main.plot_convergence_curve()
        paths_to_visualize_img_main_group2[f"MAACO (L:{maaco_len_img_m:.1f},T:{maaco_turns_img_m})"] = (maaco_path_img_m, 'orangered', '-')
    if ga_path_img_m:
        ga_solver_img_main.plot_convergence_curve("GA")
        paths_to_visualize_img_main_group2[f"GA (F:{ga_fit_img_m:.2f}|L:{ga_len_img_m:.1f},T:{ga_turns_img_m},SP:{ga_sp_img_m:.2f},DP:{ga_dp_img_m:.2f})"] = (ga_path_img_m, 'red', (0, (3, 1, 1, 1)))
    if pso_path_img_m:
        pso_solver_img_main.plot_convergence_curve("PSO")
        paths_to_visualize_img_main_group2[f"PSO (F:{pso_fit_img_m:.2f}|L:{pso_len_img_m:.1f},T:{pso_turns_img_m},SP:{pso_sp_img_m:.2f},DP:{pso_dp_img_m:.2f})"] = (pso_path_img_m, 'brown', (0, (5, 2, 1, 2)))
    
    if paths_to_visualize_img_main_group2:
        visualize_grid_and_multiple_paths(
            grid_map_img_processed_main, 
            (start_row_img_main,start_col_img_main), 
            (target_row_img_main,target_col_img_main), 
            paths_to_visualize_img_main_group2, 
            "Metaheuristic Algorithms Comparison - Image-based Map 2 (MPA, MAACO, GA, PSO)"
        )

    
    