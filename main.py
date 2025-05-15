
import numpy as np
from MAACO import MAACO
from env import (
    grid_fig7_layout_data,
    grid_map_fig13_base_data,
    grid_map_from_image_data,
    START_NODE_VAL, TARGET_NODE_VAL
)
from visualization import visualize_grid_and_multiple_paths # Only this one is called directly from main

if __name__ == '__main__':
    # --- Test Case 1: Fig 7 Environment with Fig 13 Params ---
    current_test_grid_base_fig7 = np.array(grid_fig7_layout_data)
    start_row_fig7, start_col_fig7 = 0, 0      
    target_row_fig7, target_col_fig7 = 19, 19  
    
    current_test_grid_base_fig7[start_row_fig7, start_col_fig7] = START_NODE_VAL
    current_test_grid_base_fig7[target_row_fig7, target_col_fig7] = TARGET_NODE_VAL

    maaco_params_fig13_paper = { # Renamed to avoid conflict
        "num_ants": 50, "num_iterations": 100, "alpha": 1.0, "beta": 7.0,
        "rho": 0.2, "Q": 2.5, 
        "a_turn_coef": 1.0, "wh_max": 0.9, "wh_min": 0.2, 
        "k_h_adaptive": 0.9, 
        "q0_initial": 0.5,
        "C0_initial_pheromone": 0.1
    }
    
    print("--- Running MAACO (Params from Fig 13 for Fig 7 Environment) ---")
    maaco_solver_fig7 = MAACO(grid=np.copy(current_test_grid_base_fig7), **maaco_params_fig13_paper)
    maaco_path_fig7, maaco_len_fig7, maaco_turns_fig7 = maaco_solver_fig7.solve_path_planning()

    if maaco_path_fig7:
        maaco_solver_fig7.plot_convergence_curve() # Calls MAACO's method
        visualize_grid_and_multiple_paths( # Calls visualization.py's function
            grid_data=current_test_grid_base_fig7,
            start_node_pos=maaco_solver_fig7.start_node,
            target_node_pos=maaco_solver_fig7.target_node,
            paths_dict={f"MAACO (L:{maaco_len_fig7:.2f}, T:{maaco_turns_fig7})": (maaco_path_fig7, 'orangered', '-')},
            title="MAACO Path on Fig 7 Environment (Params from Fig 13)"
        )
    
    # --- Test Case 2: Fig 13 Environment with Fig 13 Params ---
    # Note: The original grid_map_fig13_base had S and T embedded as 2 and 3
    # We'll use the raw data and set S/T explicitly for consistency
    grid_map_fig13_base = np.array(grid_map_fig13_base_data) 
    # Correcting start/target based on the provided data's 2 and 3 values
    start_row_fig13, start_col_fig13 = np.argwhere(grid_map_fig13_base == START_NODE_VAL)[0]
    target_row_fig13, target_col_fig13 = np.argwhere(grid_map_fig13_base == TARGET_NODE_VAL)[0]
    # Ensure they are set, even if already in data, to be explicit
    grid_map_fig13_base[start_row_fig13, start_col_fig13] = START_NODE_VAL
    grid_map_fig13_base[target_row_fig13, target_col_fig13] = TARGET_NODE_VAL


    maaco_params_fig13_test = { # Renamed to avoid conflict
        "num_ants": 50, "num_iterations": 100, "alpha": 1.0, "beta": 7.0,
        "rho": 0.2, "Q": 2.5,
        "a_turn_coef": 1.0, "wh_max": 0.9, "wh_min": 0.2, 
        "k_h_adaptive": 0.9, 
        "q0_initial": 0.5,
        "C0_initial_pheromone": 0.1
    }
    
    print("\n--- Running MAACO (Params and Environment from Fig 13) ---")
    maaco_solver_fig13 = MAACO(grid=np.copy(grid_map_fig13_base), **maaco_params_fig13_test)
    maaco_path_fig13, maaco_len_fig13, maaco_turns_fig13 = maaco_solver_fig13.solve_path_planning()

    if maaco_path_fig13:
        maaco_solver_fig13.plot_convergence_curve()
        visualize_grid_and_multiple_paths(
            grid_data=grid_map_fig13_base,
            start_node_pos=maaco_solver_fig13.start_node,
            target_node_pos=maaco_solver_fig13.target_node,
            paths_dict={f"MAACO (L:{maaco_len_fig13:.2f}, T:{maaco_turns_fig13})": (maaco_path_fig13, 'orangered', '-')},
            title="MAACO Path on Fig 13 Environment"
        )

    # --- Test Case 3: Image-based Map ---
    grid_map_img = np.array(grid_map_from_image_data)
    start_row_fig18, start_col_fig18 = 0, 0      
    target_row_fig18, target_col_fig18 = 19, 19  
    
    grid_map_img[start_row_fig18, start_col_fig18] = START_NODE_VAL
    grid_map_img[target_row_fig18, target_col_fig18] = TARGET_NODE_VAL

    maaco_params_fig18 = {
        "num_ants": 50, "num_iterations": 100, "alpha": 1.0, "beta": 7.0,
        "rho": 0.2, "Q": 2.5, 
        "a_turn_coef": 1.0, "wh_max": 0.9, "wh_min": 0.2, 
        "k_h_adaptive": 0.9,
        "q0_initial": 0.5,
        "C0_initial_pheromone": 0.1 
    }
    
    print("\n--- Running MAACO (Image-based Map) ---")
    maaco_solver_fig18 = MAACO(grid=np.copy(grid_map_img), **maaco_params_fig18)
    maaco_path_fig18, maaco_len_fig18, maaco_turns_fig18 = maaco_solver_fig18.solve_path_planning()

    if maaco_path_fig18:
        maaco_solver_fig18.plot_convergence_curve()
        visualize_grid_and_multiple_paths(
            grid_data=grid_map_img,
            start_node_pos=maaco_solver_fig18.start_node,
            target_node_pos=maaco_solver_fig18.target_node,
            paths_dict={f"MAACO (L:{maaco_len_fig18:.2f}, T:{maaco_turns_fig18})": (maaco_path_fig18, 'orangered', '-')},
            title="MAACO Path on Image-based Map"
        )