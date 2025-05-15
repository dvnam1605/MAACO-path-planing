import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from env import FREE_SPACE, OBSTACLE, START_NODE_VAL, TARGET_NODE_VAL # Import constants

def visualize_grid_and_multiple_paths(grid_data, start_node_pos, target_node_pos, paths_dict, title="So sánh Đường đi"):
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
    
    plt.figure(figsize=(max(8, cols_viz/1.8), max(8, rows_viz/1.8)))
    plt.imshow(display_grid_viz, cmap=cmap_viz, norm=norm_viz, origin='upper', 
              interpolation='nearest')
    plt.title(title, fontsize=16, fontweight='bold')
    
    plt.xticks(np.arange(cols_viz), fontsize=10)
    plt.yticks(np.arange(rows_viz), fontsize=10)
    plt.gca().set_xticks(np.arange(-.5, cols_viz, 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, rows_viz, 1), minor=True)
    plt.grid(True, which='minor', color='lightgray', linewidth=0.5)
    plt.grid(True, which='major', color='darkgray', linewidth=0.5)
    
    legend_handles = []
    for algo_name, (path_coords, path_color, path_linestyle) in paths_dict.items():
        if path_coords and len(path_coords) > 1:
            path_rows_viz = [p[0] for p in path_coords]
            path_cols_viz = [p[1] for p in path_coords]
            line, = plt.plot(path_cols_viz, path_rows_viz, marker='o', color=path_color,
                             linestyle=path_linestyle, markersize=5, linewidth=2.0, alpha=0.85, label=algo_name)
            legend_handles.append(line)
    if legend_handles:
        plt.legend(handles=legend_handles, fontsize=11, loc='best')
    plt.tight_layout()
    plt.show()

def visualize_pheromone_matrix(grid_data, pheromone_data, title="Mức Pheromone"):
    rows_viz, cols_viz = grid_data.shape
    plt.figure(figsize=(max(6,cols_viz/2.0), max(6,rows_viz/2.0)))
    masked_pheromones_viz = np.ma.masked_where(grid_data == OBSTACLE, pheromone_data) # Use OBSTACLE from env
    
    if np.nanmax(masked_pheromones_viz) > 0:
        log_pheromones = np.log1p(masked_pheromones_viz - np.nanmin(masked_pheromones_viz))
        im = plt.imshow(log_pheromones, cmap='viridis', origin='upper', interpolation='nearest')
    else:
        im = plt.imshow(masked_pheromones_viz, cmap='viridis', origin='upper', interpolation='nearest')

    plt.colorbar(im, label="Cường độ Pheromone (Log Scale)")
    plt.title(title, fontsize=14)
    plt.xticks(fontsize=9); plt.yticks(fontsize=9)
    plt.show()
    
def plot_convergence_curve(convergence_data, algo_name="Thuật toán", color='dodgerblue'):
    plt.figure(figsize=(9,5.5))
    valid_iterations_viz = [i+1 for i, l_val in enumerate(convergence_data) if l_val is not None and l_val != float('inf')]
    valid_lengths_viz = [l_val for l_val in convergence_data if l_val is not None and l_val != float('inf')]
    if valid_lengths_viz:
        plt.plot(valid_iterations_viz, valid_lengths_viz, marker='o', linestyle='-', markersize=5, color=color, linewidth=1.8)
        plt.xlabel("Số lần lặp", fontsize=12)
        plt.ylabel("Độ dài đường đi tốt nhất", fontsize=12)
        plt.title(f"Đường cong hội tụ của {algo_name}", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.xticks(fontsize=10); plt.yticks(fontsize=10)
        if len(valid_lengths_viz) > 1:
             min_y = min(valid_lengths_viz) * 0.98
             max_y = max(valid_lengths_viz) * 1.02
             if max_y > min_y:
                 plt.ylim(min_y, max_y)
        plt.tight_layout()
        plt.show()
    else:
        print(f"{algo_name}: Không có dữ liệu độ dài đường đi hợp lệ để vẽ đồ thị hội tụ.")