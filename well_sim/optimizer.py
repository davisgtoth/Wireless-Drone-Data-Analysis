import os
import time
import signal
import hashlib
import numpy as np
from scipy.optimize import differential_evolution
from physics import calculate_simulated_well, get_B_components
from target import generate_target_well

def build_or_load_physics_matrix(x_array, radii, hardware_params):
    """
    Checks the disk for a pre-computed magnetic field matrix based on the hardware limits.
    If it doesn't exist, it calculates it once and saves it for future runs.
    """
    # Create a unique hash based on the physical hardware limits
    hash_str = f"{hardware_params['z_height_cm']}_{hardware_params['r_min_cm']}_{hardware_params['r_max_cm']}_{hardware_params['wire_diameter_cm']}_{hardware_params['max_turns']}_{len(x_array)}_{x_array[0]}_{x_array[-1]}"
    hash_id = hashlib.md5(hash_str.encode()).hexdigest()[:8]
    
    cache_dir = "physics_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"matrix_{hash_id}.npz")
    
    max_turns = hardware_params['max_turns']
    num_slots = len(radii)
    
    # Check if we already did the math
    if os.path.exists(cache_file):
        print(f"Loading pre-computed physics matrix from cache ({hash_id})...")
        data = np.load(cache_file)
        return data['Bx'], data['Bz']
        
    print("Pre-computing Biot-Savart physics matrix. This will take a few seconds...")
    
    # 3D Arrays: [Slot_Index, Turn_Value_Index, X_Spatial_Index]
    Bx_matrix = np.zeros((num_slots, 2 * max_turns + 1, len(x_array)))
    Bz_matrix = np.zeros((num_slots, 2 * max_turns + 1, len(x_array)))
    
    for slot_idx, R in enumerate(radii):
        # We map turns from [-max_turns to +max_turns] into array indices [0 to 2*max_turns]
        for turn_idx, N in enumerate(range(-max_turns, max_turns + 1)):
            if N == 0:
                continue
                
            direction = np.sign(N)
            num_turns_abs = int(abs(N))
            
            # Stack the wires downward accurately
            for i in range(num_turns_abs):
                z_i = hardware_params['z_height_cm'] + (i * hardware_params['wire_diameter_cm'])
                Bx, Bz = get_B_components(x_array, z_i, R)
                Bx_matrix[slot_idx, turn_idx, :] += direction * Bx
                Bz_matrix[slot_idx, turn_idx, :] += direction * Bz
                
    # Save to disk
    np.savez_compressed(cache_file, Bx=Bx_matrix, Bz=Bz_matrix)
    print("Physics matrix cached successfully!")
    
    return Bx_matrix, Bz_matrix

def cost_function(turns_continuous, Bx_matrix_eval, Bz_matrix_eval, max_turns, angle_deg, V_target_eval):
    """
    Evaluates the coil winding profile using lightning-fast NumPy matrix lookups
    instead of recalculating the physical Biot-Savart integrals.
    """
    turns_array = np.round(turns_continuous).astype(int)
    num_slots = len(turns_array)
    
    # Convert physical turn counts (e.g., -2) to matrix indices (e.g., max_turns - 2)
    indices = turns_array + max_turns
    
    # --- The Matrix Magic ---
    # We use advanced indexing to pull the exact pre-calculated magnetic curve 
    # for each slot, then instantly sum them all up vertically.
    Btot_x = np.sum(Bx_matrix_eval[np.arange(num_slots), indices, :], axis=0)
    Btot_z = np.sum(Bz_matrix_eval[np.arange(num_slots), indices, :], axis=0)
    
    # Recreate the flux magnitude profile
    theta = np.radians(angle_deg)
    Flux_right = Btot_x * np.sin(theta) + Btot_z * np.cos(theta)
    Flux_left = Btot_x * np.sin(-theta) + Btot_z * np.cos(-theta)
    
    V_sim_eval = np.maximum(np.abs(Flux_left), np.abs(Flux_right))
    
    peak_val = np.max(V_sim_eval)
    if peak_val > 0:
        V_sim_eval = V_sim_eval / peak_val
    else:
        return 1e6 
        
    mse_shape = np.mean((V_sim_eval - V_target_eval)**2)
    turn_jumps = np.abs(np.diff(turns_array))
    penalty = 0.001 * np.sum(turn_jumps)
        
    return mse_shape + penalty

def run_optimization(hardware_params, target_params):
    """
    Sets up the search space and executes the Differential Evolution algorithm.
    """
    z_height = hardware_params['z_height_cm']
    angle_deg = hardware_params['angle_deg']
    R_min = hardware_params['r_min_cm']
    R_max = hardware_params['r_max_cm']
    wire_dia = hardware_params['wire_diameter_cm']
    max_turns = hardware_params['max_turns']
    eval_limit = hardware_params['eval_limit_cm']
    bucking_buffer = hardware_params['bucking_buffer_cm']
    
    V_min, x_hover, w_wall, n_shape = target_params
    eval_window = [-eval_limit, eval_limit]
    
    x_bound = R_max + 10.0
    x_array = np.linspace(-x_bound, x_bound, 400)
    
    num_slots = int(np.floor((R_max - R_min) / wire_dia))
    radii = np.linspace(R_min, R_min + (num_slots - 1) * wire_dia, num_slots)
    V_target = generate_target_well(x_array, V_min, x_hover, w_wall, n_shape)
    
    # 1. Initialize the Caching System
    Bx_matrix, Bz_matrix = build_or_load_physics_matrix(x_array, radii, hardware_params)
    
    # 2. Slice the matrices strictly to the evaluation window to save even more time
    eval_mask = (x_array >= eval_window[0]) & (x_array <= eval_window[1])
    Bx_matrix_eval = Bx_matrix[:, :, eval_mask]
    Bz_matrix_eval = Bz_matrix[:, :, eval_mask]
    V_target_eval = V_target[eval_mask]
    
    transition_radius = max(R_min, x_hover - bucking_buffer)
    bounds = []
    for R in radii:
        if R <= transition_radius:
            bounds.append((-max_turns, 0))
        else:
            bounds.append((0, max_turns))
            
    integrality = np.ones(num_slots, dtype=bool) 
    
    print(f"\nStarting evolutionary optimization across {num_slots} physical slots...")
    print("Running on a single core with Disk Caching. Press Ctrl+C AT ANY TIME to halt and save progress!\n")
    
    stop_flag = [False]

    def handle_sigint(sig, frame):
        if not stop_flag[0]:
            print("\n\n[!] Ctrl+C detected! Waiting for the current generation to finish...")
            print("[!] The best coil design will be saved shortly.")
            stop_flag[0] = True
        else:
            print("\n[!] Second Ctrl+C detected. Forcing a hard exit without saving.")
            exit(1) 

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_sigint)
    
    start_time = time.time()
    
    def early_stopping_callback(xk, convergence):
        if stop_flag[0]:
            return True 
            
    try:
        result = differential_evolution(
            cost_function,
            bounds,
            args=(Bx_matrix_eval, Bz_matrix_eval, max_turns, angle_deg, V_target_eval),
            integrality=integrality,
            strategy='best1bin',
            maxiter=10000,      
            popsize=5,         
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            disp=True,         
            workers=1,         
            callback=early_stopping_callback 
        )
    finally:
        signal.signal(signal.SIGINT, original_sigint)
    
    print(f"\nOptimization finished (or halted) in {time.time() - start_time:.1f} seconds.")
    
    best_turns = np.round(result.x).astype(int)
    
    # We call the real physics engine exactly once at the very end to get the full curves for the JSON
    V_sim, V_left, V_right = calculate_simulated_well(
        x_array, radii, best_turns, z_height, angle_deg, wire_dia
    )
    
    peak_val = np.max(V_sim[eval_mask])
    if peak_val > 0:
        V_sim /= peak_val
        V_left /= peak_val
        V_right /= peak_val
        
    return {
        "x_array": x_array,
        "radii": radii,
        "best_turns": best_turns,
        "V_target": V_target,
        "V_sim": V_sim,
        "V_left": V_left,
        "V_right": V_right,
        "eval_window": eval_window, 
        "final_cost": result.fun
    }