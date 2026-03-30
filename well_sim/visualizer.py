import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import ellipk, ellipe

# Import your target generator
from target import generate_target_well

def get_B_components(x, z, R):
    """Calculates the exact Bx and Bz components for a single circular loop."""
    x_safe = np.where(x == 0, 1e-10, x)
    k_squared = (4 * R * x_safe) / ((R + x_safe)**2 + z**2)
    K = ellipk(k_squared)
    E = ellipe(k_squared)
    
    root_term = np.sqrt((R + x_safe)**2 + z**2)
    alpha_squared = (R - x_safe)**2 + z**2
    
    Bx = (z / (x_safe * root_term)) * (-K + ((R**2 + x_safe**2 + z**2) / alpha_squared) * E)
    Bz = (1 / root_term) * (K + ((R**2 - x_safe**2 - z**2) / alpha_squared) * E)
    
    Bx[x == 0] = 0.0
    return Bx, Bz

def compute_well_profiles(x, z_height, theta_deg, radii, turns_array, wire_diameter):
    """Computes the conjoined well profile using the accurate 3D downward stacking math."""
    Btot_x = np.zeros_like(x, dtype=float)
    Btot_z = np.zeros_like(x, dtype=float)
    
    for R, N in zip(radii, turns_array):
        if N == 0:
            continue
            
        direction = np.sign(N)
        num_turns = int(abs(N))
        
        for i in range(num_turns):
            z_i = z_height + (i * wire_diameter)
            Bx, Bz = get_B_components(x, z_i, R)
            
            Btot_x += direction * Bx
            Btot_z += direction * Bz
            
    theta = np.radians(theta_deg)
    
    Flux_right = Btot_x * np.sin(theta) + Btot_z * np.cos(theta)
    V_right = np.abs(Flux_right)
    
    Flux_left = Btot_x * np.sin(-theta) + Btot_z * np.cos(-theta)
    V_left = np.abs(Flux_left)
    
    max_val = max(np.max(V_right), np.max(V_left))
    if max_val > 0:
        V_right = V_right / max_val
        V_left = V_left / max_val
        
    return V_left, V_right

def main():
    # 1. Setup Argparse
    parser = argparse.ArgumentParser(description="Visualize an optimized magnetic well from a JSON payload.")
    parser.add_argument('file', type=str, help='Path to the JSON output file')
    parser.add_argument('-w', '--width', type=float, default=8.2, help='Width of the drone in cm (default: 8.2)')
    args = parser.parse_args()

    # 2. Load the JSON Data
    with open(args.file, 'r') as f:
        data = json.load(f)

    hw = data["hardware"]
    coil = data["optimized_coil"]
    
    # Extract Target Well parameters
    target = data["target_well"]
    V_min = target["V_min"]
    x_hover = target["x_hover"]
    w_wall = target["w_wall"]
    n_shape = target["n_shape"]
    
    theta_deg = hw["angle_deg"]
    wire_dia = hw["wire_diameter_cm"]
    initial_z = hw["z_height_cm"]
    eval_limit = hw.get("eval_limit_cm", 20.0)
    
    radii = np.array(coil["radii_cm"])
    turns = np.array(coil["turns"])
    
    drone_width = args.width
    initial_drone_x = 0.0
    x_array = np.linspace(-35, 35, 400)

    print(f"Loaded {args.file}...")
    print(f"Visualizing with drone width = {drone_width} cm.")

    # 3. Setup the Figure and Subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(bottom=0.25, right=0.85, left=0.08, top=0.92, hspace=0.3)

    # --- TOP PLOT: The Interactive Well ---
    # Generate and plot the static target well background
    V_target = generate_target_well(x_array, V_min, x_hover, w_wall, n_shape)
    ax1.plot(x_array, V_target, color='black', linewidth=3, linestyle=':', label="Target Well", zorder=1)

    # Compute and plot the simulated curves
    V_left, V_right = compute_well_profiles(x_array, initial_z, theta_deg, radii, turns, wire_dia)
    line_left, = ax1.plot(x_array, V_left, label=f'Left Motor V_sim (-{theta_deg}°)', color='red', linewidth=2.5, zorder=2)
    line_right, = ax1.plot(x_array, V_right, label=f'Right Motor V_sim (+{theta_deg}°)', color='blue', linewidth=2.5, zorder=2)
    
    # Drone visualization lines
    y_height = 1.05
    drone_line, = ax1.plot([initial_drone_x - drone_width/2, initial_drone_x + drone_width/2], 
                           [y_height, y_height], color='black', linestyle='-', linewidth=4, 
                           marker='|', markersize=15, label='Drone Receiver Width')
    
    drop_left = ax1.axvline(initial_drone_x - drone_width/2, color='red', linestyle=':', alpha=0.5)
    drop_right = ax1.axvline(initial_drone_x + drone_width/2, color='blue', linestyle=':', alpha=0.5)

    ax1.axvspan(-eval_limit, eval_limit, color='gray', alpha=0.1, label='Optimization Window')

    ax1.set_title(f'Optimized Well Profile | Drone x={initial_drone_x:.1f} cm | Flight Height z={initial_z:.1f} cm')
    ax1.set_ylabel('Normalized Flux (Voltage)')
    ax1.set_xlim(x_array[0], x_array[-1])
    ax1.set_ylim(0, 1.15)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.4)

    # --- BOTTOM PLOT: The Physical Winding Histogram ---
    colors = ['mediumseagreen' if t >= 0 else 'coral' for t in turns]
    wire_width = radii[1] - radii[0] if len(radii) > 1 else wire_dia
    
    ax2.bar(radii, turns, width=wire_width*0.8, color=colors, edgecolor='black')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title("Physical Winding Profile (Cross-Section)")
    ax2.set_xlabel("Radius (cm)")
    ax2.set_ylabel("Number of Turns")
    ax2.set_xlim(radii[0] - 1, radii[-1] + 1)
    ax2.grid(axis='y', alpha=0.4)

    # 4. Setup the Matplotlib Sliders
    ax_x_slider = plt.axes([0.15, 0.12, 0.65, 0.03])
    drone_slider = Slider(
        ax=ax_x_slider,
        label='Slide Drone (x): ',
        valmin=-25.0,
        valmax=25.0,
        valinit=initial_drone_x,
        valstep=0.2
    )

    ax_z_slider = plt.axes([0.9, 0.35, 0.02, 0.5])
    z_slider = Slider(
        ax=ax_z_slider,
        label='Flight Height (z) ',
        valmin=1.0, 
        valmax=15.0,
        valinit=initial_z,
        valstep=0.2,
        orientation='vertical'
    )

    # 5. The Update Function
    def update(val):
        current_x = drone_slider.val
        current_z = z_slider.val
        
        # Update drone visual position
        drone_line.set_xdata([current_x - drone_width/2, current_x + drone_width/2])
        drop_left.set_xdata([current_x - drone_width/2, current_x - drone_width/2])
        drop_right.set_xdata([current_x + drone_width/2, current_x + drone_width/2])
        
        # Recalculate physics based on new Z height
        new_v_left, new_v_right = compute_well_profiles(x_array, current_z, theta_deg, radii, turns, wire_dia)
        
        line_left.set_ydata(new_v_left)
        line_right.set_ydata(new_v_right)
        
        ax1.set_title(f'Optimized Well Profile | Drone x={current_x:.1f} cm | Flight Height z={current_z:.1f} cm')
        fig.canvas.draw_idle()

    drone_slider.on_changed(update)
    z_slider.on_changed(update)

    plt.show()

if __name__ == "__main__":
    main()