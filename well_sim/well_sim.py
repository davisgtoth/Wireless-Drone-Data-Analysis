import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import ellipk, ellipe

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

if __name__ == "__main__":
    # 1. Setup Argparse for Terminal Inputs
    parser = argparse.ArgumentParser(description="Model the passive stability of a drone in a magnetic well.")
    
    parser.add_argument('-a', '--angle', type=float, required=True, 
                        help='Motor/Coil mount angle in degrees (e.g., 45)')
    parser.add_argument('-r', '--rings', type=int, default=15, 
                        help='Number of loops in the transmitter spiral (default: 15)')
    parser.add_argument('-w', '--width', type=float, default=8.2, 
                        help='Width of the drone in cm (default: 8.2)')
    
    args = parser.parse_args()

    theta_deg = args.angle
    theta = theta_deg * np.pi / 180
    num_rings = args.rings
    drone_width = args.width

    # 2. THE HEAVY MATH
    print(f"Calculating well for angle={theta_deg}°, rings={num_rings}, width={drone_width}cm...")
    x = np.linspace(-30, 30, 300)
    z = 1.0

    # --- Geometry A: Spiral ---
    Rs_multi = np.arange(10, 10 + num_rings, 1)
    Btot_x_multi = np.zeros_like(x)
    Btot_z_multi = np.zeros_like(x)

    for R_val in Rs_multi:
        Bx, Bz = get_B_components(x, z, R_val)
        Btot_x_multi += Bx
        Btot_z_multi += Bz

    Flux_1_multi = Btot_x_multi * np.sin(theta) + Btot_z_multi * np.cos(theta)
    V_1_multi = np.abs(Flux_1_multi) / np.max(np.abs(Flux_1_multi))

    Flux_2_multi = Btot_x_multi * np.sin(-theta) + Btot_z_multi * np.cos(-theta)
    V_2_multi = np.abs(Flux_2_multi) / np.max(np.abs(Flux_2_multi))

    # --- Geometry B: Single Ring (R=10) ---
    Btot_x_single, Btot_z_single = get_B_components(x, z, 10.0)

    Flux_1_single = Btot_x_single * np.sin(theta) + Btot_z_single * np.cos(theta)
    V_1_single = np.abs(Flux_1_single) / np.max(np.abs(Flux_1_single))

    Flux_2_single = Btot_x_single * np.sin(-theta) + Btot_z_single * np.cos(-theta)
    V_2_single = np.abs(Flux_2_single) / np.max(np.abs(Flux_2_single))

    # 3. THE PLOT SETUP
    # INCREASED FIGSIZE: Now 14x7 for a nice widescreen view
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # ADJUSTED MARGINS: bottom=0.25 (for slider), right=0.75 (for the legend)
    # plt.subplots_adjust(bottom=0.25, right=0.75)
    # plt.subplots_adjust(bottom=0.25, right=0.75, left=0.05, top=0.92)
    plt.subplots_adjust(bottom=0.18, right=0.79, left=0.05, top=0.92)

    # Plot static curves
    ax.plot(x, V_1_multi, label=f'Spiral Right (+{theta_deg}°)', color='blue', linewidth=2.5)
    ax.plot(x, V_2_multi, label=f'Spiral Left (-{theta_deg}°)', color='red', linewidth=2.5)
    ax.plot(x, V_1_single, '--', label=f'Single R=10 Right (+{theta_deg}°)', color='dodgerblue', linewidth=2)
    ax.plot(x, V_2_single, '--', label=f'Single R=10 Left (-{theta_deg}°)', color='salmon', linewidth=2)

    # Plot the initial movable drone
    initial_drone_x = 0.0
    y_height = 0.7
    
    drone_line, = ax.plot([initial_drone_x - drone_width/2, initial_drone_x + drone_width/2], 
                          [y_height, y_height], color='black', linestyle=':', linewidth=3, 
                          marker='|', markersize=15, label=f'Drone ({drone_width} cm)')
    
    drop_left = ax.axvline(initial_drone_x - drone_width/2, color='red', linestyle=':', alpha=0.5)
    drop_right = ax.axvline(initial_drone_x + drone_width/2, color='blue', linestyle=':', alpha=0.5)

    # Formatting
    ax.set_title(f'Passive Stability Comparison | Drone x = {initial_drone_x:.1f} cm')
    ax.set_xlabel('Distance from Center (x-axis)')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 1.1)
    
    # LEGEND MOVED: Placed outside the plot area on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    ax.grid(alpha=0.4)

    # 4. THE MATPLOTLIB SLIDER
    # Keep the slider centered under the main plot (ignoring the legend area)
    # ax_slider = plt.axes([0.15, 0.1, 0.55, 0.03])
    ax_slider = plt.axes([0.15, 0.05, 0.55, 0.03])
    drone_slider = Slider(
        ax=ax_slider,
        label='Drag Drone (x): ',
        valmin=-20.0,
        valmax=20.0,
        valinit=initial_drone_x,
        valstep=0.2
    )

    def update(val):
        current_x = drone_slider.val
        
        # Update lines
        drone_line.set_xdata([current_x - drone_width/2, current_x + drone_width/2])
        drop_left.set_xdata([current_x - drone_width/2, current_x - drone_width/2])
        drop_right.set_xdata([current_x + drone_width/2, current_x + drone_width/2])
        
        # Update title
        ax.set_title(f'Passive Stability Comparison | Drone x = {current_x:.1f} cm')
        fig.canvas.draw_idle()

    drone_slider.on_changed(update)
    plt.show()