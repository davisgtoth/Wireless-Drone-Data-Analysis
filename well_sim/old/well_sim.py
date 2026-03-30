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
    
    # Handle mathematical singularity at x=0
    Bx[x == 0] = 0.0
    return Bx, Bz

def compute_B_fields(x, z, num_rings):
    """HEAVY MATH: Only recalculate this if Z changes."""
    # --- Geometry A: Spiral ---
    Rs_multi = np.arange(10, 10 + num_rings, 1)
    Btot_x_multi = np.zeros_like(x)
    Btot_z_multi = np.zeros_like(x)

    for R_val in Rs_multi:
        Bx, Bz = get_B_components(x, z, R_val)
        Btot_x_multi += Bx
        Btot_z_multi += Bz

    # --- Geometry B: Single Ring (R=10) ---
    Btot_x_single, Btot_z_single = get_B_components(x, z, 10.0)
    
    return Btot_x_multi, Btot_z_multi, Btot_x_single, Btot_z_single

def compute_well_voltages(B_fields, theta):
    """LIGHT MATH: Instantaneous recalculation for Angle and X changes."""
    Btx_m, Btz_m, Btx_s, Btz_s = B_fields

    # Spiral Projections
    Flux_1_multi = Btx_m * np.sin(theta) + Btz_m * np.cos(theta)
    V_1_multi = np.abs(Flux_1_multi) / np.max(np.abs(Flux_1_multi))

    Flux_2_multi = Btx_m * np.sin(-theta) + Btz_m * np.cos(-theta)
    V_2_multi = np.abs(Flux_2_multi) / np.max(np.abs(Flux_2_multi))

    # Single Ring Projections
    Flux_1_single = Btx_s * np.sin(theta) + Btz_s * np.cos(theta)
    V_1_single = np.abs(Flux_1_single) / np.max(np.abs(Flux_1_single))

    Flux_2_single = Btx_s * np.sin(-theta) + Btz_s * np.cos(-theta)
    V_2_single = np.abs(Flux_2_single) / np.max(np.abs(Flux_2_single))
    
    return V_1_multi, V_2_multi, V_1_single, V_2_single

if __name__ == "__main__":
    # 1. Setup Argparse for Terminal Inputs (Removed Angle Arg)
    parser = argparse.ArgumentParser(description="Model the passive stability of a drone in a magnetic well.")
    
    parser.add_argument('-r', '--rings', type=int, default=15, 
                        help='Number of loops in the transmitter spiral (default: 15)')
    parser.add_argument('-w', '--width', type=float, default=8.2, 
                        help='Width of the drone in cm (default: 8.2)')
    parser.add_argument('-s', '--show_single', action='store_true',
                        help='Add this flag to also plot the single R=10 ring comparison')
    
    args = parser.parse_args()

    num_rings = args.rings
    drone_width = args.width
    show_single = args.show_single

    # 2. INITIAL CALCULATION
    x = np.linspace(-30, 30, 300)
    initial_z = 1.0
    initial_drone_x = 0.0
    initial_theta_deg = 45.0
    
    print(f"Calculating well for rings={num_rings}, width={drone_width}cm...")

    # Compute initial fields and voltages
    cached_z = initial_z
    cached_B_fields = compute_B_fields(x, cached_z, num_rings)
    
    initial_theta_rad = initial_theta_deg * np.pi / 180
    V_1_m, V_2_m, V_1_s, V_2_s = compute_well_voltages(cached_B_fields, initial_theta_rad)

    # 3. THE PLOT SETUP
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Made bottom margin larger to fit the second slider
    plt.subplots_adjust(bottom=0.25, right=0.85, left=0.08, top=0.92)

    # Simplified legend labels so they don't look weird when dragging the angle rapidly
    line_m1, = ax.plot(x, V_1_m, label='Spiral Right Coil', color='blue', linewidth=2.5)
    line_m2, = ax.plot(x, V_2_m, label='Spiral Left Coil', color='red', linewidth=2.5)
    
    if show_single:
        line_s1, = ax.plot(x, V_1_s, '--', label='Single R=10 Right', color='dodgerblue', linewidth=2)
        line_s2, = ax.plot(x, V_2_s, '--', label='Single R=10 Left', color='salmon', linewidth=2)

    # Calculate initial Y height using 1/z^2 decay
    initial_decay_factor = (1.0 / initial_z**2 - 0.01) / 0.99
    y_height = 1.05 - 0.35 * initial_decay_factor
    
    drone_line, = ax.plot([initial_drone_x - drone_width/2, initial_drone_x + drone_width/2], 
                          [y_height, y_height], color='black', linestyle=':', linewidth=3, 
                          marker='|', markersize=15, label=f'Drone ({drone_width} cm)')
    
    drop_left = ax.axvline(initial_drone_x - drone_width/2, color='red', linestyle=':', alpha=0.5)
    drop_right = ax.axvline(initial_drone_x + drone_width/2, color='blue', linestyle=':', alpha=0.5)

    ax.set_title(f'Passive Stability | Drone x={initial_drone_x:.1f} cm | Height z={initial_z:.1f} cm | Angle: {initial_theta_deg:.1f}°')
    ax.set_xlabel('Distance from Center (x-axis)')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 1.2) # Bumped up slightly to fit max 1.05 height cleanly
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.4)

    # 4. THE MATPLOTLIB SLIDERS
    # Horizontal slider for X position
    ax_x_slider = plt.axes([0.15, 0.12, 0.65, 0.03])
    drone_slider = Slider(
        ax=ax_x_slider,
        label='Drag Drone (x): ',
        valmin=-20.0,
        valmax=20.0,
        valinit=initial_drone_x,
        valstep=0.2
    )

    # Horizontal slider for Angle (tucked below the X slider)
    ax_angle_slider = plt.axes([0.15, 0.05, 0.65, 0.03])
    angle_slider = Slider(
        ax=ax_angle_slider,
        label='Coil Angle (°): ',
        valmin=0.0,
        valmax=90.0,
        valinit=initial_theta_deg,
        valstep=1.0,
        color='purple'
    )

    # Vertical slider for Z height
    ax_z_slider = plt.axes([0.9, 0.25, 0.02, 0.6])
    z_slider = Slider(
        ax=ax_z_slider,
        label='Z Height ',
        valmin=1.0, 
        valmax=10.0,
        valinit=initial_z,
        valstep=0.1,
        orientation='vertical'
    )

    def update(val):
        global cached_z, cached_B_fields
        
        current_x = drone_slider.val
        current_z = z_slider.val
        current_theta_deg = angle_slider.val
        current_theta_rad = current_theta_deg * np.pi / 180
        
        # 1. Update Drone visual lines
        decay_factor = (1.0 / current_z**2 - 0.01) / 0.99 
        current_y = 1.05 - 0.35 * decay_factor
        
        drone_line.set_xdata([current_x - drone_width/2, current_x + drone_width/2])
        drone_line.set_ydata([current_y, current_y]) 
        
        drop_left.set_xdata([current_x - drone_width/2, current_x - drone_width/2])
        drop_right.set_xdata([current_x + drone_width/2, current_x + drone_width/2])
        
        # 2. Check if Z changed. If yes, run the heavy math.
        if current_z != cached_z:
            cached_B_fields = compute_B_fields(x, current_z, num_rings)
            cached_z = current_z
            
        # 3. Always run the fast projection math
        new_v1_m, new_v2_m, new_v1_s, new_v2_s = compute_well_voltages(cached_B_fields, current_theta_rad)
        
        line_m1.set_ydata(new_v1_m)
        line_m2.set_ydata(new_v2_m)
        
        if show_single:
            line_s1.set_ydata(new_v1_s)
            line_s2.set_ydata(new_v2_s)
        
        # 4. Update title
        ax.set_title(f'Passive Stability | Drone x={current_x:.1f} cm | Height z={current_z:.1f} cm | Angle: {current_theta_deg:.0f}°')
        fig.canvas.draw_idle()

    drone_slider.on_changed(update)
    angle_slider.on_changed(update)
    z_slider.on_changed(update)

    plt.show()