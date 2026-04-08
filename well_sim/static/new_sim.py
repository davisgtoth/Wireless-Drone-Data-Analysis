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

def compute_B_fields(x, z, num_rings, show_single=False):
    """Calculates the B-field for a specific coil at a specific Z height."""
    Rs_multi = np.arange(10, 10 + num_rings, 1)
    Btot_x_multi = np.zeros_like(x)
    Btot_z_multi = np.zeros_like(x)

    for R_val in Rs_multi:
        Bx, Bz = get_B_components(x, z, R_val)
        Btot_x_multi += Bx
        Btot_z_multi += Bz

    if show_single:
        Btot_x_single, Btot_z_single = get_B_components(x, z, 10.0)
        return Btot_x_multi, Btot_z_multi, Btot_x_single, Btot_z_single
    
    return Btot_x_multi, Btot_z_multi, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model the passive stability of a drone in a magnetic well.")
    
    parser.add_argument('-a', '--angle', type=float, required=True, 
                        help='Base motor/coil mount angle in degrees (e.g., 45)')
    parser.add_argument('-r', '--rings', type=int, default=15, 
                        help='Number of loops in the transmitter spiral (default: 15)')
    parser.add_argument('-w', '--width', type=float, default=8.2, 
                        help='Width of the drone in cm (default: 8.2)')
    parser.add_argument('-s', '--show_single', action='store_true',
                        help='Add this flag to also plot the single R=10 ring comparison')
    
    args = parser.parse_args()

    base_theta_deg = args.angle
    base_theta_rad = base_theta_deg * np.pi / 180
    num_rings = args.rings
    drone_width = args.width
    show_single = args.show_single

    # --- PLOT SETUP ---
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.25, right=0.85, left=0.08, top=0.92)

    x_grid = np.linspace(-30, 30, 300)

    # Initialize empty lines
    line_m1, = ax.plot([], [], label=f'Spiral Right Coil (+{base_theta_deg}°)', color='blue', linewidth=2.5)
    line_m2, = ax.plot([], [], label=f'Spiral Left Coil (-{base_theta_deg}°)', color='red', linewidth=2.5)
    
    if show_single:
        line_s1, = ax.plot([], [], '--', label='Single R=10 Right', color='dodgerblue', linewidth=2)
        line_s2, = ax.plot([], [], '--', label='Single R=10 Left', color='salmon', linewidth=2)

    drone_line, = ax.plot([], [], color='black', linestyle='-', linewidth=4, 
                          marker='o', markersize=8, label=f'Drone ({drone_width} cm)')
    
    drop_left = ax.axvline(0, color='red', linestyle=':', alpha=0.5)
    drop_right = ax.axvline(0, color='blue', linestyle=':', alpha=0.5)

    ax.set_xlabel('Distance from Center (x-axis)')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 1.2)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.4)

    # --- SLIDERS ---
    ax_x_slider = plt.axes([0.15, 0.12, 0.65, 0.03])
    drone_slider = Slider(ax=ax_x_slider, label='Drone CoM (x): ', valmin=-20.0, valmax=20.0, valinit=0.0, valstep=0.2)

    ax_tilt_slider = plt.axes([0.15, 0.05, 0.65, 0.03])
    tilt_slider = Slider(ax=ax_tilt_slider, label='Tilt / Roll (°): ', valmin=-45.0, valmax=45.0, valinit=0.0, valstep=1.0, color='purple')

    ax_z_slider = plt.axes([0.9, 0.25, 0.02, 0.6])
    z_slider = Slider(ax=ax_z_slider, label='CoM Z Height ', valmin=1.0, valmax=10.0, valinit=1.0, valstep=0.1, orientation='vertical')

    def map_z_to_visual_y(z_val):
        """Helper to map physical Z to visual Y using your inverse square law."""
        # Cap minimum z at 0.5 to prevent visual line from shooting completely off-screen
        safe_z = max(0.5, z_val)
        decay = (1.0 / safe_z**2 - 0.01) / 0.99
        return 1.05 - 0.35 * decay

    def update(val):
        X_center = drone_slider.val
        Z_center = z_slider.val
        tilt_deg = tilt_slider.val
        phi = tilt_deg * np.pi / 180
        
        # 1. Kinematics: Calculate actual spatial positions of the coils
        # Left coil moves left and down (if CCW roll)
        X_L_grid = x_grid - (drone_width/2) * np.cos(phi)
        Z_L = Z_center - (drone_width/2) * np.sin(phi)
        
        # Right coil moves right and up (if CCW roll)
        X_R_grid = x_grid + (drone_width/2) * np.cos(phi)
        Z_R = Z_center + (drone_width/2) * np.sin(phi)

        # Protect against mathematical singularities exactly on the pad
        Z_L = max(0.1, Z_L)
        Z_R = max(0.1, Z_R)

        # 2. Physics: Evaluate the magnetic well for the two different heights
        Bx_L, Bz_L, Bx_L_s, Bz_L_s = compute_B_fields(X_L_grid, Z_L, num_rings, show_single)
        Bx_R, Bz_R, Bx_R_s, Bz_R_s = compute_B_fields(X_R_grid, Z_R, num_rings, show_single)

        # 3. Projection: Modify absolute angles based on tilt
        alpha_L = -base_theta_rad + phi
        alpha_R = base_theta_rad + phi

        # Calculate Left Coil Voltage Profile
        Flux_L = Bx_L * np.sin(alpha_L) + Bz_L * np.cos(alpha_L)
        V_L = np.abs(Flux_L) / np.max(np.abs(Flux_L))

        # Calculate Right Coil Voltage Profile
        Flux_R = Bx_R * np.sin(alpha_R) + Bz_R * np.cos(alpha_R)
        V_R = np.abs(Flux_R) / np.max(np.abs(Flux_R))

        line_m1.set_data(x_grid, V_R)
        line_m2.set_data(x_grid, V_L)

        if show_single:
            Flux_L_s = Bx_L_s * np.sin(alpha_L) + Bz_L_s * np.cos(alpha_L)
            V_L_s = np.abs(Flux_L_s) / np.max(np.abs(Flux_L_s))
            
            Flux_R_s = Bx_R_s * np.sin(alpha_R) + Bz_R_s * np.cos(alpha_R)
            V_R_s = np.abs(Flux_R_s) / np.max(np.abs(Flux_R_s))
            
            line_s1.set_data(x_grid, V_R_s)
            line_s2.set_data(x_grid, V_L_s)

        # 4. Visuals: Update the drone line on the screen
        x_drone_left = X_center - (drone_width/2) * np.cos(phi)
        x_drone_right = X_center + (drone_width/2) * np.cos(phi)
        
        # Map their specific Z-heights to the visual Y-axis bounds
        y_visual_left = map_z_to_visual_y(Z_L)
        y_visual_right = map_z_to_visual_y(Z_R)

        drone_line.set_xdata([x_drone_left, x_drone_right])
        drone_line.set_ydata([y_visual_left, y_visual_right]) 
        
        drop_left.set_xdata([x_drone_left, x_drone_left])
        drop_right.set_xdata([x_drone_right, x_drone_right])

        ax.set_title(f'Drone Dynamics | CoM X={X_center:.1f} cm | CoM Z={Z_center:.1f} cm | Roll Tilt={tilt_deg:.1f}°')
        fig.canvas.draw_idle()

    # Trigger initial draw
    update(0)

    drone_slider.on_changed(update)
    tilt_slider.on_changed(update)
    z_slider.on_changed(update)

    plt.show()