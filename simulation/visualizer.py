import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import environment

def animate_flight(time_array, state_history, params, env_data):
    """
    Takes the state history from the ODE solver and creates a 60 FPS animation
    of the drone flying inside the magnetic well.
    """
    # 1. Unpack parameters and convert to mm for visualization
    r_coil_mm = params['r_coil'] * 1000.0
    r_motor_mm = params['r_motor'] * 1000.0
    max_r_mm = max(r_coil_mm, r_motor_mm)
    
    # 2. Setup the Figure and Axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlim(-150, 150)
    ax1.set_ylim(-10, 100)  # View from 10mm below ground up to 100mm height
    ax1.set_xlabel('Radial Position X (mm)')
    ax1.set_ylabel('Height Z (mm)', color='black')
    ax1.set_title('Passive Magnetic Drone Dynamics Simulation')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box') # Keeps the drone from looking stretched

    # 3. Plot the Background Environment (Secondary Y-Axis)
    # We plot the force profile on a twin axis so its scale (mN) doesn't warp 
    # the physical height scale (mm) of the primary axis.
    ax2 = ax1.twinx()
    ax2.set_ylabel('Magnetic Well Force Profile (mN)', color='blue')
    ax2.plot(env_data['r_45_low'], env_data['rx_45_low'], 'b--', alpha=0.2, label='Well Profile (z=12.5)')
    
    # --- FIX: DYNAMIC Y-LIMIT (ZOOMED IN) ---
    min_well_force = np.min(env_data['rx_45_low'])
    max_well_force = np.max(env_data['rx_45_low'])
    well_depth = max_well_force - min_well_force
    
    # Set the bottom of the axis slightly below the well, and the top 
    # way above it so the well stays in the lower portion of the screen
    ax2.set_ylim(min_well_force - (well_depth * 0.5), max_well_force + (well_depth * 4))
    # ----------------------------------------

    ax2.tick_params(axis='y', labelcolor='blue')

    # 4. Initialize the Animation Artists (Empty objects to be updated)
    drone_line, = ax1.plot([], [], 'k-', lw=4, label='Drone Chassis')
    coils_scatter, = ax1.plot([], [], 'bo', ms=8, label='Pickup Coils')
    
    # Force vectors (Red lines)
    force_L_line, = ax1.plot([], [], 'r-', lw=2)
    force_R_line, = ax1.plot([], [], 'r-', lw=2, label='Thrust Vectors')
    
    # Text overlay for telemetry
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    state_text = ax1.text(0.02, 0.90, '', transform=ax1.transAxes)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    # Visual scaling factor for the thrust arrows (mm of arrow length per mN of force)
    arrow_scale = 0.5 

    # 5. The Update Function (Runs once per frame)
    def update(frame):
        # Extract state for this frame and convert meters to mm
        x_m, z_m, theta, _, _, _ = state_history[:, frame]
        x_mm = x_m * 1000.0
        z_mm = z_m * 1000.0
        
        # --- KINEMATICS ---
        # Chassis endpoints
        x_left = x_mm - max_r_mm * np.cos(theta)
        z_left = z_mm - max_r_mm * np.sin(theta)
        x_right = x_mm + max_r_mm * np.cos(theta)
        z_right = z_mm + max_r_mm * np.sin(theta)
        
        # Coil positions
        xc_L = x_mm - r_coil_mm * np.cos(theta)
        zc_L = z_mm - r_coil_mm * np.sin(theta)
        xc_R = x_mm + r_coil_mm * np.cos(theta)
        zc_R = z_mm + r_coil_mm * np.sin(theta)
        
        # Motor positions
        xm_L = x_mm - r_motor_mm * np.cos(theta)
        zm_L = z_mm - r_motor_mm * np.sin(theta)
        xm_R = x_mm + r_motor_mm * np.cos(theta)
        zm_R = z_mm + r_motor_mm * np.sin(theta)

        # --- UPDATE ARTISTS ---
        # 1. Update Drone Chassis
        drone_line.set_data([x_left, x_right], [z_left, z_right])
        
        # 2. Update Coils
        coils_scatter.set_data([xc_L, xc_R], [zc_L, zc_R])
        
        # 3. Calculate Forces to scale the arrows
        # Re-query the environment so we can draw the arrows correctly
        f_L_mN = environment.get_coil_lift(xc_L, zc_L, theta, True, env_data)
        f_R_mN = environment.get_coil_lift(xc_R, zc_R, theta, False, env_data)
        
        # Apply visual scale to the magnitude
        mag_L = f_L_mN * arrow_scale
        mag_R = f_R_mN * arrow_scale
        
        # Normal vector components (pointing "UP" relative to tilted drone)
        nx = -np.sin(theta)
        nz =  np.cos(theta)
        
        # Update Left Force Arrow (from motor pos to motor pos + force vector)
        force_L_line.set_data([xm_L, xm_L + nx * mag_L], [zm_L, zm_L + nz * mag_L])
        
        # Update Right Force Arrow
        force_R_line.set_data([xm_R, xm_R + nx * mag_R], [zm_R, zm_R + nz * mag_R])

        # 4. Update Telemetry Text
        time_text.set_text(f'Time: {time_array[frame]:.2f} s')
        state_text.set_text(f'X: {x_mm:5.1f} | Z: {z_mm:5.1f} | Tilt: {np.degrees(theta):5.1f}°')

        return drone_line, coils_scatter, force_L_line, force_R_line, time_text, state_text

    # 6. Build and save the animation
    fps = 60
    interval_ms = 1000.0 / fps
    
    print("Rendering animation frames...")
    ani = animation.FuncAnimation(
        fig, update, frames=len(time_array),
        interval=interval_ms, blit=True, repeat=False
    )
    
    # --- NEW FFmpeg SAVING LOGIC ---
    output_file = 'drone_flight.mp4'
    print(f"Saving video to '{output_file}' (this may take a few seconds)...")
    
    # Configure the FFmpeg writer (adjust bitrate for quality/file size)
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Drone Simulator'), bitrate=2000)
    
    # Save the file!
    ani.save(output_file, writer=writer)
    print("Video saved successfully!")
    # -------------------------------
    
    plt.tight_layout()
    plt.show()  # Keep this if you still want the live window to pop up, or comment it out if you only want the MP4