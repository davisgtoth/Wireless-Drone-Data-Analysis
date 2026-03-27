import numpy as np
from scipy.integrate import solve_ivp
import environment
import physics
import visualizer
import time

def main():
    print("Loading environmental data...")
    env_data = environment.load_environment(
        path_flat='data/flat.csv',
        path_45_low='data/45_low.csv',
        path_45_high='data/45_high.csv'
    )
    
    # --- NEW: APPLY HOVER OFFSET TO RAW DATA ---
    mass_kg = 0.040  # 40 grams
    weight_mN = mass_kg * 9.81 * 1000.0
    hover_offset_mN = weight_mN / 2.0  # Approx 196.2 mN per coil
    
    # Shift the 12.5mm well data up by the hover requirement
    env_data['rx_flat'] += hover_offset_mN
    env_data['rx_45_low'] += hover_offset_mN
    
    # Shift the 37.5mm well data up, but scaled down by your physical law
    # so the interpolation doesn't flatten out the vertical gradient
    scale_factor_37 = (12.5 / 37.5)**0.4
    env_data['rx_45_high'] += (hover_offset_mN * scale_factor_37)
    
    print(f"Data loaded. Applied {hover_offset_mN:.1f} mN baseline hover offset.")
    # -------------------------------------------

    # 1. Define Physical Parameters
    # ---------------------------------------------------------
    r_coil_m = 40e-3   
    r_motor_m = 40e-3  
    
    # Dynamically calculate the total width of the drone
    drone_width_m = 2.0 * max(r_coil_m, r_motor_m)
    
    # Dynamically calculate Rotational Inertia (Thin rod approximation)
    inertia_kgm2 = (1.0 / 12.0) * mass_kg * (drone_width_m)**2

    params = {
        'mass': mass_kg,
        'inertia': inertia_kgm2,
        'r_coil': r_coil_m,
        'r_motor': r_motor_m,
        'gravity': 9.81,
        'c_linear': 0.005,
        'c_angular': 0.0001
    }
    
    # ... rest of main.py remains exactly the same

    # 2. Define Initial Conditions (The "Drop Test")
    # ---------------------------------------------------------
    # State = [x, z, theta, vx, vz, omega]
    x0 = 0.01              # Start 10mm off-center (to test radial stability)
    z0 = 0.025             # Drop from 25mm height
    theta0 = np.radians(5) # Start with a 5-degree tilt (to test torque stability)
    vx0, vz0, omega0 = 0.0, 0.0, 0.0
    
    initial_state = [x0, z0, theta0, vx0, vz0, omega0]

    # 3. Setup the Integrator
    # ---------------------------------------------------------
    t_span = (0, 10.0)  # Simulate 3 seconds of flight
    fps = 60
    t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) * fps))

    print("Starting simulation...")
    start_time = time.time()
    
    # Run the physics engine!
    solution = solve_ivp(
        fun=physics.state_derivatives,
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        events=physics.out_of_bounds_event,
        args=(params, env_data),
        method='RK45',          # Runge-Kutta 4(5) is excellent for rigid body dynamics
        rtol=1e-6, atol=1e-8    # Tight tolerances to prevent numerical energy leaks
    )
    
    calc_time = time.time() - start_time
    print(f"Simulation completed in {calc_time:.3f} seconds.")
    
    # 4. Evaluate the Results
    # ---------------------------------------------------------
    if solution.status == 1:
        print(f"TERMINATION: Drone fell out of the well at t={solution.t[-1]:.2f}s")
    elif solution.status == 0:
        print("SUCCESS: Drone remained in the well for the full duration.")
    
    # 5. Hand off to the Visualizer
    # ---------------------------------------------------------
    print("Launching visualizer...")
    visualizer.animate_flight(solution.t, solution.y, params, env_data)

if __name__ == "__main__":
    main()