import numpy as np
import environment

# ---------------------------------------------------------
# KINEMATICS (POSITION RESOLUTION)
# ---------------------------------------------------------

def calculate_kinematics(state, params):
    """
    Unpacks the state vector and calculates the global (X, Z) coordinates 
    for both the pickup coils and the motors in standard SI units (meters).
    """
    x, z, theta, vx, vz, omega = state
    r_coil = params['r_coil']
    r_motor = params['r_motor']

    # Coil positions (Where power is harvested from the B-field)
    x_coil_left = x - r_coil * np.cos(theta)
    z_coil_left = z - r_coil * np.sin(theta)
    
    x_coil_right = x + r_coil * np.cos(theta)
    z_coil_right = z + r_coil * np.sin(theta)

    # Motor positions (Where thrust is applied to the chassis)
    x_motor_left = x - r_motor * np.cos(theta)
    z_motor_left = z - r_motor * np.sin(theta)
    
    x_motor_right = x + r_motor * np.cos(theta)
    z_motor_right = z + r_motor * np.sin(theta)

    return {
        'coil_L': (x_coil_left, z_coil_left),
        'coil_R': (x_coil_right, z_coil_right),
        'motor_L': (x_motor_left, z_motor_left),
        'motor_R': (x_motor_right, z_motor_right)
    }

# ---------------------------------------------------------
# DYNAMICS (THE PHYSICS PLANT)
# ---------------------------------------------------------

def state_derivatives(t, state, params, env_data):
    """
    The core plant passed to scipy.integrate.solve_ivp.
    Calculates forces, torques, and returns the state derivatives.
    """
    x, z, theta, vx, vz, omega = state
    
    mass = params['mass']
    inertia = params['inertia']
    g = params['gravity']
    c_lin = params['c_linear']
    c_ang = params['c_angular']
    r_motor = params['r_motor']

    # 1. Get positions in meters
    kinematics = calculate_kinematics(state, params)
    (xc_L, zc_L) = kinematics['coil_L']
    (xc_R, zc_R) = kinematics['coil_R']

    # 2. Convert meters to millimeters for the LUT queries
    r_target_L_mm = xc_L * 1000.0
    z_target_L_mm = zc_L * 1000.0
    r_target_R_mm = xc_R * 1000.0
    z_target_R_mm = zc_R * 1000.0

    # 3. Query forces in millinewtons (mN) from the environment
    f_L_mN = environment.get_coil_lift(r_target_L_mm, z_target_L_mm, theta, True, env_data)
    f_R_mN = environment.get_coil_lift(r_target_R_mm, z_target_R_mm, theta, False, env_data)
    
    # 4. Convert millinewtons to standard Newtons (N) for physics math
    F_left = f_L_mN / 1000.0
    F_right = f_R_mN / 1000.0

    # 5. Force Resolution (Normal to Chassis)
    # A perfectly vertical force vectors against a tilted drone 
    # translates to global X and Z components.
    sum_Fx = -(F_left + F_right) * np.sin(theta) - (c_lin * vx)
    sum_Fz =  (F_left + F_right) * np.cos(theta) - (mass * g) - (c_lin * vz)

    # 6. Torque Calculation
    # Upward force on the Right side causes +theta (Counter-Clockwise) rotation.
    # Upward force on the Left side causes -theta (Clockwise) rotation.
    sum_tau = (F_right * r_motor) - (F_left * r_motor) - (c_ang * omega)

    # 7. Accelerations (F = ma -> a = F/m)
    ax = sum_Fx / mass
    az = sum_Fz / mass
    alpha = sum_tau / inertia

    return [vx, vz, omega, ax, az, alpha]

# ---------------------------------------------------------
# TERMINAL EVENTS (STOP CONDITIONS)
# ---------------------------------------------------------

def out_of_bounds_event(t, state, params, env_data):
    """
    Triggers when the drone escapes the measured well (150 mm).
    scipy.integrate looks for when this function's return value crosses zero.
    """
    x = state[0]
    boundary_meters = 0.15 # 150 mm
    
    # Returns a positive number inside the well, drops below 0 outside it.
    return boundary_meters - abs(x)

# Tell SciPy to stop the simulation when this event occurs
out_of_bounds_event.terminal = True
# Only trigger when going from positive (inside well) to negative (outside well)
out_of_bounds_event.direction = -1