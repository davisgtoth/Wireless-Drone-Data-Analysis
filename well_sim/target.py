import numpy as np

def generate_target_well(x_array, V_min, x_hover, w_wall, n):
    """
    Generates the symmetric, parameterized piecewise target magnetic well.
    
    Parameters:
    - x_array: NumPy array of x-coordinates (e.g., from -20 to 20 cm)
    - V_min: Baseline normalized voltage at the center (x=0)
    - x_hover: Radius defining the flat center floor
    - w_wall: Horizontal width over which the voltage ramps from V_min to 1.0
    - n: Polynomial shape exponent for the wall curvature
    
    Returns:
    - V_target: NumPy array of the idealized well profile
    """
    
    # Exploit symmetry: evaluate based on the absolute distance from the center
    x_abs = np.abs(x_array)
    
    # Initialize the output array
    V_target = np.zeros_like(x_array, dtype=float)
    
    # ---------------------------------------------------------
    # ZONE 1: The Flat Floor (Hover Zone)
    # ---------------------------------------------------------
    mask_floor = x_abs <= x_hover
    V_target[mask_floor] = V_min
    
    # ---------------------------------------------------------
    # ZONE 2: The Rising Wall (Polynomial Transition)
    # ---------------------------------------------------------
    mask_wall = (x_abs > x_hover) & (x_abs <= (x_hover + w_wall))
    
    # Calculate normalized progress along the wall (0.0 at bottom, 1.0 at top)
    wall_progress = (x_abs[mask_wall] - x_hover) / w_wall
    
    # Apply the shape exponent and scale to fit between V_min and 1.0
    V_target[mask_wall] = V_min + (1.0 - V_min) * (wall_progress ** n)
    
    # ---------------------------------------------------------
    # ZONE 3: The Outer Plateau
    # ---------------------------------------------------------
    mask_plateau = x_abs > (x_hover + w_wall)
    V_target[mask_plateau] = 1.0
    
    return V_target