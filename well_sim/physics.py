import numpy as np
from scipy.special import ellipk, ellipe

def get_B_components(x, z, R):
    """
    Calculates the normalized Bx and Bz magnetic field components 
    for a single circular current loop of radius R.
    """
    # Safe handling for x=0 to prevent division by zero errors in the math
    x_safe = np.where(x == 0, 1e-10, x)
    
    # Elliptic integral parameters
    k_squared = (4 * R * x_safe) / ((R + x_safe)**2 + z**2)
    K = ellipk(k_squared)
    E = ellipe(k_squared)
    
    root_term = np.sqrt((R + x_safe)**2 + z**2)
    alpha_squared = (R - x_safe)**2 + z**2
    
    # Evaluate the exact field components
    Bx = (z / (x_safe * root_term)) * (-K + ((R**2 + x_safe**2 + z**2) / alpha_squared) * E)
    Bz = (1 / root_term) * (K + ((R**2 - x_safe**2 - z**2) / alpha_squared) * E)
    
    # Explicitly force Bx to 0 where x is exactly 0 (axis of symmetry)
    if isinstance(x, np.ndarray):
        Bx[x == 0] = 0.0
    elif x == 0:
        Bx = 0.0
        
    return Bx, Bz

def calculate_simulated_well(x_array, radii, turns_array, z_height, angle_deg, wire_diameter):
    """
    Calculates the conjoined V_sim profile for the left and right drone motors,
    accounting for accurate downward 3D wire stacking and bucking coils.
    """
    Btot_x = np.zeros_like(x_array, dtype=float)
    Btot_z = np.zeros_like(x_array, dtype=float)
    
    # Iterate over each radial slot and its assigned number of turns
    for R, N in zip(radii, turns_array):
        if N == 0:
            continue
            
        # Determine current direction (positive or negative bucking coil)
        direction = np.sign(N)
        num_turns = int(abs(N))
        
        # Calculate the stack: wires are pushed DOWNWARD into the base pad
        for i in range(num_turns):
            z_i = z_height + (i * wire_diameter)
            Bx, Bz = get_B_components(x_array, z_i, R)
            
            Btot_x += direction * Bx
            Btot_z += direction * Bz
            
    # Calculate the normalized flux passing through the tilted drone coils
    theta = np.radians(angle_deg)
    
    V_right = np.abs(Btot_x * np.sin(theta) + Btot_z * np.cos(theta))
    V_left = np.abs(Btot_x * np.sin(-theta) + Btot_z * np.cos(-theta))
    
    # Stitch them together at the origin (x=0) to form the conjoined well
    V_sim = np.where(x_array >= 0, V_right, V_left)
    
    # Returning all three allows the GUI to plot the individual red/blue curves
    return V_sim, V_left, V_right

def estimate_inductance(radii_cm, turns_array, wire_diameter_cm):
    """
    Estimates the total equivalent inductance of the multi-zoned, 3D stacked coil.
    Accounts for both the self-inductance of every loop and the mutual inductance 
    between every possible pair of loops (including the negative bucking effect).
    Returns the value in microhenries (µH).
    """
    # 1. Convert everything to standard SI units (Meters)
    radii_m = np.array(radii_cm) * 0.01
    wire_dia_m = wire_diameter_cm * 0.01
    wire_radius_m = wire_dia_m / 2.0
    mu_0 = 4 * np.pi * 1e-7
    
    # 2. Flatten the coil cross-section into a 1D list of individual physical loops
    # Each loop gets a tuple: (Radius, Z-depth, Current Direction)
    loops = []
    for R_m, N in zip(radii_m, turns_array):
        if N == 0:
            continue
            
        direction = np.sign(N)
        num_turns = int(abs(N))
        
        for i in range(num_turns):
            z_m = i * wire_dia_m # Downward stack depth
            loops.append((R_m, z_m, direction))
            
    num_loops = len(loops)
    if num_loops == 0:
        return 0.0
        
    total_inductance_H = 0.0
    
    # 3. Calculate Self-Inductance for every individual loop
    for R_m, _, _ in loops:
        # Standard approximation for a thin circular wire loop
        L_self = mu_0 * R_m * (np.log(8 * R_m / wire_radius_m) - 2)
        total_inductance_H += L_self
        
    # 4. Calculate Mutual Inductance between EVERY pair of loops
    for i in range(num_loops):
        for j in range(num_loops):
            if i == j:
                continue # Skip self (already calculated above)
                
            R1, z1, dir1 = loops[i]
            R2, z2, dir2 = loops[j]
            
            # Physical vertical distance between the two loops
            d = abs(z1 - z2)
            
            # Elliptic integral geometric factor
            k_squared = (4 * R1 * R2) / ((R1 + R2)**2 + d**2)
            
            # Safety catch: if loops perfectly overlap (they shouldn't), skip to avoid divide-by-zero
            if k_squared >= 1.0:
                continue 
                
            k = np.sqrt(k_squared)
            K = ellipk(k_squared)
            E = ellipe(k_squared)
            
            # Maxwell's formula for mutual inductance of coaxial circular filaments
            M = mu_0 * np.sqrt(R1 * R2) * ((2/k - k) * K - (2/k) * E)
            
            # If one loop is positive (outer) and one is negative (bucking), 
            # their mutual inductance physically subtracts from the total!
            total_inductance_H += dir1 * dir2 * M
            
    # Convert Henries to Microhenries for easier reading
    return total_inductance_H * 1e6