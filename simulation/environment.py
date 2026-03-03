import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline

# ---------------------------------------------------------
# DATA LOADING & CLEANING
# ---------------------------------------------------------

def mirror_r_rx(df):
    """Mirrors the radial data to cover the negative axis and smooths it."""
    pos_r = df['r (mm)'].to_numpy()
    neg_r = -pos_r[1:][::-1]
    r = np.concatenate((neg_r, pos_r))
    
    rx_pos = df['RX Force (mN)'].to_numpy()
    rx_pos = rx_pos - rx_pos[0]
    rx_neg = rx_pos[1:][::-1]
    rx = np.concatenate((rx_neg, rx_pos))

    cs = CubicSpline(r, rx, bc_type='natural')
    r_smooth = np.linspace(np.min(r), np.max(r), 500)
    rx_smooth = cs(r_smooth)

    return r_smooth, rx_smooth

def load_environment(path_flat='data/flat.csv', 
                     path_45_low='data/45_low.csv', 
                     path_45_high='data/45_high.csv'):
    """
    Loads all datasets and returns them in a single dictionary so the 
    physics engine doesn't have to keep reloading files.
    """
    df_flat = pd.read_csv(path_flat).query('`r (mm)` >= 0').sort_values('r (mm)')
    df_45_low = pd.read_csv(path_45_low).query('`r (mm)` >= 0').sort_values('r (mm)')
    df_45_high = pd.read_csv(path_45_high).query('`r (mm)` >= 0').sort_values('r (mm)')

    r_flat, rx_flat = mirror_r_rx(df_flat)
    r_45_low, rx_45_low = mirror_r_rx(df_45_low)
    r_45_high, rx_45_high = mirror_r_rx(df_45_high)
    
    return {
        'r_flat': r_flat, 'rx_flat': rx_flat,
        'r_45_low': r_45_low, 'rx_45_low': rx_45_low,
        'r_45_high': r_45_high, 'rx_45_high': rx_45_high
    }

# ---------------------------------------------------------
# Z-INTERPOLATION
# ---------------------------------------------------------

def get_force_at_pos(r_target, z_target, r_low, f_low, r_high, f_high):
    """Calculates force at a specific (r, z) based on the 45-degree datasets."""
    # Prevent negative or zero Z values from breaking the fractional power
    z_target = max(0.001, z_target)

    z1, z2 = 12.5, 37.5

    # Base empirical scaling factor you found
    scale_factor = (z2/z1)**(0.4)
    f_low_synthetic_full = f_high * scale_factor
    mask = (r_high <= 110) & (r_high >= -110)

    # Note: Using float() casts to ensure scipy.integrate doesn't get confused by array[1] shapes
    real_f_low_interpolated = interp1d(r_low, f_low, bounds_error=False, fill_value="extrapolate")(r_high[mask])
    f_low_synthetic_full[mask] = real_f_low_interpolated

    f_at_z1 = float(interp1d(r_high, f_low_synthetic_full, bounds_error=False, fill_value=0)(r_target))
    f_at_z2 = float(interp1d(r_high, f_high, bounds_error=False, fill_value=0)(r_target))
    
    if z1 <= z_target <= z2:
        weight = (z_target - z1) / (z2 - z1)
        return (1 - weight) * f_at_z1 + weight * f_at_z2
    
    elif z_target < z1:
        scale = (z1/z_target)**(0.4)
        return f_at_z1 * scale
    else: # z > z2
        scale = (z2/z_target)**(0.4)
        return f_at_z2 * scale

def get_flat_force_at_z(r_target, z_target, r_flat, rx_flat):
    """
    Since we only have z=12.5 data for the flat coil, we use the same empirical 
    0.4 power scaling law to estimate its strength at other heights.
    """
    # Prevent negative or zero Z values from breaking the fractional power
    z_target = max(0.001, z_target)
    
    z1 = 12.5
    f_base = float(interp1d(r_flat, rx_flat, bounds_error=False, fill_value="extrapolate")(r_target))
    
    # Apply identical scaling law as the 45-degree coil
    scale = (z1 / z_target)**(0.4)
    return f_base * scale

# ---------------------------------------------------------
# TILT INTERPOLATION & MASTER WRAPPER
# ---------------------------------------------------------

def get_force_with_tilt(theta_drone, is_left_coil, f_at_0, f_at_45):
    """Calculates final force morphed by the local angle of the coil."""
    coil_base_angle = 45.0
    
    if is_left_coil:
        phi = coil_base_angle + np.degrees(theta_drone)
    else:
        phi = coil_base_angle - np.degrees(theta_drone)
    
    if 0 <= abs(phi) <= 45:
        weight = abs(phi) / 45.0
        f_final = (1 - weight) * f_at_0 + weight * f_at_45
    else:
        f_final = f_at_45 * np.cos(np.radians(abs(phi) - 45))
        
    return f_final

def get_coil_lift(r_target, z_target, theta_drone, is_left_coil, env_data):
    """
    MASTER FUNCTION FOR PHYSICS ENGINE.
    Takes the drone's position, tilt, and environment dictionary, and returns 
    the exact lift force in millinewtons (mN) for that specific coil.
    """
    # 1. Get the theoretical force if the coil were perfectly flat (0 deg)
    f_at_0 = get_flat_force_at_z(
        r_target, z_target, 
        env_data['r_flat'], env_data['rx_flat']
    )
    
    # 2. Get the theoretical force if the coil were at its base 45 deg
    f_at_45 = get_force_at_pos(
        r_target, z_target, 
        env_data['r_45_low'], env_data['rx_45_low'], 
        env_data['r_45_high'], env_data['rx_45_high']
    )
    
    # 3. Apply the angle morphing based on the drone's current tilt
    final_force_mN = get_force_with_tilt(theta_drone, is_left_coil, f_at_0, f_at_45)
    
    return final_force_mN