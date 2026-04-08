import argparse
import json
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.special import ellipk, ellipe

# --- 1. Physics Engine ---
def get_B_components(x, z, R):
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

def build_z_cache(x_array, radii, z_height, turns_limit, wire_dia):
    num_slots = len(radii)
    Bx_cache = np.zeros((num_slots, 2 * turns_limit + 1, len(x_array)))
    Bz_cache = np.zeros((num_slots, 2 * turns_limit + 1, len(x_array)))
    
    for slot_idx, R in enumerate(radii):
        for N in range(-turns_limit, turns_limit + 1):
            if N == 0: continue
            direction = np.sign(N)
            turn_idx = N + turns_limit 
            for i in range(int(abs(N))):
                z_i = z_height + (i * wire_dia)
                Bx, Bz = get_B_components(x_array, z_i, R)
                Bx_cache[slot_idx, turn_idx, :] += direction * Bx
                Bz_cache[slot_idx, turn_idx, :] += direction * Bz
    return Bx_cache, Bz_cache

def compute_instant_well(turns_array, Bx_cache, Bz_cache, theta_deg, turns_limit):
    num_slots = len(turns_array)
    indices = turns_array + turns_limit
    Btot_x = np.sum(Bx_cache[np.arange(num_slots), indices, :], axis=0)
    Btot_z = np.sum(Bz_cache[np.arange(num_slots), indices, :], axis=0)
    
    theta = np.radians(theta_deg)
    V_right = np.abs(Btot_x * np.sin(theta) + Btot_z * np.cos(theta))
    V_left = np.abs(Btot_x * np.sin(-theta) + Btot_z * np.cos(-theta))
    
    max_val = max(np.max(V_right), np.max(V_left))
    if max_val > 0:
        V_right /= max_val
        V_left /= max_val
    return V_left, V_right

# --- 2. Main Tuner App ---
def main():
    parser = argparse.ArgumentParser(description="Interactive Coil Tuner with Clickable Histogram.")
    parser.add_argument('-t', '--turns_limit', type=int, default=5, help='Max turns allowed (Y-axis limit)')
    parser.add_argument('-w', '--width', type=float, default=8.2, help='Drone width in cm')
    parser.add_argument('-r', '--rmax', type=float, default=30.0, help='Max radius if starting from scratch')
    parser.add_argument('--rmin', type=float, default=1.0, help='Min radius if starting from scratch')
    parser.add_argument('-j', '--json', type=str, default=None, help='Load initial windings from JSON')
    args = parser.parse_args()

    wire_dia = 0.4
    initial_z = 3.0
    initial_theta_deg = 45.0
    initial_drone_x = 0.0
    original_json_data = None # Keep a copy to preserve metadata during export

    if args.json:
        print(f"Loading initial state from {args.json}...")
        with open(args.json, 'r') as f:
            original_json_data = json.load(f)
        radii = np.array(original_json_data["optimized_coil"]["radii_cm"])
        turns = np.array(original_json_data["optimized_coil"]["turns"])
        wire_dia = original_json_data["hardware"]["wire_diameter_cm"]
        initial_z = original_json_data["hardware"]["z_height_cm"]
        initial_theta_deg = original_json_data["hardware"]["angle_deg"]
    else:
        print(f"No JSON provided. Starting empty canvas from R={args.rmin} to R={args.rmax}...")
        num_slots = int(np.floor((args.rmax - args.rmin) / wire_dia))
        radii = np.linspace(args.rmin, args.rmin + (num_slots - 1) * wire_dia, num_slots)
        
        # Initialize with exactly 1 positive turn at ~10cm
        turns = np.zeros(num_slots, dtype=int)
        start_idx = np.argmin(np.abs(radii - 10.0))
        if start_idx < len(turns):
            turns[start_idx] = 1

    x_array = np.linspace(-35, 35, 300)

    print("Building initial physics cache. Takes ~1 second...")
    Bx_cache, Bz_cache = build_z_cache(x_array, radii, initial_z, args.turns_limit, wire_dia)
    
    # --- 3. Figure Setup ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(bottom=0.25, right=0.85, left=0.08, top=0.92, hspace=0.3)

    V_left, V_right = compute_instant_well(turns, Bx_cache, Bz_cache, initial_theta_deg, args.turns_limit)
    line_left, = ax1.plot(x_array, V_left, label='Left Field', color='red', linewidth=2.5)
    line_right, = ax1.plot(x_array, V_right, label='Right Field', color='blue', linewidth=2.5)
    
    y_height = 1.05
    drone_line, = ax1.plot([initial_drone_x - args.width/2, initial_drone_x + args.width/2], 
                           [y_height, y_height], 'k:', linewidth=3, marker='|', markersize=15)
    drop_left = ax1.axvline(initial_drone_x - args.width/2, color='red', linestyle=':', alpha=0.5)
    drop_right = ax1.axvline(initial_drone_x + args.width/2, color='blue', linestyle=':', alpha=0.5)

    ax1.set_title(f"Interactive Tuner | Z = {initial_z:.1f} cm | Angle = {initial_theta_deg:.0f}°")
    ax1.set_ylabel("Normalized Flux")
    ax1.set_xlim(-args.rmax - 5, args.rmax + 5)
    ax1.set_ylim(0, 1.15)
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.4)

    wire_width = radii[1] - radii[0] if len(radii) > 1 else wire_dia
    colors = ['mediumseagreen' if t >= 0 else 'coral' for t in turns]
    bars = ax2.bar(radii, turns, width=wire_width*0.8, color=colors, edgecolor='black')
    ax2.axhline(0, color='black', linewidth=1.5)

    for i, r in enumerate(radii):
        bg_color = '#f0f0f0' if i % 2 == 0 else '#ffffff'
        ax2.axvspan(r - wire_width/2, r + wire_width/2, color=bg_color, alpha=0.5, zorder=-1)

    ax2.set_title(f"Click top half to add turns. Click bottom half to remove turns.")
    ax2.set_xlabel("Radius (cm)")
    ax2.set_ylabel("Turns")
    ax2.set_xlim(radii[0] - 2, radii[-1] + 2)
    ax2.set_ylim(-args.turns_limit - 0.5, args.turns_limit + 0.5)
    ax2.set_yticks(range(-args.turns_limit, args.turns_limit + 1))
    ax2.grid(axis='y', alpha=0.4)

    # --- 4. UI Elements ---
    # Shrunk the width of the horizontal sliders from 0.65 to 0.55 to leave room for text
    ax_x = plt.axes([0.15, 0.14, 0.55, 0.03])
    slider_x = Slider(ax_x, 'Drone X: ', -25.0, 25.0, valinit=initial_drone_x, valstep=0.2)
    
    ax_angle = plt.axes([0.15, 0.06, 0.55, 0.03])
    slider_angle = Slider(ax_angle, 'Angle (°): ', 0.0, 90.0, valinit=initial_theta_deg, valstep=1.0)

    # Pushed the Z slider slightly further to the right edge (from 0.90 to 0.92)
    ax_z = plt.axes([0.92, 0.35, 0.02, 0.5])
    slider_z = Slider(ax_z, 'Z Height ', 1.0, 15.0, valinit=initial_z, valstep=0.2, orientation='vertical')

    # Tucked the export button neatly into the new empty space on the bottom right
    ax_export = plt.axes([0.78, 0.06, 0.12, 0.06])
    btn_export = Button(ax_export, 'Export JSON', hovercolor='0.975')

    # --- 5. Event Handling ---
    state = {'z': initial_z, 'Bx': Bx_cache, 'Bz': Bz_cache}

    def update_plot():
        V_l, V_r = compute_instant_well(turns, state['Bx'], state['Bz'], slider_angle.val, args.turns_limit)
        line_left.set_ydata(V_l)
        line_right.set_ydata(V_r)
        ax1.set_title(f"Interactive Tuner | Z = {slider_z.val:.1f} cm | Angle = {slider_angle.val:.0f}°")
        fig.canvas.draw_idle()

    def on_slider_change(val):
        cx = slider_x.val
        drone_line.set_xdata([cx - args.width/2, cx + args.width/2])
        drop_left.set_xdata([cx - args.width/2, cx - args.width/2])
        drop_right.set_xdata([cx + args.width/2, cx + args.width/2])
        
        if slider_z.val != state['z']:
            state['z'] = slider_z.val
            state['Bx'], state['Bz'] = build_z_cache(x_array, radii, state['z'], args.turns_limit, wire_dia)
            
        update_plot()

    slider_x.on_changed(on_slider_change)
    slider_angle.on_changed(on_slider_change)
    slider_z.on_changed(on_slider_change)

    def on_click(event):
        if event.inaxes != ax2: return 
        if event.ydata is None or event.xdata is None: return
        if abs(event.ydata) < 0.2: return 
        
        slot_idx = np.argmin(np.abs(radii - event.xdata))
        
        if event.ydata > 0 and turns[slot_idx] < args.turns_limit:
            turns[slot_idx] += 1
        elif event.ydata < 0 and turns[slot_idx] > -args.turns_limit:
            turns[slot_idx] -= 1
            
        bars[slot_idx].set_height(turns[slot_idx])
        bars[slot_idx].set_color('mediumseagreen' if turns[slot_idx] >= 0 else 'coral')
        bars[slot_idx].set_edgecolor('black')
        update_plot()

    fig.canvas.mpl_connect('button_press_event', on_click)

    def export_data(event):
        os.makedirs("tuned", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tuned/tuned_coil_{timestamp}.json"
        
        if original_json_data:
            payload = original_json_data.copy()
            payload["metadata"]["timestamp"] = timestamp
            payload["metadata"]["optimizer"] = "manual_tuner"
            payload["optimized_coil"]["radii_cm"] = np.round(radii, 3).tolist()
            payload["optimized_coil"]["turns"] = turns.tolist()
        else:
            # Boilerplate to keep the visualizer happy
            payload = {
                "metadata": {
                    "timestamp": timestamp,
                    "optimizer": "manual_tuner_scratch"
                },
                "hardware": {
                    "z_height_cm": slider_z.val,
                    "angle_deg": slider_angle.val,
                    "r_min_cm": args.rmin,
                    "r_max_cm": args.rmax,
                    "wire_diameter_cm": wire_dia,
                    "max_turns": args.turns_limit,
                    "eval_limit_cm": 20.0,
                    "bucking_buffer_cm": 1.0
                },
                "target_well": {
                    "V_min": 0.4, "x_hover": 9.0, "w_wall": 2.0, "n_shape": 4.5
                },
                "electrical_properties": {"estimated_inductance_uH": "N/A"},
                "optimized_coil": {
                    "radii_cm": np.round(radii, 3).tolist(),
                    "turns": turns.tolist()
                }
            }
            
        with open(filename, 'w') as f:
            json.dump(payload, f, indent=4)
        print(f"\n[+] Successfully exported current coil design to: {filename}")

    btn_export.on_clicked(export_data)

    plt.show()

if __name__ == "__main__":
    main()