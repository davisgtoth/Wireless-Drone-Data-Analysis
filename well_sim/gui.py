import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from target import generate_target_well

def launch_tuning_gui():
    """
    Launches an interactive Matplotlib GUI to tune the 4 target well parameters.
    Blocks execution until the user clicks 'Confirm & Run'.
    Returns the selected parameters as a tuple: (V_min, x_hover, w_wall, n)
    """
    # 1. Setup the initial figure and data
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.35)
    
    x_array = np.linspace(-30, 30, 300)
    
    # Initial default parameters
    init_V_min = 0.2
    init_x_hover = 4.0
    init_w_wall = 8.0
    init_n = 2.0
    
    V_target = generate_target_well(x_array, init_V_min, init_x_hover, init_w_wall, init_n)
    
    # Plot the initial curve
    line, = ax.plot(x_array, V_target, color='black', linewidth=2.5, linestyle=':')
    ax.set_title("Interactive Target Well Definition")
    ax.set_xlabel("Distance from Center (cm)")
    ax.set_ylabel("Normalized Voltage")
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.4)

    # 2. Setup the Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_vmin = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_xhover = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
    ax_wwall = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_n = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)

    s_vmin = Slider(ax_vmin, 'V_min', 0.0, 1.0, valinit=init_V_min, valstep=0.05)
    s_xhover = Slider(ax_xhover, 'x_hover', 0.0, 15.0, valinit=init_x_hover, valstep=0.5)
    s_wwall = Slider(ax_wwall, 'w_wall', 1.0, 20.0, valinit=init_w_wall, valstep=0.5)
    s_n = Slider(ax_n, 'n (Shape)', 0.5, 10.0, valinit=init_n, valstep=0.5)

    def update(val):
        new_V = generate_target_well(x_array, s_vmin.val, s_xhover.val, s_wwall.val, s_n.val)
        line.set_ydata(new_V)
        fig.canvas.draw_idle()

    s_vmin.on_changed(update)
    s_xhover.on_changed(update)
    s_wwall.on_changed(update)
    s_n.on_changed(update)

    # 3. Setup the Confirm Button
    confirm_ax = plt.axes([0.8, 0.025, 0.15, 0.04])
    button = Button(confirm_ax, 'Confirm & Run', color='lightblue', hovercolor='0.975')
    
    # We use a list to store the return values so it can be modified inside the callback
    selected_params = []
    
    def on_confirm(event):
        selected_params.extend([s_vmin.val, s_xhover.val, s_wwall.val, s_n.val])
        plt.close(fig) # Closes the window to resume main script execution

    button.on_clicked(on_confirm)
    
    # This blocks the Python script until the window is closed
    plt.show()
    
    return tuple(selected_params)

def plot_final_results(results_dict):
    """
    Plots the final optimized target vs. simulated well, and a bar chart of the coil.
    """
    x_array = results_dict["x_array"]
    radii = results_dict["radii"]
    best_turns = results_dict["best_turns"]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(hspace=0.3)
    
    # --- Top Plot: The Magnetic Wells ---
    ax1.plot(x_array, results_dict["V_target"], 'k:', linewidth=3, label="Target Well")
    ax1.plot(x_array, results_dict["V_left"], 'r', linewidth=2, label="Left Motor V_sim")
    ax1.plot(x_array, results_dict["V_right"], 'b', linewidth=2, label="Right Motor V_sim")
    
    ax1.set_title(f"Optimized Magnetic Well (Final Cost: {results_dict['final_cost']:.4f})")
    ax1.set_ylabel("Normalized Voltage")
    x_min, x_max = x_array[0], x_array[-1]
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.4)
    
    # Highlight the evaluation window
    eval_win = results_dict["eval_window"]
    ax1.axvspan(eval_win[0], eval_win[1], color='gray', alpha=0.1, label='Evaluation Window')
    
    # --- Bottom Plot: The Physical Winding Profile ---
    # Colors: Green for positive turns, Orange for negative (bucking) turns
    colors = ['mediumseagreen' if t >= 0 else 'coral' for t in best_turns]
    
    # The width of the bars visually represents the wire diameter
    wire_width = radii[1] - radii[0] if len(radii) > 1 else 0.3
    
    ax2.bar(radii, best_turns, width=wire_width*0.8, color=colors, edgecolor='black')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title("Physical Winding Profile (Cross-Section)")
    ax2.set_xlabel("Radius (cm)")
    ax2.set_ylabel("Number of Turns")
    ax2.set_xlim(radii[0] - 1, radii[-1] + 1)
    ax2.grid(axis='y', alpha=0.4)
    
    plt.show()