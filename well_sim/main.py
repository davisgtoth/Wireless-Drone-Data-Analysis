import argparse
import json
import os
from datetime import datetime
import numpy as np

# Import our custom modules
import gui
from optimizer import run_optimization
from physics import estimate_inductance

def main():
    # 1. Setup Argparse
    parser = argparse.ArgumentParser(description="Optimize a discrete magnetic well for a passive drone.")
    
    # Required parameters
    parser.add_argument('-z', '--z_height', type=float, required=True, 
                        help='Vertical distance from pad to drone receiver (cm).')
    parser.add_argument('-a', '--angle', type=float, required=True, 
                        help='Mounting angle of the drone receiver coils (degrees).')
    
    # Optional hardware parameters
    parser.add_argument('--rmin', type=float, default=3.0, help='Inner winding radius (cm).')
    parser.add_argument('--rmax', type=float, default=30.0, help='Outer winding radius (cm).')
    parser.add_argument('--wire_diameter', type=float, default=0.3, help='Wire diameter (cm).')
    parser.add_argument('--max_turns', type=int, default=5, help='Max vertical stack depth.')
    parser.add_argument('--eval_limit', type=float, default=20.0, help='Max distance for shape evaluation (cm).')
    parser.add_argument('--bucking_buffer', type=float, default=1.0, 
                        help='Distance inside x_hover where bucking coils must stop (cm).')
    
    # Optional bypass for the GUI
    parser.add_argument('--well_params', nargs=4, type=float, metavar=('V_MIN', 'X_HOVER', 'W_WALL', 'N'),
                        help='Provide 4 floats to bypass the GUI: V_min x_hover w_wall n')
    
    args = parser.parse_args()
    
    hardware_params = {
        "z_height_cm": args.z_height,
        "angle_deg": args.angle,
        "r_min_cm": args.rmin,
        "r_max_cm": args.rmax,
        "wire_diameter_cm": args.wire_diameter,
        "max_turns": args.max_turns,
        "eval_limit_cm": args.eval_limit,
        "bucking_buffer_cm": args.bucking_buffer
    }
    
    # 2. Determine Target Parameters (GUI vs CLI Bypass)
    if args.well_params is not None:
        print("Bypassing GUI. Using provided well parameters.")
        target_params = tuple(args.well_params)
    else:
        print("Launching interactive GUI. Please tune your well and click 'Confirm & Run'.")
        target_params = gui.launch_tuning_gui()
        if not target_params:
            print("GUI closed without confirmation. Exiting.")
            return

    # 3. Run the Optimization
    print("\n--- Hardware Constraints ---")
    for k, v in hardware_params.items():
        print(f"{k}: {v}")
    print(f"\nTarget Parameters [V_min, x_hover, w_wall, n]: {target_params}")
    
    results = run_optimization(hardware_params, target_params)

    print("\nCalculating physical coil inductance (this takes a second)...")
    estimated_L_uH = estimate_inductance(
        results["radii"], 
        results["best_turns"], 
        hardware_params["wire_diameter_cm"]
    )
    print(f"Estimated Inductance: {estimated_L_uH:.2f} µH")
    
    # 4. JSON Payload Generation
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/well_design_{timestamp}.json"
    
    payload = {
        "metadata": {
            "timestamp": timestamp,
            "final_cost_score": float(results["final_cost"]),
            "optimizer": "differential_evolution"
        },
        "hardware": hardware_params,
        "target_well": {
            "V_min": target_params[0],
            "x_hover": target_params[1],
            "w_wall": target_params[2],
            "n_shape": target_params[3]
        },
        "electrical_properties": {
            "estimated_inductance_uH": round(estimated_L_uH, 2)
        },
        "optimized_coil": {
            "radii_cm": np.round(results["radii"], 3).tolist(),
            "turns": results["best_turns"].tolist()
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(payload, f, indent=4)
        
    print(f"\nOptimization complete! JSON payload saved to: {filename}")
    
    # 5. Show Final Plot
    gui.plot_final_results(results)

if __name__ == "__main__":
    main()