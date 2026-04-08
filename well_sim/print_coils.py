import argparse
import json

def main():
    # 1. Setup Argparse
    parser = argparse.ArgumentParser(description="Generate a physical winding blueprint from a coil JSON.")
    parser.add_argument('file', type=str, help='Path to the JSON file (e.g., tuned_coil_123.json)')
    args = parser.parse_args()

    # 2. Load the JSON Data
    try:
        with open(args.file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[!] Error: Could not find file '{args.file}'")
        return

    # Extract just the arrays we care about
    radii = data["optimized_coil"]["radii_cm"]
    turns = data["optimized_coil"]["turns"]
    wire_dia = data["hardware"]["wire_diameter_cm"]

    # 3. Print the Blueprint Header
    print("\n" + "="*50)
    print(" 🛠️  PHYSICAL COIL WINDING BLUEPRINT  🛠️")
    print("="*50)
    print(f"File: {args.file}")
    print(f"Wire Diameter: {wire_dia} cm")
    print("-" * 50)
    print(f"{'Radius (cm)':<15} | {'Turns':<10} | {'Winding Direction'}")
    print("-" * 50)

    total_active_slots = 0
    total_physical_turns = 0

    # 4. Filter and Print Non-Zero Windings
    for r, t in zip(radii, turns):
        if t != 0:
            # Determine the physical direction (bucking vs. main)
            if t > 0:
                direction = "Positive (Main/Outer)"
            else:
                direction = "Negative (Bucking/Inner)"
            
            # Print the formatted row
            print(f"{r:<15.2f} | {abs(t):<10} | {direction}")
            
            total_active_slots += 1
            total_physical_turns += abs(t)

    # 5. Print the Summary Stats
    print("-" * 50)
    print(f"Total Active Slots:     {total_active_slots}")
    print(f"Total Physical Turns:   {total_physical_turns}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()