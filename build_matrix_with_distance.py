import pandas as pd
import json
import time
from optimization_solver_layers import get_real_travel_time # Assumes this file exists

# --- CONFIGURATION FOR NEW CUSTOM DEPOT ---
#
# EDIT THIS DICTIONARY to define your new depot.
# This depot will be at INDEX 0 in the new time matrix.
# All locations from 'geocoded_locations.csv' will be treated as customers.
#
NEW_DEPOT_INFO = {
  "original_address": "My New Central Kitchen",
  "latitude": 28.5707,
  "longitude": 77.3262,
  "formatted_address": "My New Central Kitchen, Dadar, Mumbai, Maharashtra, India"
}

# The file containing all your CUSTOMER locations
CUSTOMER_LOCATIONS_FILE = 'geocoded_locations.csv'

# The new output file that will be created
OUTPUT_FILE = 'matrix_data_with_distance.json'
# --------------------------------------------------


def build_and_save_matrix_custom():
    """
    Builds a time matrix using a manually defined depot and treats all
    locations from the CSV as customers.
    """
    print("--- Starting Custom Time Matrix Builder ---")
    
    # --- Step 1: Define the Depot ---
    # We use the manually-defined dictionary from the configuration
    depot_info = NEW_DEPOT_INFO
    print(f"Step 1: Using custom depot '{depot_info['original_address']}'")
    print(f"       at (Lat: {depot_info['latitude']}, Lon: {depot_info['longitude']})")

    # --- Step 2: Load Customer Locations ---
    try:
        print(f"Step 2: Loading all locations from '{CUSTOMER_LOCATIONS_FILE}' to be customers...")
        df_customers = pd.read_csv(CUSTOMER_LOCATIONS_FILE)
        print(f"...Loaded {len(df_customers)} customer locations.")
        
    except FileNotFoundError:
        print(f"\nFATAL ERROR: The file '{CUSTOMER_LOCATIONS_FILE}' was not found.")
        print("Please make sure it is in the same folder as the script.")
        return # Exit the function
    except Exception as e:
        print(f"\nFATAL ERROR: Could not read '{CUSTOMER_LOCATIONS_FILE}'. Error: {e}")
        return

    # --- Step 3: Combine Lists and Build Matrix ---
    try:
        # Sort customers by address for consistent ordering
        customer_locations = df_customers.sort_values(by='original_address').reset_index(drop=True)
        
        # Create the final master list:
        # Our custom depot is at index 0
        # All locations from the CSV follow (at indices 1, 2, 3...)
        all_locations = [depot_info] + customer_locations.to_dict('records')
        
        num_locations = len(all_locations)
        
        print(f"Step 3: Found {num_locations} total locations (1 custom depot + {num_locations - 1} customers).")
        print("Building matrix... This may take a long time on the first run.")

        # Set departure time 24h in the future for predictive traffic
        departure_timestamp = int(time.time()) + (3600 * 24)
        
        # Initialize an N x N matrix
        time_matrix = [[0] * num_locations for _ in range(num_locations)]
        distance_matrix = [[0.0] * num_locations for _ in range(num_locations)] # Use floats for distance

        # Iterate over every possible pair of locations
        for i in range(num_locations):
            for j in range(num_locations):
                if i == j: continue # Skip travel from a place to itself
                
                loc1 = all_locations[i] # Origin
                loc2 = all_locations[j] # Destination
                
                duration_min, distance_km = get_real_travel_time( # Use the new function name
                    loc1['latitude'], loc1['longitude'],
                    loc2['latitude'], loc2['longitude'],
                    departure_timestamp
                )

                # --- 5. ASSIGN TO BOTH MATRICES ---
                time_matrix[i][j] = duration_min
                distance_matrix[i][j] = distance_km
                # Optional: Add a check for error values if needed
                if duration_min == 99999 or distance_km == float('inf'):
                    print(f"  WARNING: Received error value for pair ({i},{j}). Check API logs.")
            # Log progress
            print(f"Computed routes for location {i+1}/{num_locations}...")

        # --- Step 4: Save the Output ---
        output_data = {
            "locations": all_locations,
            "time_matrix": time_matrix,
            # --- 6. ADD DISTANCE MATRIX TO OUTPUT ---
            "distance_matrix": distance_matrix
        }
        
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(output_data, f, indent=2) # indent=2 for readability
            
        print(f"\n✅ Master time matrix with custom depot successfully built and saved to '{OUTPUT_FILE}'")

    except Exception as e:
        print(f"\nAn unexpected error occurred during matrix building: {e}")
        print("This could be an issue with the 'get_real_travel_time' function or API limits.")


if __name__ == "__main__":
    build_and_save_matrix_custom()
