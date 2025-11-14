import pandas as pd
import time
from datetime import datetime
import re
import json
# --- Configuration ---
# This file must contain ALL unique locations your orders go to.
# It's the "master list" of locations.
LOCATION_FILE = 'geocoded_locations.csv' # Or 'time_matrix.json'
ORDER_HISTORY_FILE = 'order_history_kaggle_data.csv' # Your big data file
OUTPUT_FILE = 'preprocessed_orders.csv'

# Regex to count items. Finds "1 x", "2 x", etc.
# Adjust if your format is different (e.g., "x2", "Qty: 1")
ITEM_COUNT_REGEX = re.compile(r'(\d+)\s*x')
# ---------------------

def build_location_to_index_map():
    """
    Loads the geocoded locations and creates a mapping of
    'original_address' -> index.
    
    The Depot (if it's in this file) will be at some index,
    but our simulation script (which uses time_matrix.json)
    will know the Depot is at index 0. This is fine.
    
    Let's load from geocoded_locations.csv for simplicity.
    """
    print(f"Loading locations from {LOCATION_FILE} to build map...")
    try:
        # Check if we're loading from CSV or JSON
        if LOCATION_FILE.endswith('.json'):
            # This handles loading from a 'build_master_matrix' output
            with open(LOCATION_FILE, 'r') as f:
                data = json.load(f)
            locations_df = pd.DataFrame(data['locations'])
        else:
            locations_df = pd.read_csv(LOCATION_FILE)
            
        # Create the map
        location_map = {}
        for index, row in locations_df.iterrows():
            location_map[row['original_address']] = index
            
        print(f"Built map with {len(location_map)} locations.")
        return location_map
        
    except FileNotFoundError:
        print(f"FATAL ERROR: Location file not found: {LOCATION_FILE}")
        return None
    except Exception as e:
        print(f"FATAL ERROR reading location file: {e}")
        return None

def parse_order_time(time_str):
    """
    Parses your specific time format: "11:41 PM, September 10 2024"
    and converts it to a standard simulation timestamp (minutes from day start).
    
    Returns (day_of_year, minutes_from_midnight)
    """
    try:
        # Parse the complex time string
        dt_obj = datetime.strptime(str(time_str).strip(), "%I:%M %p, %B %d %Y")
        
        # We need two things:
        # 1. A way to group by day
        day_of_year = dt_obj.timetuple().tm_yday
        # 2. Minutes from start of that day
        minutes_from_midnight = dt_obj.hour * 60 + dt_obj.minute
        
        return dt_obj.timestamp(), day_of_year, minutes_from_midnight
        
    except ValueError:
        print(f"Warning: Could not parse time: {time_str}")
        return None, None, None

def parse_demand(item_str):
    """
    Parses the 'Item' string to calculate total demand (total number of items).
    e.g., "1 x Bone in..., 1 x Bone in..." -> 2
    e.g., "1 x Grilled Chicken..." -> 1
    """
    if not isinstance(item_str, str):
        return 1 # Default to 1 if item is blank or unreadable
        
    matches = ITEM_COUNT_REGEX.findall(item_str)
    if not matches:
        # This handles cases like "Animal Fries" (no "1 x" prefix)
        return 1 # Default to 1 item if no "1 x" prefix is found
    
    try:
        total_demand = sum(int(count) for count in matches)
        return total_demand
    except:
        return 1 # Default to 1 on any error

def run_preprocessing():
    """
    Main function to read, process, and save the historical order data.
    """
    location_map = build_location_to_index_map()
    if location_map is None:
        return

    print(f"Starting to process large order file: {ORDER_HISTORY_FILE}...")
    
    try:
        # Load the entire history file.
        df = pd.read_csv(ORDER_HISTORY_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"FATAL ERROR: Order history file not found: {ORDER_HISTORY_FILE}")
        return
    except Exception as e:
        print(f"FATAL ERROR reading order history: {e}")
        return

    print(f"Loaded {len(df)} historical orders. Now processing...")
    
    processed_orders = []
    skipped_count = 0
    
    # Define the exact column names from your CSV
    RESTAURANT_COL = 'Restaurant name'
    SUBZONE_COL = 'Subzone'
    CITY_COL = 'City'
    TIME_COL = 'Order Placed At'
    ITEM_COL = 'Items in order'
    ORDER_ID_COL = 'Order ID'

    # Iterate over every row
    for index, row in df.iterrows():
        try:
            # 1. Create the full location string to match our map
            # e.g., "Swaad, Greater Kailash 2 (GK2), Delhi NCR"
            location_name = f"{row[RESTAURANT_COL]}, {row[SUBZONE_COL]}, {row[CITY_COL]}"

            # 2. Find the location_index for this order
            location_index = location_map.get(location_name)
            
            if location_index is None:
                # This order's location is not in our geocoded file
                if (skipped_count < 20): # Print first 20 warnings
                    print(f"Warning: Skipping order. Location not in map: '{location_name}'")
                skipped_count += 1
                continue
                
            # 3. Parse the time
            timestamp, day, minute = parse_order_time(row[TIME_COL])

            if timestamp is None:
                skipped_count += 1
                continue
                
            # 4. Parse the demand
            demand = parse_demand(row[ITEM_COL])
                
            # 5. Add to our clean list
            processed_orders.append({
                'timestamp': int(timestamp),
                'day_of_year': day,
                'minute_of_day': minute,
                'location_index': location_index,
                'original_address': location_name,
                'demand': demand,
                'order_id': row[ORDER_ID_COL]
            })
            
            if (index + 1) % 50000 == 0:
                print(f"   ...processed {index + 1} orders...")
        
        except KeyError as e:
            print(f"FATAL ERROR: A required column is missing: {e}. Check your CSV file headers.")
            return
        except Exception as e:
            print(f"Error on row {index}: {e}")
            skipped_count += 1

    print(f"Processing complete.")
    print(f"Successfully processed {len(processed_orders)} orders.")
    print(f"Skipped {skipped_count} orders (due to missing locations, bad timestamps, or other errors).")
    
    # Convert to DataFrame and save
    if processed_orders:
        output_df = pd.DataFrame(processed_orders)
        # Sort by time, so the simulation can read it chronologically
        output_df = output_df.sort_values(by='timestamp')
        
        output_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Successfully saved clean data to {OUTPUT_FILE}")
    else:
        print("No orders were processed.")

if __name__ == "__main__":
    run_preprocessing()

