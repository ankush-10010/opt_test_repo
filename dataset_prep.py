import pandas as pd

# This is the code I will run once your file is accessible.
file_path = 'order_history_kaggle_data.csv'

try:
    df = pd.read_csv(file_path)

    # --- Data Cleaning and Preparation ---
    print("File read successfully. Starting data preparation...")
    location_cols = ['Restaurant name', 'Subzone', 'City']
    df_locations = df[location_cols].copy()
    df_locations['City'] = 'Delhi NCR'
    df_locations.dropna(subset=['Restaurant name', 'Subzone'], inplace=True)

    # --- Identify Unique Locations for Geocoding ---
    df_locations['restaurant_address'] = df_locations['Restaurant name'].str.strip() + ', ' + df_locations['Subzone'].str.strip() + ', ' + df_locations['City'].str.strip()
    unique_restaurants = df_locations['restaurant_address'].unique()

    df_locations['subzone_address'] = df_locations['Subzone'].str.strip() + ', ' + df_locations['City'].str.strip()
    unique_subzones = df_locations['subzone_address'].unique()

    locations_to_geocode = sorted(list(set(unique_restaurants.tolist() + unique_subzones.tolist())))

    print(f"Identified {len(locations_to_geocode)} total unique locations to geocode.")

    # --- Save the list of locations to a new CSV ---
    df_to_geocode = pd.DataFrame(locations_to_geocode, columns=['location_address'])
    output_csv_path = 'locations_to_geocode.csv'
    df_to_geocode.to_csv(output_csv_path, index=False)
    
    print(f"Successfully saved all unique locations to '{output_csv_path}'")
    print("You can now use this file for the geocoding phase.")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' is still not found.")
except Exception as e:
    print(f"An error occurred during processing: {e}")