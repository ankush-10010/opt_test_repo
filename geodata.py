import googlemaps
import pandas as pd
import time

# --- CONFIGURATION ---
API_KEY = "AIzaSyC_hI6BowrJPojeBiRldmuFVf3aqsSRZbg"  # 👈 Paste your Google Maps API key here
INPUT_FILE = 'locations_to_geocode.csv'
OUTPUT_FILE = 'geocoded_locations.csv'
# ---------------------

# Initialize the Google Maps client
try:
    gmaps = googlemaps.Client(key=API_KEY)
except Exception as e:
    print(f"Error initializing Google Maps client: {e}")
    print("Please ensure your API key is correct and has the Geocoding API enabled.")
    exit()

# Read the addresses from your CSV file
try:
    df = pd.read_csv(INPUT_FILE)
    if 'location_address' not in df.columns:
        print(f"Error: The input file '{INPUT_FILE}' must have a column named 'location_address'.")
        exit()
    addresses = df['location_address'].tolist()
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILE}' was not found. Make sure it's in the same folder.")
    exit()

geocoded_data = []
print(f"Starting to geocode {len(addresses)} unique locations...")

# Loop through each address and geocode it
for i, address in enumerate(addresses):
    try:
        # Make the API call
        geocode_result = gmaps.geocode(address)

        if geocode_result:
            # Extract the relevant data
            lat = geocode_result[0]['geometry']['location']['lat']
            lng = geocode_result[0]['geometry']['location']['lng']
            formatted_address = geocode_result[0]['formatted_address']
            
            geocoded_data.append({
                'original_address': address,
                'latitude': lat,
                'longitude': lng,
                'formatted_address': formatted_address
            })
            print(f"({i+1}/{len(addresses)}) Successfully geocoded: {address}")
        else:
            # Handle cases where the address was not found
            geocoded_data.append({
                'original_address': address,
                'latitude': None,
                'longitude': None,
                'formatted_address': 'Not Found'
            })
            print(f"({i+1}/{len(addresses)}) Could not geocode: {address}")

        # A small delay to respect API usage limits and avoid errors
        time.sleep(0.1)

    except Exception as e:
        print(f"An error occurred with address '{address}': {e}")
        geocoded_data.append({
            'original_address': address,
            'latitude': None,
            'longitude': None,
            'formatted_address': f'Error: {e}'
        })

# Convert the results to a DataFrame and save to a new CSV file
df_results = pd.DataFrame(geocoded_data)
df_results.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Geocoding complete! Results saved to '{OUTPUT_FILE}'.")