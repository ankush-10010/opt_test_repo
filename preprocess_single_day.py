import pandas as pd

# Define the input and output file names
input_filename = 'preprocessed_orders.csv'
output_filename = 'preprocessed_orders_single_day.csv'

# The single day you want to set for all rows
day_to_set = 254

try:
    # Read the CSV file into a pandas DataFrame
    print(f"Reading '{input_filename}'...")
    df = pd.read_csv(input_filename)
    
    # Check if the 'day_of_year' column exists
    if 'day_of_year' in df.columns:
        # Replace all values in the 'day_of_year' column with the specified day
        print(f"Replacing all 'day_of_year' values with {day_to_set}...")
        df['day_of_year'] = day_to_set
        
        # Save the modified DataFrame to a new CSV file
        df.to_csv(output_filename, index=False)
        
        print(f"Successfully processed the file and saved it as '{output_filename}'.")
        print("\nFirst 5 rows of the new file:")
        print(df.head())
    else:
        print(f"Error: Column 'day_of_year' not found in '{input_filename}'.")

except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")