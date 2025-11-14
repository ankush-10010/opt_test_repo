# 🚚 My Awesome Delivery Route Planner!

Hello! This project is like a smart helper that figures out the best and fastest way for a delivery truck to visit many different places.
It looks at a big list of orders, finds all the addresses on a map, and then does some super-smart thinking to create the most efficient route.

## 🗺️ The Workflow: From a Big List to a Smart Map

Here is the journey our data takes, one step at a time.

### Step 1: Find All the Addresses

First, we need to read our giant list of orders and pull out only the addresses we need to visit.
**Program to Run:** 🐍 `dataset_prep.py`
**What it Needs (Input):** 📄 `order_history_kaggle_data.csv` (This is your original, big list of all past orders.)
**What it Makes (Output):** 📄 `locations_to_geocode.csv` (This is a new, clean list of just the addresses we need to find.)

### Step 2: Put the Addresses on a Map

An address like "123 Main Street" is for humans. A computer needs map coordinates (latitude and longitude) to understand where that is. This step turns the addresses into map coordinates.
**Program to Run:** 🐍 `geodata.py`
**What it Needs (Input):** 📄 `locations_to_geocode.csv` (The address list from Step 1.)
**What it Makes (Output):** 📄 `geocoded_locations.csv` (The same list of locations, but now with their exact map coordinates!)

### Step 3: Tidy Up the Final Order List

Now that we have our locations, we clean up the order list one last time to make it perfect for our final program.
**Program to Run:** 🐍 `preprocess_order_history.py`
**What it Needs (Input):** 📄 `geocoded_locations.csv` (The list of locations with coordinates from Step 2.)
**What it Makes (Output):** 📄 `preprocessed_orders.csv` (The final, super-clean list of orders we need to plan routes for.)
**Important Stop!** For the next steps, we only need two files you just made:
* `geocoded_locations.csv` (from Step 2)
* `preprocessed_orders.csv` (from Step 3)

### Step 4: Build the "Travel Rulebook"

This is a very important step. Before we can find the fastest route, we need to know how long it takes to travel between every single location. This program builds a giant "mileage chart" or "rulebook" that lists the distance and time between all the points.
**Program to Run:** 🐍 `build_matrix_with_distance.py`
**What it Needs (Input):** 📄 `geocoded_locations.csv` (All our map locations.)
**What it Makes (Outputs):**
* 📄 `matrix_data_with_distance.json` (The giant "rulebook" with all the times and distances.)
* 📄 `distance_cache.json` (A memory file, so it doesn't have to re-calculate everything if we run it again.)
**Heads up!** This program has to do a lot of calculations. It might take a long time to finish. This is normal! Go get a snack. 🍪

### Step 5: Find the Smartest Route! (The Magic Step)

This is the final step! This program is the "brain." It takes our clean order list (from Step 3) and our new "travel rulebook" (from Step 4) and does all the hard thinking to find the single best, most efficient route.
**Program to Run:** 🐍 `run_hybrid_solver_layers.py`
**What it Needs (Inputs):** It uses the files we just made, like `preprocessed_orders.csv` and `matrix_data_with_distance.json`. It also uses two "helper" files to do its thinking: `optimization_solver_layers.py` and `hybrid_solver_layers.py`.
**What it Makes (Output):** 🗺️ An HTML file (like `output.html`)! You can open this file in your web browser (like Chrome or Firefox) to see the final, super-smart route all planned out.

## ⭐ How to Run Everything (The Short Version)

Just run these Python programs in order:
1. Run: `dataset_prep.py`
   - Waits for: `order_history_kaggle_data.csv`
   - Makes: `locations_to_geocode.csv`
2. Run: `geodata.py`
   - Waits for: `locations_to_geocode.csv`
   - Makes: `geocoded_locations.csv`
3. Run: `preprocess_order_history.py`
   - Waits for: `geocoded_locations.csv`
   - Makes: `preprocessed_orders.csv`
4. Run: `build_matrix_with_distance.py`
   - Waits for: `geocoded_locations.csv`
   - Makes: `matrix_data_with_distance.json` and `distance_cache.json`
   (Remember: This one can take a long time!)
5. Run: `run_hybrid_solver_layers.py`
   - Waits for: The files from steps 3 and 4.
   - Makes: Your final HTML output file.

That's it! After step 5, just open the HTML file to see your finished route. 🎉
