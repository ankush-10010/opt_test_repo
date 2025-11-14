import copy
import random
from collections import deque
from optimization_solver_layers import solve_vrp_with_capacity # We import our new engine
import math
import time
# --- Helper Function ---

# --- ALNS Parameters (can be tuned) ---
ALNS_ITERATIONS = 5000     # How many iterations to run
ALNS_SEGMENT_LENGTH = 50   # How often to update operator weights
ALNS_REACTION_FACTOR = 0.7 # How much new scores influence weights (0 to 1)
# Acceptance criteria (Simulated Annealing based)
ALNS_TEMP_START = 1000
ALNS_TEMP_END = 1
ALNS_COOLING_RATE = 0.995 # T_new = T_old * cooling_rate
# Destroy control (% of orders to remove)
ALNS_DESTROY_MIN_PERCENT = 0.15
ALNS_DESTROY_MAX_PERCENT = 0.40
# Operator Scores
ALNS_SIGMA1 = 1 # Score for finding new global best  10
ALNS_SIGMA2 = 1  # Score for finding solution better than current 5
ALNS_SIGMA3 = 1  # Score for accepting worse solution (exploration) 2


def calculate_raw_route_distance(stop_indices, distance_matrix):
    """
    Calculates the total travel distance for a list of stop indices.
    Assumes depot is start (0) and end (0).
    e.g., [5, 3] -> dist(0,5) + dist(5,3) + dist(3,0)
    """
    if not stop_indices:
        return 0
    
    total_distance = 0
    last_idx = 0 # Start at depot
    
    for stop_idx in stop_indices:
        # --- Basic Bounds Check ---
        if last_idx >= len(distance_matrix) or stop_idx >= len(distance_matrix[last_idx]):
             print(f"Error: Index out of bounds in distance_matrix access ({last_idx}, {stop_idx})")
             return float('inf') # Return infinity on error
        total_distance += distance_matrix[last_idx][stop_idx]
        last_idx = stop_idx
        
    # Return to depot
    if last_idx >= len(distance_matrix) or 0 >= len(distance_matrix[last_idx]):
        print(f"Error: Index out of bounds returning to depot ({last_idx}, 0)")
        return float('inf')
    total_distance += distance_matrix[last_idx][0]
    return total_distance

def calculate_raw_route_time(stop_indices, time_matrix):
    """
    Calculates the total travel time (in seconds) for a list of stop indices.
    Assumes depot is start (0) and end (0).
    e.g., [5, 3] -> time(0,5) + time(5,3) + time(3,0)
    """
    if not stop_indices:
        return 0
    
    total_time = 0
    last_idx = 0 # Start at depot
    
    for stop_idx in stop_indices:
        # --- Basic Bounds Check ---
        if last_idx >= len(time_matrix) or stop_idx >= len(time_matrix[last_idx]):
             print(f"Error: Index out of bounds in time_matrix access ({last_idx}, {stop_idx})")
             return float('inf') # Return infinity on error
        total_time += time_matrix[last_idx][stop_idx]
        last_idx = stop_idx
        
    # Return to depot
    if last_idx >= len(time_matrix) or 0 >= len(time_matrix[last_idx]):
        print(f"Error: Index out of bounds returning to depot ({last_idx}, 0)")
        return float('inf')
        
    total_time += time_matrix[last_idx][0]
    return total_time
# --- Main Cost Calculation Functions ---
def calculate_total_fleet_cost(routes_dict, distance_matrix, 
                               fixed_cost_per_truck, variable_cost_per_km):
    """
    Calculates the total fleet cost based on trucks used and distance traveled.
    Input: routes_dict = { v_id: [order_obj_1, ...], ... }
           distance_matrix = The matrix loaded earlier
           fixed_cost_per_truck = Cost parameter
           variable_cost_per_km = Cost parameter
    Returns: total_cost (float), num_trucks_used (int), total_distance (float)
    """
    total_distance = 0
    num_trucks_used = 0
    
    for vehicle_id, route_orders in routes_dict.items():
        if route_orders: # If the vehicle is used
            num_trucks_used += 1
            
            # Get unique stop indices for distance calculation
            stop_indices = list(dict.fromkeys([order['index'] for order in route_orders]))
            
            route_distance = calculate_raw_route_distance(stop_indices, distance_matrix)
            if route_distance == float('inf'): # Handle error from helper
                 print(f"Warning: Could not calculate distance for Vehicle {vehicle_id}, cost will be inaccurate.")
            else:
                 total_distance += route_distance
            
    # Calculate final cost
    total_cost = (fixed_cost_per_truck * num_trucks_used) + \
                 (variable_cost_per_km * total_distance)
                 
    return total_cost, num_trucks_used, total_distance
def calculate_route_cost(route_orders, time_matrix):
    """
    Calculates the travel time (in minutes) for a route
    represented by a list of order objects.
    """
    if not route_orders:
        return 0
        
    # Get unique stop indices, preserving insertion order
    stop_indices = list(dict.fromkeys([order['index'] for order in route_orders]))
    
    # Use the helper to get time in seconds
    time_seconds = calculate_raw_route_distance(stop_indices, time_matrix)
    
    return time_seconds / 60.0 # Convert to minutes

def calculate_total_cost(routes_dict, time_matrix):
    """
    Calculates the total fleet cost (in minutes) for a dictionary
    of routes (represented by lists of order objects).
    """
    total_cost = 0
    for route_orders in routes_dict.values():
        total_cost += calculate_route_cost(route_orders, time_matrix)
    return total_cost

# --- LAYER 1: IMMEDIATE ASSIGNMENT (Greedy + Tabu) ---

def _greedy_insert_capacity(new_order, current_routes, time_matrix, 
                           vehicle_capacity, max_route_duration_mins):
    """
    (This was the old assign_new_order_realtime function)
    Finds the best single insertion point (or new vehicle) for a new order.
    """
    best_cost_increase = float('inf')
    best_vehicle = -1
    best_insertion_idx = -1 # We don't use this for simple append, but good to have
    
    new_order_idx = new_order['index']
    new_order_demand = new_order['demand']

    # 1. Try to find the "cheapest" insertion in an EXISTING route
    for v_id, route_orders in current_routes.items():
        
        current_load = sum(o['demand'] for o in route_orders)
        
        # --- Constraint 1: Check Capacity ---
        if current_load + new_order_demand > vehicle_capacity:
            continue # This vehicle is too full

        # Get current unique stops
        stop_indices = list(dict.fromkeys([o['index'] for o in route_orders]))
        original_cost = calculate_raw_route_distance(stop_indices, time_matrix)
        
        # ---
        # Note: A true "best insertion" would try inserting at every *index*.
        # For a fast L1, we often just check "appending" to the route.
        # Let's adapt your original logic which tried every index.
        # ---
        
        # Try inserting the new order at every possible position in the *order list*
        for i in range(len(route_orders) + 1):
            
            # Create a potential new list of *orders*
            temp_route_orders = route_orders[:i] + [new_order] + route_orders[i:]
            
            # Get the *unique stop indices* from this new order list
            new_unique_stops = list(dict.fromkeys([o['index'] for o in temp_route_orders]))
            
            new_cost = calculate_raw_route_distance(new_unique_stops, time_matrix)
            
            # --- Constraint 2: Check Duration ---
            if (new_cost / 60.0) > max_route_duration_mins:
                continue # This new route would be too long
                
            cost_increase = new_cost - original_cost
            
            if cost_increase < best_cost_increase:
                best_cost_increase = cost_increase
                best_vehicle = v_id
                best_insertion_idx = i # Now we store the index

    # 2. If we found a good insertion, return the new state
    if best_vehicle != -1:
        new_routes_state = copy.deepcopy(current_routes)
        new_routes_state[best_vehicle].insert(best_insertion_idx, new_order)
        return new_routes_state, "Best Insertion"

    # 3. If no insertion worked, try to put it in an EMPTY vehicle
    cost_of_new_route = calculate_raw_route_distance([new_order_idx], time_matrix)
    duration_mins = cost_of_new_route / 60.0
    
    if (new_order_demand <= vehicle_capacity) and (duration_mins <= max_route_duration_mins):
        # Find the first empty vehicle
        for v_id, route_orders in current_routes.items():
            if not route_orders:
                new_routes_state = copy.deepcopy(current_routes)
                new_routes_state[v_id].append(new_order)
                return new_routes_state, "New Vehicle"

    # 4. If all else fails, return None
    return None, "Failed"


def log_vehicle_changes(old_routes, new_routes, assigned_order, method, timestamp, time_matrix , all_locations , global_route_log):
    """
    Compares route states and logs completed/changed trips to the global log.
    This is a simplified logger.
    """
    # global global_route_log
    
    # This logic assumes a route is "changed" if the list of orders is different.
    # We log the *new* route that was created.
    
    # Find which vehicle was changed
    vehicle_id = -1
    for v_id, new_route_orders in new_routes.items():
        if not new_route_orders:
            continue
        
        # Check if this order is in the new route
        if assigned_order['id'] in [o['id'] for o in new_route_orders]:
            vehicle_id = v_id
            
            # Optimization: only log if the route is *new* or *different*
            old_route_ids = [o['id'] for o in old_routes.get(v_id, [])]
            new_route_ids = [o['id'] for o in new_route_orders]
            
            if old_route_ids != new_route_ids:
                route_cost = calculate_route_cost(new_route_orders, time_matrix)
                total_demand = sum(o['demand'] for o in new_route_orders)
                unique_stops = list(dict.fromkeys([o['index'] for o in new_route_orders]))
                
                # Log this new/updated trip
                global_route_log.append({
                    "vehicle_id": v_id,
                    "timestamp": timestamp,
                    "trigger_order": assigned_order['id'],
                    "method": method,
                    "stops": len(unique_stops),
                    "demand": total_demand,
                    "duration_min": route_cost,
                    "stop_names": [all_locations[idx]['original_address'].split(',')[0] for idx in unique_stops]
                })
            break # Stop after finding the vehicle that changed
def _tabu_search_capacity(initial_solution, time_matrix, 
                         vehicle_capacity, max_route_duration_mins,
                         iterations=50, tabu_tenure=10):
    """
    (Adapted from your v1)
    Performs a Tabu Search (2-opt swap) on routes made of *order objects*.
    It swaps *orders* in the list, and calculate_route_cost handles the
    change in path cost automatically.
    """
    if not initial_solution: 
        return None
        
    best_solution = copy.deepcopy(initial_solution)
    best_cost = calculate_total_cost(best_solution, time_matrix)
    current_solution = copy.deepcopy(initial_solution)
    tabu_list = deque(maxlen=tabu_tenure)
    
    for _ in range(iterations):
        neighborhood = []
        
        # Find all valid swaps
        for vehicle_id, route_orders in current_solution.items():
            if len(route_orders) > 1:
                for i in range(len(route_orders)):
                    for j in range(i + 1, len(route_orders)):
                        
                        # Get the *location indices* of the orders
                        order_i_idx = route_orders[i]['index']
                        order_j_idx = route_orders[j]['index']
                        
                        # Don't swap orders for the same location
                        if order_i_idx == order_j_idx:
                            continue
                        
                        # Check tabu list (uses location indices)
                        if (order_i_idx, order_j_idx) in tabu_list or (order_j_idx, order_i_idx) in tabu_list:
                            continue
                            
                        # Create neighbor by swapping *orders*
                        neighbor_route_orders = route_orders[:]
                        neighbor_route_orders[i], neighbor_route_orders[j] = neighbor_route_orders[j], neighbor_route_orders[i]
                        
                        # Check duration constraint
                        if calculate_route_cost(neighbor_route_orders, time_matrix) <= max_route_duration_mins:
                            # (Capacity is unchanged by a swap)
                            neighborhood.append((vehicle_id, neighbor_route_orders, (order_i_idx, order_j_idx)))
        
        if not neighborhood:
            break # No valid moves found

        # Find the best move in the neighborhood
        best_neighbor_vehicle, best_neighbor_route, best_neighbor_cost_change, best_tabu_move = -1, None, float('inf'), None
        
        for vehicle_id, neighbor_route_orders, tabu_move in neighborhood:
            cost_change = calculate_route_cost(neighbor_route_orders, time_matrix) - calculate_route_cost(current_solution[vehicle_id], time_matrix)
            
            if cost_change < best_neighbor_cost_change:
                best_neighbor_cost_change = cost_change
                best_neighbor_vehicle = vehicle_id
                best_neighbor_route = neighbor_route_orders
                best_tabu_move = tabu_move
                
        if best_neighbor_vehicle != -1:
            # Apply the best move
            current_solution[best_neighbor_vehicle] = best_neighbor_route
            tabu_list.append(best_tabu_move)
            
            current_cost = calculate_total_cost(current_solution, time_matrix)
            if current_cost < best_cost:
                best_solution = copy.deepcopy(current_solution)
                best_cost = current_cost
                
    return best_solution

# --- ALNS Helper: Roulette Wheel Selection ---
def _roulette_wheel_selection(weights):
    total_weight = sum(weights)
    if total_weight == 0:
        # Fallback if all weights somehow become zero
        return random.randrange(len(weights))
    
    pick = random.uniform(0, total_weight)
    current = 0
    for i, weight in enumerate(weights):
        current += weight
        if current > pick:
            return i
    # Should not happen if total_weight > 0, but as a fallback:
    return len(weights) - 1
# --- NEW MAIN L1 Function ---
def assign_new_order_realtime(new_order, current_routes, time_matrix, 
                            vehicle_capacity, max_route_duration_mins):
    """
    (Adapted from your v1)
    Orchestrator for Layer 1 assignment.
    1. Runs a greedy insertion.
    2. Runs a Tabu Search refinement.
    3. Returns the best solution found.
    """
    
    # 1. Find initial solution with Greedy Insertion
    greedy_solution, method = _greedy_insert_capacity(
        new_order, current_routes, time_matrix, 
        vehicle_capacity, max_route_duration_mins
    )
    
    if greedy_solution is None:
        return None, "Failed"
        
    greedy_total_cost = calculate_total_cost(greedy_solution, time_matrix)
    
    # 2. Refine the greedy solution with Tabu Search
    tabu_solution = _tabu_search_capacity(
        greedy_solution, time_matrix, 
        vehicle_capacity, max_route_duration_mins,
        iterations=50, tabu_tenure=7 # L1 should be fast
    )
    
    if tabu_solution:
        tabu_total_cost = calculate_total_cost(tabu_solution, time_matrix)
        
        # Compare costs and return the best
        if tabu_total_cost < greedy_total_cost - 0.1: # Use 0.1 min threshold
            return tabu_solution, "Tabu Search"
    
    # If tabu failed or wasn't better, return the greedy one
    return greedy_solution, method # "Best Insertion" or "New Vehicle"

# --- ALNS Repair Operators ---
def _repair_greedy(partial_solution_routes, request_bank, time_matrix, distance_matrix,
                   vehicle_capacity, max_route_duration_mins, num_vehicles):
    """
    Reinserts orders from request_bank into partial_solution using best insertion heuristic.
    Returns: repaired_solution_routes, uninserted_orders
    """
    repaired_solution = copy.deepcopy(partial_solution_routes)
    uninserted_orders = []
    
    # Optional: Shuffle request bank to avoid bias
    random.shuffle(request_bank)
    
    for order in request_bank:
        best_cost_increase = float('inf')
        best_insertion_vehicle = -1
        best_insertion_index = -1
        insertion_found = False

        new_order_demand = order['demand']

        # Try inserting into existing routes
        for v_id, route_orders in repaired_solution.items():
            current_load = sum(o['demand'] for o in route_orders)

            # --- Constraint 1: Check Capacity ---
            if current_load + new_order_demand > vehicle_capacity:
                continue

            # Calculate original route time cost (only needed if route is not empty)
            original_cost_seconds = 0
            if route_orders:
                 original_stop_indices = list(dict.fromkeys([o['index'] for o in route_orders]))
                 original_cost_seconds = calculate_raw_route_time(original_stop_indices, time_matrix)

            # Try inserting at every position
            for i in range(len(route_orders) + 1):
                temp_route_orders = route_orders[:i] + [order] + route_orders[i:]
                new_unique_stops = list(dict.fromkeys([o['index'] for o in temp_route_orders]))
                new_cost_seconds = calculate_raw_route_time(new_unique_stops, time_matrix)
                new_duration_minutes = new_cost_seconds / 60.0

                # --- Constraint 2: Check Duration ---
                if new_duration_minutes > max_route_duration_mins:
                    continue

                cost_increase = new_cost_seconds - original_cost_seconds

                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_insertion_vehicle = v_id
                    best_insertion_index = i
                    insertion_found = True

        # If insertion found, apply it
        if insertion_found:
            repaired_solution[best_insertion_vehicle].insert(best_insertion_index, order)
        else:
            # Try starting a new route on an empty vehicle
            cost_of_new_route_sec = calculate_raw_route_time([order['index']], time_matrix)
            duration_mins = cost_of_new_route_sec / 60.0
            
            can_start_new_route = False
            if (new_order_demand <= vehicle_capacity) and (duration_mins <= max_route_duration_mins):
                # Find the first empty vehicle
                for v_id in range(num_vehicles):
                    if v_id not in repaired_solution or not repaired_solution[v_id]:
                        if v_id not in repaired_solution: # Ensure key exists
                             repaired_solution[v_id] = []
                        repaired_solution[v_id].append(order)
                        can_start_new_route = True
                        break # Stop after finding one empty vehicle
            
            if not can_start_new_route:
                 uninserted_orders.append(order) # Could not insert this order

    return repaired_solution, uninserted_orders

# --- ALNS Destroy Operators ---
def _destroy_random(solution_routes, num_to_remove):
    """
    Removes num_to_remove randomly selected orders from the solution.
    Returns: partial_solution_routes, request_bank (list of removed order objects)
    """
    partial_solution = copy.deepcopy(solution_routes)
    request_bank = []
    
    # Create a flat list of all assigned (vehicle_id, order_index_in_route, order_object)
    assigned_orders_info = []
    for v_id, route in partial_solution.items():
        for idx, order in enumerate(route):
            assigned_orders_info.append({'v_id': v_id, 'idx': idx, 'order': order})

    if not assigned_orders_info:
        return partial_solution, [] # Nothing to remove

    # Determine actual number to remove (don't exceed available orders)
    actual_num_to_remove = min(num_to_remove, len(assigned_orders_info))
    if actual_num_to_remove <= 0:
         return partial_solution, []

    # Shuffle and select orders to remove
    random.shuffle(assigned_orders_info)
    orders_to_remove_info = assigned_orders_info[:actual_num_to_remove]
    
    # Store removed orders and create a quick lookup for removal
    removed_lookup = {} # Key: v_id, Value: list of indices to remove (sorted desc)
    for info in orders_to_remove_info:
        request_bank.append(info['order'])
        v_id = info['v_id']
        if v_id not in removed_lookup:
            removed_lookup[v_id] = []
        removed_lookup[v_id].append(info['idx'])

    # Remove orders efficiently (by index, descending to avoid shifting issues)
    for v_id, indices in removed_lookup.items():
        indices.sort(reverse=True) # Sort descending
        route = partial_solution[v_id]
        for idx in indices:
            del route[idx] # Remove by index

    return partial_solution, request_bank
# --- Main Function 3: Layer 2 (Batch VRP Optimization) ---

def batch_optimization_vrp(current_routes, pending_orders, time_matrix, 
                         num_vehicles, vehicle_capacity, max_route_duration_mins):
    """
    Re-optimizes all current routes AND tries to include pending orders.
    
    *** NEW LOGIC ***
    This version treats every individual ORDER as a unique stop.
    This solves the "super-order" problem where demand for one location
    (e.g., 50 units for Aura Pizzas) exceeds a single vehicle's capacity (e.g., 20).
    The solver can now create multiple routes to the same location.
    """
    
    # 1. Combine all orders (from routes + pending) into one big list
    all_orders_to_assign = pending_orders[:]
    for route in current_routes.values():
        all_orders_to_assign.extend(route)
        
    if not all_orders_to_assign:
        # Nothing to do
        return {i: [] for i in range(num_vehicles)}, []

    # 2. Create the "Solver Data Model"
    
    # The "solver locations" are now the DEPOT (index 0) + all individual orders.
    # If we have 79 orders, we have 80 "solver locations".
    num_orders = len(all_orders_to_assign)
    num_solver_locs = num_orders + 1 # (Depot + all orders)
    
    # Create mapping
    # map_solver_to_order[1] -> all_orders_to_assign[0]
    # map_solver_to_order[2] -> all_orders_to_assign[1]
    # ...
    # (index 0 is reserved for the depot)
    map_solver_to_order = {i + 1: order for i, order in enumerate(all_orders_to_assign)}

    # 3. Build the inputs for the solver engine
    
    # a) Solver Time Matrix (an (N+1) x (N+1) matrix)
    solver_time_matrix = [[0] * num_solver_locs for _ in range(num_solver_locs)]
    
    # This is now a big loop.
    for i in range(num_solver_locs):
        for j in range(num_solver_locs):
            if i == j:
                continue
                
            # Get the "original" location index (from all_locations)
            if i == 0 and j > 0:
                # Depot to Order
                order_j_loc_idx = map_solver_to_order[j]['index']
                solver_time_matrix[i][j] = time_matrix[0][order_j_loc_idx]
            elif i > 0 and j == 0:
                # Order to Depot
                order_i_loc_idx = map_solver_to_order[i]['index']
                solver_time_matrix[i][j] = time_matrix[order_i_loc_idx][0]
            elif i > 0 and j > 0:
                # Order to Order
                order_i_loc_idx = map_solver_to_order[i]['index']
                order_j_loc_idx = map_solver_to_order[j]['index']
                solver_time_matrix[i][j] = time_matrix[order_i_loc_idx][order_j_loc_idx]
            
    # b) Solver Demands List (one entry for each solver location)
    # The demand for the depot (index 0) is 0.
    # The demand for solver_loc 1 is the demand of order 1.
    solver_demands = [0] + [order['demand'] for order in all_orders_to_assign]

    # c) Vehicle Capacities & Durations
    vehicle_capacities = [vehicle_capacity] * num_vehicles
    # Solver needs duration in seconds
    vehicle_max_durations_sec = [int(max_route_duration_mins * 60)] * num_vehicles

    # 4. Call the Solver Engine!
    # The solver now sees 80 locations, each with its own small demand.
    solution_routes_solver, unassigned_solver_indices = solve_vrp_with_capacity(
        solver_time_matrix,
        solver_demands,
        vehicle_capacities,
        vehicle_max_durations_sec,
        num_vehicles
    )
    
    # 5. Parse the solution (convert solver indices back to order objects)
    
    new_optimized_routes = {i: [] for i in range(num_vehicles)}
    
    # Keep track of which orders (by solver index) were assigned
    assigned_solver_indices = set()
    
    for v_id, vehicle_route_solver in enumerate(solution_routes_solver):
        for solver_stop_idx in vehicle_route_solver:
            # solver_stop_idx is a solver index, e.g., 1, 2, ... 79
            
            # Get the full order object from our map
            order_obj = map_solver_to_order.get(solver_stop_idx)
            
            if order_obj:
                new_optimized_routes[v_id].append(order_obj)
                assigned_solver_indices.add(solver_stop_idx)

    # 6. Find any orders that were *not* assigned
    final_unassigned_orders = []
    
    # Check any orders returned by the solver
    for unassigned_idx in unassigned_solver_indices:
        order_obj = map_solver_to_order.get(unassigned_idx)
        if order_obj:
            final_unassigned_orders.append(order_obj)
            
    # Also check our *original* list for any orders not in the "assigned" set
    for solver_idx, order_obj in map_solver_to_order.items():
        if solver_idx not in assigned_solver_indices and solver_idx not in unassigned_solver_indices:
            # This order was not in *any* solution route
            final_unassigned_orders.append(order_obj)

    # De-duplicate the unassigned list
    final_unassigned_orders_deduped = []
    seen_ids = set()
    for order in final_unassigned_orders:
        if order['id'] not in seen_ids:
            final_unassigned_orders_deduped.append(order)
            seen_ids.add(order['id'])

    return new_optimized_routes, final_unassigned_orders_deduped


# --- Main ALNS Function ---
def run_alns_optimization(current_routes_input, pending_orders_input, time_matrix, distance_matrix,
                          num_vehicles, vehicle_capacity, max_route_duration_mins,
                          fixed_cost_per_truck, variable_cost_per_km, # Add cost params
                          alns_iterations=ALNS_ITERATIONS):
    """
    Performs Adaptive Large Neighborhood Search (ALNS) to optimize routes.
    Starts with the provided routes/pending orders and tries to improve them.
    Returns the best solution found (routes_dict, unassigned_orders_list).
    """
    print(f"--- [LAYER 3 ALNS] Starting optimization for {alns_iterations} iterations... ---")
    start_time_alns = time.time()

    # --- Initialize ---
    # Combine all orders into one pool for the initial insertion attempt
    initial_pending = pending_orders_input[:]
    initial_routes = copy.deepcopy(current_routes_input)
    for route in initial_routes.values():
        initial_pending.extend(route)
    
    # Try a simple initial solution: Greedily insert all pending orders
    # Start with empty routes
    initial_solution_routes = {v_id: [] for v_id in range(num_vehicles)}
    initial_solution_routes, initial_unassigned = _repair_greedy(
        initial_solution_routes, initial_pending, time_matrix, distance_matrix,
        vehicle_capacity, max_route_duration_mins, num_vehicles
    )
    
    # If initial greedy insert fails badly, fall back to input routes + try inserting pending
    if len(initial_unassigned) > len(pending_orders_input):
         print("--- [LAYER 3 ALNS] Initial greedy insertion performed poorly, starting from input routes... ---")
         initial_solution_routes = copy.deepcopy(current_routes_input)
         initial_solution_routes, initial_unassigned = _repair_greedy(
             initial_solution_routes, pending_orders_input[:], time_matrix, distance_matrix,
             vehicle_capacity, max_route_duration_mins, num_vehicles
        )

    print(f"--- [LAYER 3 ALNS] Initial solution created with {len(initial_unassigned)} unassigned orders. ---")

    current_solution_routes = initial_solution_routes
    current_unassigned = initial_unassigned
    current_cost, _, _ = calculate_total_fleet_cost(current_solution_routes, distance_matrix, fixed_cost_per_truck, variable_cost_per_km)
    # Add penalty for unassigned orders to guide the search
    current_objective = current_cost + (len(current_unassigned) * fixed_cost_per_truck * 10) # Heavy penalty

    best_solution_routes = current_solution_routes
    best_unassigned = current_unassigned
    best_cost = current_cost
    best_objective = current_objective

    # --- Operators ---
    # For now, only one of each. Extend these lists later.
    destroy_operators = [_destroy_random]
    repair_operators = [_repair_greedy]
    
    # --- Operator Weights & Scores ---
    destroy_weights = [1.0] * len(destroy_operators)
    repair_weights = [1.0] * len(repair_operators)
    destroy_scores = [0.0] * len(destroy_operators)
    repair_scores = [0.0] * len(repair_operators)
    destroy_counts = [0] * len(destroy_operators)
    repair_counts = [0] * len(repair_operators)

    # --- Temperature for Acceptance ---
    temperature = ALNS_TEMP_START

    # --- ALNS Main Loop ---
    for i in range(alns_iterations):
        
        # --- 1. Select Operators ---
        destroy_op_idx = _roulette_wheel_selection(destroy_weights)
        repair_op_idx = _roulette_wheel_selection(repair_weights)
        destroy_op = destroy_operators[destroy_op_idx]
        repair_op = repair_operators[repair_op_idx]
        
        destroy_counts[destroy_op_idx] += 1
        repair_counts[repair_op_idx] += 1

        # --- 2. Destroy ---
        # Determine how many orders to remove
        total_assigned_now = sum(len(r) for r in current_solution_routes.values())
        if total_assigned_now == 0: continue # Skip if no orders are assigned

        destroy_percent = random.uniform(ALNS_DESTROY_MIN_PERCENT, ALNS_DESTROY_MAX_PERCENT)
        num_to_remove = max(1, int(total_assigned_now * destroy_percent))

        partial_routes, request_bank = destroy_op(current_solution_routes, num_to_remove)
        
        # Add any currently unassigned orders to the request bank too
        request_bank.extend(current_unassigned)
        
        if not request_bank: continue # Nothing to repair

        # --- 3. Repair ---
        new_solution_routes, new_unassigned = repair_op(
            partial_routes, request_bank, time_matrix, distance_matrix,
            vehicle_capacity, max_route_duration_mins, num_vehicles
        )

        # --- 4. Evaluate New Solution ---
        new_cost, _, _ = calculate_total_fleet_cost(new_solution_routes, distance_matrix, fixed_cost_per_truck, variable_cost_per_km)
        new_objective = new_cost + (len(new_unassigned) * fixed_cost_per_truck * 10) # Apply penalty

        # --- 5. Acceptance Criterion ---
        accepted = False
        score_update = 0
        
        delta_objective = new_objective - current_objective

        if delta_objective < 0:
            # Improvement found
            current_solution_routes = new_solution_routes
            current_unassigned = new_unassigned
            current_objective = new_objective
            accepted = True
            
            if new_objective < best_objective:
                # New global best found
                best_solution_routes = copy.deepcopy(new_solution_routes) # Deep copy best
                best_unassigned = new_unassigned[:]
                best_objective = new_objective
                best_cost = new_cost # Store the actual cost without penalty
                score_update = ALNS_SIGMA1
                # print(f"Iter {i}: New best found! Cost={best_cost:.2f}, Unassigned={len(best_unassigned)}") # Optional: Log improvements
            else:
                score_update = ALNS_SIGMA2
        elif temperature > ALNS_TEMP_END:
            # Acceptance probability for worse solutions (Simulated Annealing)
            acceptance_prob = math.exp(-delta_objective / temperature)
            if random.random() < acceptance_prob:
                current_solution_routes = new_solution_routes
                current_unassigned = new_unassigned
                current_objective = new_objective
                accepted = True
                score_update = ALNS_SIGMA3

        # --- 6. Update Operator Scores ---
        if accepted:
             destroy_scores[destroy_op_idx] += score_update
             repair_scores[repair_op_idx] += score_update

        # --- 7. Update Temperature ---
        temperature *= ALNS_COOLING_RATE

        # --- 8. Update Operator Weights Periodically ---
        if (i + 1) % ALNS_SEGMENT_LENGTH == 0:
            for op_idx in range(len(destroy_operators)):
                if destroy_counts[op_idx] > 0:
                     avg_score = destroy_scores[op_idx] / destroy_counts[op_idx]
                     destroy_weights[op_idx] = (1 - ALNS_REACTION_FACTOR) * destroy_weights[op_idx] + \
                                               ALNS_REACTION_FACTOR * avg_score
                # Reset scores/counts for next segment
                destroy_scores[op_idx] = 0.0
                destroy_counts[op_idx] = 0
                
            for op_idx in range(len(repair_operators)):
                if repair_counts[op_idx] > 0:
                     avg_score = repair_scores[op_idx] / repair_counts[op_idx]
                     repair_weights[op_idx] = (1 - ALNS_REACTION_FACTOR) * repair_weights[op_idx] + \
                                              ALNS_REACTION_FACTOR * avg_score
                # Reset scores/counts
                repair_scores[op_idx] = 0.0
                repair_counts[op_idx] = 0
            # print(f"Iter {i}: Updated operator weights. D:{destroy_weights}, R:{repair_weights}") # Optional debug

    # --- End of ALNS Loop ---
    end_time_alns = time.time()
    print(f"--- [LAYER 3 ALNS] Finished {alns_iterations} iterations in {end_time_alns - start_time_alns:.2f} seconds. ---")
    print(f"--- [LAYER 3 ALNS] Best solution found: Cost={best_cost:.2f}, Unassigned={len(best_unassigned)} ---")

    # Return the best solution found during the search
    return best_solution_routes, best_unassigned