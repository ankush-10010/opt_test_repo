import time
import random
import json
import threading
import pandas as pd
from datetime import datetime, timedelta
import webbrowser
import os
import io
import base64

# NEW: Import for static chart generation
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt

# Assuming these are in a separate file named hybrid_solver_layers.py
from hybrid_solver_layers import (
    assign_new_order_realtime,
    calculate_route_cost,
    batch_optimization_vrp,
    calculate_total_fleet_cost,
    run_alns_optimization
)

random.seed(42)

# --- Configuration ---
SIMULATION_START_HOUR = 9
SIMULATION_END_HOUR = 22
MINUTES_PER_TICK = 10
NUM_VEHICLES = 1000
TIME_MATRIX_FILE = 'matrix_data_with_distance.json'

VEHICLE_CAPACITY = 200
MAX_ROUTE_DURATION_MINS = 200
PREPROCESSED_ORDER_FILE = 'preprocessed_orders_single_day.csv'
# SIMULATION_DAY_OF_YEAR = 254
SIMULATION_DAY_OF_YEAR = 254

LAYER_2_INTERVAL_SECONDS = 60
OUTPUT_HTML_FILE = 'outputs/hybrid_simulation_live_capacity.html'

FIXED_COST_PER_TRUCK = 5000
VARIABLE_COST_PER_KM = 15

DISTANCE_MATRIX_FILE = 'distance_matrix.json'
distance_matrix = []

# --- Shared State ---
current_routes = {}
pending_orders = []
state_lock = threading.Lock()
simulation_running = True
simulation_events = []
all_locations = []
time_matrix = []
global_order_assignments_log = []
simulation_start_time = None

# NEW: Synchronization Event for final optimization
optimizer_finished_event = threading.Event()

# NEW: Additional metrics tracking
optimization_performance_log = []  # Track each optimization cycle performance
vehicle_utilization_history = []  # Track vehicle usage over time
order_wait_times = {}  # Track how long orders waited before assignment
cost_over_time_log = [] # NEW: Track L1 cost vs L2/L3 optimized cost
pending_orders_history = [] # NEW: Track pending orders over time

# --- HTML Generation Functions ---
def format_time(minutes_from_start):
    """Convert minutes from 9:00 AM to readable time"""
    start_dt = datetime.now().replace(hour=SIMULATION_START_HOUR, minute=0, second=0, microsecond=0)
    result_dt = start_dt + timedelta(minutes=minutes_from_start)
    return result_dt.strftime("%I:%M %p")

def generate_route_coordinates(route_orders, locations):
    """Generate coordinate pairs for a route from a list of order objects"""
    coords = []
    if not locations:
        return []
        
    sample_loc = locations[0]
    lat_key, lng_key = ('latitude', 'longitude')
    if 'lat' in sample_loc:
        lat_key, lng_key = 'lat', 'lng'
    
    coords.append({
        'lat': locations[0][lat_key], 'lng': locations[0][lng_key],
        'address': locations[0]['original_address'], 'type': 'depot', 'index': 0
    })
    
    stop_indices = []
    if route_orders:
        stop_indices = list(dict.fromkeys([order['index'] for order in route_orders]))

    for idx in stop_indices:
        if idx < len(locations):
            coords.append({
                'lat': locations[idx][lat_key], 'lng': locations[idx][lng_key],
                'address': locations[idx]['original_address'], 'type': 'stop', 'index': idx
            })
        else:
            print(f"Warning: Index {idx} out of bounds for locations.")
            
    coords.append({
        'lat': locations[0][lat_key], 'lng': locations[0][lng_key],
        'address': locations[0]['original_address'], 'type': 'depot', 'index': 0
    })
    return coords


# IMPORTANT: Replace with your own API key
GOOGLE_MAPS_API_KEY = 'AIzaSyC_hI6BowrJPojeBiRldmuFVf3aqsSRZbg'

# NEW: Function to generate static charts with Matplotlib
def generate_static_charts(cost_log, wait_times, pending_history):
    """
    Generates Matplotlib charts and returns them as Base64 encoded strings.
    """
    charts = {}
    plt.style.use('seaborn-v0_8-darkgrid')

    try:
        # --- 1. Cost vs. Optimized Cost Chart ---
        if cost_log:
            l1_data = [d for d in cost_log if d['type'] == 'l1_assignment']
            opt_data = [d for d in cost_log if d['type'] == 'optimization']
            
            plt.figure(figsize=(10, 5))
            if l1_data:
                plt.step([d['time'] for d in l1_data], [d['cost'] for d in l1_data], 
                         label='Layer 1 Cost (Post-Assignment)', color='#f87171', where='post')
            if opt_data:
                plt.plot([d['time'] for d in opt_data], [d['cost'] for d in opt_data], 
                         label='Optimized Cost (L2/L3)', color='#4ade80', marker='o', linestyle='--')
            
            plt.title('💸 L1 Cost (Greedy) vs. Optimized Cost (L2/L3)', fontsize=14)
            plt.xlabel('Time (Minutes from Simulation Start)', fontsize=10)
            plt.ylabel('Total Fleet Cost', fontsize=10)
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            charts['cost_chart'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        else:
            charts['cost_chart'] = None

        # --- 2. Wait Time Histogram ---
        if wait_times:
            wait_values = list(wait_times.values())
            buckets = {'0-5 min': 0, '5-10 min': 0, '10-30 min': 0, '30+ min': 0}
            for wait in wait_values:
                if wait <= 5: buckets['0-5 min'] += 1
                elif wait <= 10: buckets['5-10 min'] += 1
                elif wait <= 30: buckets['10-30 min'] += 1
                else: buckets['30+ min'] += 1
            
            plt.figure(figsize=(10, 5))
            colors = ['#4ade80', '#fbbf24', '#f87171', '#dc3545']
            plt.bar(buckets.keys(), buckets.values(), color=colors)
            plt.title('⏱️ Order Wait Time Distribution', fontsize=14)
            plt.xlabel('Wait Time Buckets', fontsize=10)
            plt.ylabel('Number of Orders', fontsize=10)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            charts['wait_chart'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        else:
            charts['wait_chart'] = None

        # --- 3. Pending Orders Chart ---
        if pending_history:
            plt.figure(figsize=(10, 5))
            plt.step([d['time'] for d in pending_history], [d['count'] for d in pending_history], 
                     label='Pending Orders', color='#fbbf24', where='post')
            plt.fill_between([d['time'] for d in pending_history], [d['count'] for d in pending_history], 
                             step='post', color='#fbbf24', alpha=0.2)
            plt.title('📈 Pending Orders Over Time (System Stress)', fontsize=14)
            plt.xlabel('Time (Minutes from Simulation Start)', fontsize=10)
            plt.ylabel('Number of Pending Orders', fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            charts['pending_chart'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        else:
            charts['pending_chart'] = None

    except Exception as e:
        print(f"Error generating Matplotlib charts: {e}")
        return {'cost_chart': None, 'wait_chart': None, 'pending_chart': None}

    return charts

def generate_html_report():
    """Generate comprehensive HTML report of the simulation"""
    
    map_routes = []
    try:
        with state_lock:
            # Ensure we are reading the (potentially) final, optimized routes
            routes_for_report = {vid: r[:] for vid, r in current_routes.items()}
            pending_for_report = pending_orders[:]
            
            # Get snapshots of logs for chart generation
            cost_log_snapshot = cost_over_time_log[:]
            wait_times_snapshot = order_wait_times.copy()
            pending_history_snapshot = pending_orders_history[:]

        for v_id in sorted(routes_for_report.keys()):
            route_orders = routes_for_report[v_id]
            if route_orders:
                coords = generate_route_coordinates(route_orders, all_locations)
                map_routes.append({
                    'vehicle_id': v_id,
                    'coordinates': coords,
                    'color': ['#667eea', '#28a745', '#ffc107', '#dc3545'][v_id % 4]
                })
    except Exception as e:
        print(f"Warning: Could not generate map routes: {e}")
        map_routes = []
    
    map_routes_json = json.dumps(map_routes)
    time_matrix_json = json.dumps(time_matrix)
    
    # NEW: Generate static charts
    print("Generating static charts with Matplotlib...")
    static_charts = generate_static_charts(cost_log_snapshot, wait_times_snapshot, pending_history_snapshot)
    print("...Charts generated.")

    # NEW: Calculate additional metrics
    total_orders = len([e for e in simulation_events if e['type'] == 'new_order'])
    assigned_orders = len(global_order_assignments_log)
    optimization_count = len([e for e in simulation_events if e['type'] == 'optimization'])
    
    with state_lock:
        # Use the routes captured at the start of this function
        pending_count = len(pending_for_report)
        active_vehicles = len([r for r in routes_for_report.values() if r])
        avg_duration = 0
        if active_vehicles > 0:
            total_duration = sum([calculate_route_cost(r, time_matrix) for r in routes_for_report.values() if r])
            avg_duration = int(total_duration / active_vehicles) if total_duration > 0 else 0
        
        # NEW: Calculate utilization metrics
        total_capacity_used = sum([sum(order['demand'] for order in route) for route in routes_for_report.values() if route])
        total_capacity_available = NUM_VEHICLES * VEHICLE_CAPACITY
        fleet_utilization = (total_capacity_used / total_capacity_available * 100) if total_capacity_available > 0 else 0
    
    # Calculate average wait time for assigned orders
    avg_wait_time = 0
    if wait_times_snapshot:
        avg_wait_time = sum(wait_times_snapshot.values()) / len(wait_times_snapshot)
    
    # Calculate success rate
    success_rate = (assigned_orders / total_orders * 100) if total_orders > 0 else 0
    
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hybrid Delivery Simulation - Enhanced Dashboard</title>
    <!-- REMOVED Chart.js import -->
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px; color: #333;
        }
        .container {
            max-width: 1400px; margin: 0 auto; background: white;
            border-radius: 15px; box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 30px; text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.1em; opacity: 0.9; }
        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; padding: 30px; background: #f8f9fa;
        }
        .stat-card {
            background: white; padding: 25px; border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;
            transition: transform 0.3s;
        }
        .stat-card:hover { transform: translateY(-5px); }
        .stat-value {
            font-size: 2.5em; font-weight: bold; color: #667eea; margin: 10px 0;
        }
        .stat-label {
            color: #666; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;
        }
        .stat-trend {
            font-size: 0.85em; margin-top: 8px; padding: 4px 8px;
            border-radius: 12px; display: inline-block;
        }
        .trend-positive { background: #d4edda; color: #155724; }
        .trend-negative { background: #f8d7da; color: #721c24; }
        .trend-neutral { background: #e2e3e5; color: #383d41; }
        
        .content { padding: 30px; }
        .section { margin-bottom: 40px; }
        .section-title {
            font-size: 1.8em; color: #667eea; margin-bottom: 20px;
            padding-bottom: 10px; border-bottom: 3px solid #667eea;
        }
        
        /* NEW: Performance Chart Styles */
        .chart-container {
            background: white; padding: 25px; border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px;
        }
        .chart-title {
            font-size: 1.3em; color: #667eea; margin-bottom: 15px;
            font-weight: bold;
        }
        /* NEW: Style for static chart images */
        .chart-image {
            width: 100%;
            height: auto;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        
        /* NEW: Heatmap Styles */
        .heatmap-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(60px, 1fr));
            gap: 5px;
            margin-top: 20px;
        }
        .heatmap-cell {
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 6px;
            font-size: 0.85em;
            font-weight: bold;
            color: white;
            transition: transform 0.2s;
            cursor: pointer;
        }
        .heatmap-cell:hover {
            transform: scale(1.1);
            z-index: 10;
        }
        .heatmap-legend {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 15px;
            font-size: 0.9em;
        }
        .legend-gradient {
            width: 200px;
            height: 20px;
            border-radius: 10px;
            background: linear-gradient(to right, #4ade80, #fbbf24, #f87171);
        }
        
        /* NEW: Comparison Table Styles */
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .comparison-table th {
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: center; /* Center all headers */
            font-weight: bold;
            vertical-align: middle;
        }
        .comparison-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
            text-align: right; /* Default right-align all cells */
            font-size: 0.9em;
        }
        .comparison-table td:nth-child(1), .comparison-table td:nth-child(2), .comparison-table td:last-child {
            text-align: center; /* Center #, Time, and Winner */
        }
        .comparison-table tr:hover {
            background: #f8f9fa;
        }
        .winner-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
            background: #28a745;
            color: white;
        }
        
        #map {
            width: 100%; height: 600px; border-radius: 12px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2); margin-bottom: 20px;
        }
        .map-controls {
            display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap;
        }
        .vehicle-toggle {
            padding: 10px 20px; border: 2px solid #667eea; border-radius: 25px;
            background: white; cursor: pointer; transition: all 0.3s; font-weight: bold;
        }
        .vehicle-toggle.active { background: #667eea; color: white; }
        .vehicle-toggle:hover { transform: scale(1.05); }
        .distance-info {
            background: #f8f9fa; padding: 20px; border-radius: 10px;
            border-left: 4px solid #667eea; margin-top: 20px; display: none;
        }
        .distance-info.active { display: block; }
        .distance-info h4 { color: #667eea; margin-bottom: 10px; }
        .distance-details {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin-top: 15px;
        }
        .distance-detail {
            background: white; padding: 15px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .distance-label { font-size: 0.9em; color: #666; margin-bottom: 5px; }
        .distance-value { font-size: 1.5em; font-weight: bold; color: #667eea; }
        
        .timeline { position: relative; padding-left: 30px; }
        .timeline::before {
            content: ''; position: absolute; left: 0; top: 0; bottom: 0;
            width: 3px; background: linear-gradient(to bottom, #667eea, #764ba2);
        }
        .timeline-event {
            position: relative; margin-bottom: 30px; padding: 20px;
            background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea;
        }
        .timeline-event::before {
            content: ''; position: absolute; left: -36px; top: 20px;
            width: 12px; height: 12px; border-radius: 50%;
            background: #667eea; border: 3px solid white; box-shadow: 0 0 0 3px #667eea;
        }
        .event-time {
            font-weight: bold; color: #667eea; font-size: 1.1em; margin-bottom: 8px;
        }
        .event-type {
            display: inline-block; padding: 4px 12px; border-radius: 20px;
            font-size: 0.85em; font-weight: bold; margin-bottom: 10px;
        }
        .event-new-order { background: #ffd93d; color: #333; }
        .event-assignment { background: #6bcf7f; color: white; }
        .event-optimization { background: #667eea; color: white; }
        .event-rejected { background: #ff6b6b; color: white; }
        .event-details { color: #555; line-height: 1.6; }
        
        .vehicle-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .vehicle-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 25px; border-radius: 12px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .vehicle-header {
            font-size: 1.5em; font-weight: bold; margin-bottom: 15px;
            padding-bottom: 10px; border-bottom: 2px solid rgba(255,255,255,0.3);
        }
        .route-stop {
            padding: 10px; margin: 8px 0; background: rgba(255,255,255,0.2);
            border-radius: 6px; font-size: 0.95em; cursor: pointer;
            transition: all 0.2s ease-in-out;
        }
        .route-stop:hover {
            background: rgba(255,255,255,0.4); transform: scale(1.02);
        }
        .route-stop.highlighted {
            background: #ffc107 !important; color: #333 !important;
            transform: scale(1.05); font-weight: bold;
        }
        .route-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px; margin-top: 15px;
        }
        .metric {
            background: rgba(255,255,255,0.2); padding: 10px;
            border-radius: 6px; text-align: center;
        }
        .metric-value { font-size: 1.3em; font-weight: bold; }
        .metric-label { font-size: 0.85em; opacity: 0.9; }
        
        .pending-orders {
            background: #fff3cd; border: 2px solid #ffc107;
            border-radius: 10px; padding: 20px; margin-bottom: 20px;
        }
        .pending-orders h3 { color: #856404; margin-bottom: 15px; }
        .pending-order {
            background: white; padding: 12px; margin: 8px 0;
            border-radius: 6px; border-left: 4px solid #ffc107;
        }
        
        .map-legend {
            background: white; padding: 15px; border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
        }
        .legend-item { display: flex; align-items: center; margin: 8px 0; }
        .legend-icon {
            width: 30px; height: 30px; margin-right: 10px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center; font-weight: bold;
        }
        .legend-icon.depot { background: #dc3545; color: white; border-radius: 4px; }
        .legend-icon.stop { background: #667eea; color: white; }

        .log-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        .log-table th, .log-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .log-table th {
            background-color: #667eea;
            color: white;
            font-size: 0.9em;
            text-transform: uppercase;
        }
        .log-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .log-table tr:hover {
            background-color: #f1f1f1;
        }

        .collapsible-section {
            margin-bottom: 30px;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .collapsible-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 25px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s ease;
            user-select: none;
        }
        
        .collapsible-header:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
            transform: translateY(-2px);
        }
        
        .collapsible-title {
            font-size: 1.4em;
            font-weight: bold;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .collapsible-icon {
            font-size: 1.2em;
            transition: transform 0.3s ease;
        }
        
        .collapsible-icon.expanded {
            transform: rotate(180deg);
        }
        
        .collapsible-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease, padding 0.3s ease;
            background: white;
        }
        
        /* --- CSS FIX HERE --- */
        /* I removed the duplicate 'max-height: none;' property */
        .collapsible-content.expanded {
            overflow-y: auto; /* Allow internal scrolling if content overflows */
            max-height: 800px; /* Set a reasonable max height for the panel */
            padding: 25px;
        }
        /* --- END CSS FIX --- */
        
        .collapsible-content.collapsed {
            max-height: 0;
            padding: 0 25px;
        }
        
        .central-controls {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            text-align: center;
            border: 2px solid #e9ecef;
        }
        
        .central-controls h3 {
            margin: 0 0 15px 0;
            color: #495057;
            font-size: 1.3em;
        }
        
        .control-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .control-btn {
            padding: 12px 24px;
            border: none;
            border-radius: 25px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .control-btn.expand-all {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
        }
        
        .control-btn.expand-all:hover {
            background: linear-gradient(135deg, #218838 0%, #1ea085 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(40, 167, 69, 0.3);
        }
        
        .control-btn.collapse-all {
            background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
            color: white;
        }
        
        .control-btn.collapse-all:hover {
            background: linear-gradient(135deg, #c82333 0%, #e8650e 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(220, 53, 69, 0.3);
        }
        
        .control-btn.toggle-all {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .control-btn.toggle-all:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        }
        
        .section-icon {
            font-size: 1.3em;
            margin-right: 8px;
        }
        
        .fleet-status-content {
            padding: 0;
        }
        
        .timeline-content {
            padding: 0;
        }
        
        .log-content {
            padding: 0;
        }
        
        @media (max-width: 768px) {
            .control-buttons {
                flex-direction: column;
                align-items: center;
            }
            
            .control-btn {
                width: 200px;
            }
            
            .collapsible-header {
                padding: 15px 20px;
            }
            
            .collapsible-title {
                font-size: 1.2em;
            }
            
            .stats-grid {
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚚 Hybrid Delivery Simulation Dashboard</h1>
            <p>Real-time Vehicle Routing with Capacity Constraints & Advanced Analytics</p>
            <p><strong>Simulation Period:</strong> {{start_time}} - {{end_time}}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Orders</div>
                <div class="stat-value">{{total_orders}}</div>
            </div>

            <div class="stat-card">
                <div class="stat-label">Pending (End of Day)</div>
                <div class="stat-value">{{pending_count}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Active Vehicles</div>
                <div class="stat-value">{{active_vehicles}}/{{total_vehicles}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Fleet Utilization</div>
                <div class="stat-value">{{fleet_utilization}}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Optimizations</div>
                <div class="stat-value">{{optimization_count}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Route Duration</div>
                <div class="stat-value">{{avg_duration}} min</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Wait Time</div>
                <div class="stat-value">{{avg_wait_time}} min</div>
            </div>
        </div>
        
        <div class="content">
            {{pending_orders_section}}
            
            <div class="central-controls">
                <h3>📋 Section Controls</h3>
                <div class="control-buttons">
                    <button class="control-btn expand-all" onclick="expandAllSections()">
                        📖 Expand All
                    </button>
                    <button class="control-btn collapse-all" onclick="collapseAllSections()">
                        📕 Collapse All
                    </button>
                    <button class="control-btn toggle-all" onclick="toggleAllSections()">
                        🔄 Toggle All
                    </button>
                </div>
            </div>
            
            <!-- NEW: Performance Analytics Section -->
            <div class="collapsible-section" id="analytics-section">
                <div class="collapsible-header" onclick="toggleSection('analytics-section')">
                    <h3 class="collapsible-title">
                        <span class="section-icon">📊</span>
                        Performance Analytics
                    </h3>
                    <span class="collapsible-icon">▼</span>
                </div>
                <div class="collapsible-content" id="analytics-content">
                    {{analytics_section}}
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">🗺️ Interactive Route Map (Final State)</h2>
                <div class="map-legend">
                    <div class="legend-item">
                        <div class="legend-icon depot">D</div> <span>Depot (Start/End Point)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon stop">●</div> <span>Delivery Stop</span>
                    </div>
                </div>
                <div class="map-controls" id="vehicleToggles"></div>
                <div id="map"></div>
                <div class="distance-info" id="distanceInfo">
                    <h4>🔍 Route Segment Information</h4>
                    <p id="segmentDescription">Select two consecutive stops on a route to see travel details.</p>
                    <div class="distance-details" id="distanceDetails"></div>
                </div>
            </div>

            <div class="collapsible-section" id="historical-log-section">
                <div class="collapsible-header" onclick="toggleSection('historical-log-section')">
                    <h3 class="collapsible-title">
                        <span class="section-icon">📈</span>
                        Historical Order Assignment Log
                    </h3>
                    <span class="collapsible-icon">▼</span>
                </div>
                <div class="collapsible-content log-content" id="historical-log-content">
                    {{historical_log_section}}
                </div>
            </div>

            <div class="collapsible-section" id="fleet-status-section">
                <div class="collapsible-header" onclick="toggleSection('fleet-status-section')">
                    <h3 class="collapsible-title">
                        <span class="section-icon">🚚</span>
                        Final Fleet Status
                    </h3>
                    <span class="collapsible-icon">▼</span>
                </div>
                <div class="collapsible-content fleet-status-content" id="fleet-status-content">
                    <div class="vehicle-grid">
                        {{vehicle_cards}}
                    </div>
                </div>
            </div>
            
            <div class="collapsible-section" id="timeline-section">
                <div class="collapsible-header" onclick="toggleSection('timeline-section')">
                    <h3 class="collapsible-title">
                        <span class="section-icon">⏰</span>
                        Simulation Event Timeline
                    </h3>
                    <span class="collapsible-icon">▼</span>
                </div>
                <div class="collapsible-content timeline-content" id="timeline-content">
                    <div class="timeline">
                        {{timeline_events}}
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const routesData = {{map_routes_json}};
        const timeMatrix = {{time_matrix_json}};
        
        let map;
        let directionsService;
        let directionsRenderers = [];
        let markers = [];
        let activeRoutes = new Set();
        let selectedMarkers = [];
        let mapMarkers = {};
        let highlightRenderer = null;
        let activeHighlightVehicle = null;
        let selectedStopIndices = [];
        
        let sectionStates = {
            'analytics-section': false,
            'historical-log-section': false,
            'fleet-status-section': false,
            'timeline-section': false
        };

        // REMOVED all Chart.js rendering functions
        
        document.addEventListener('DOMContentLoaded', function() {
            Object.keys(sectionStates).forEach(sectionId => {
                const content = document.getElementById(sectionId.replace('-section', '-content'));
                if (content) {
                    content.classList.add('collapsed');
                }
            });
        });

        function initMap() {
            const depotLocation = routesData.length > 0 && routesData[0].coordinates.length > 0
                ? { lat: routesData[0].coordinates[0].lat, lng: routesData[0].coordinates[0].lng }
                : { lat: 22.7196, lng: 75.8577 };
            
            map = new google.maps.Map(document.getElementById('map'), {
                zoom: 12, center: depotLocation, mapTypeControl: true,
                streetViewControl: false, fullscreenControl: true
            });
            
            directionsService = new google.maps.DirectionsService();

            routesData.forEach(route => { mapMarkers[route.vehicle_id] = []; });
            
            const toggleContainer = document.getElementById('vehicleToggles');
            routesData.forEach((route, idx) => {
                const button = document.createElement('button');
                button.className = 'vehicle-toggle active';
                button.textContent = `Vehicle ${route.vehicle_id}`;
                button.style.borderColor = route.color;
                button.onclick = () => toggleRoute(route.vehicle_id, button);
                toggleContainer.appendChild(button);
                activeRoutes.add(route.vehicle_id);
            });
            
            routesData.forEach(route => { displayRoute(route); });
        }
        
        function toggleRoute(vehicleId, button) {
            if (activeRoutes.has(vehicleId)) {
                activeRoutes.delete(vehicleId);
                button.classList.remove('active');
            } else {
                activeRoutes.add(vehicleId);
                button.classList.add('active');
            }
            clearMap();
            routesData.forEach(route => {
                if (activeRoutes.has(route.vehicle_id)) { displayRoute(route); }
            });
        }
        
        function displayRoute(routeData) {
            const coords = routeData.coordinates;
            if (coords.length < 2) return;
            
            const origin = { lat: coords[0].lat, lng: coords[0].lng };
            const destination = { lat: coords[coords.length - 1].lat, lng: coords[coords.length - 1].lng };
            const waypoints = coords.slice(1, -1).map(coord => ({
                location: { lat: coord.lat, lng: coord.lng }, stopover: true
            }));
            
            const directionsRenderer = new google.maps.DirectionsRenderer({
                map: map, suppressMarkers: true,
                polylineOptions: {
                    strokeColor: routeData.color, strokeWeight: 5, strokeOpacity: 0.7
                }
            });
            
            const request = {
                origin: origin, destination: destination, waypoints: waypoints,
                travelMode: google.maps.TravelMode.DRIVING, optimizeWaypoints: false
            };
            
            directionsService.route(request, (result, status) => {
                if (status === 'OK') {
                    directionsRenderer.setDirections(result);
                } else {
                    console.error('Directions request failed:', status);
                    drawSimplePolyline(coords, routeData.color);
                }
            });
            
            directionsRenderers.push(directionsRenderer);
            
            coords.forEach((coord, idx) => {
                const isDepot = coord.type === 'depot';
                const markerIcon = {
                    path: isDepot ? google.maps.SymbolPath.FORWARD_CLOSED_ARROW : google.maps.SymbolPath.CIRCLE,
                    rotation: isDepot ? -90 : 0,
                    fillColor: isDepot ? '#dc3545' : routeData.color,
                    fillOpacity: 1, strokeColor: 'white', strokeWeight: 2.5,
                    scale: isDepot ? 9 : 10
                };
                const marker = new google.maps.Marker({
                    position: { lat: coord.lat, lng: coord.lng }, map: map,
                    title: coord.address, icon: markerIcon,
                    label: {
                        text: isDepot ? 'D' : (idx).toString(),
                        color: 'white', fontSize: '11px', fontWeight: 'bold'
                    }
                });
                
                mapMarkers[routeData.vehicle_id][idx] = { marker, coord };

                const infoWindow = new google.maps.InfoWindow({
                    content: `<div style="padding: 10px;">
                            <strong>${isDepot ? 'Depot' : 'Stop ' + idx}</strong><br>
                            ${coord.address}<br>
                            <small>Vehicle ${routeData.vehicle_id}</small>
                        </div>`
                });
                
                marker.addListener('click', () => {
                    infoWindow.open(map, marker);
                    handleMarkerClick(marker, coord, routeData);
                });
                markers.push({ marker, coord, routeData });
            });
        }
        
        function drawSimplePolyline(coords, color) {
            const path = coords.map(c => ({ lat: c.lat, lng: c.lng }));
            const polyline = new google.maps.Polyline({
                path: path, strokeColor: color, strokeWeight: 4,
                strokeOpacity: 0.7, map: map
            });
        }
        
        function handleMarkerClick(marker, coord, routeData) {
            if (selectedMarkers.length === 2) { selectedMarkers = []; }
            selectedMarkers.push({ marker, coord, routeData });
            if (selectedMarkers.length === 2) { calculateSegmentInfo(); }
        }
        
        function calculateSegmentInfo() {
            const [first, second] = selectedMarkers;
            if (first.routeData.vehicle_id !== second.routeData.vehicle_id) {
                document.getElementById('segmentDescription').textContent = 
                    'Please select two stops from the same vehicle route.';
                return;
            }
            const route = first.routeData.coordinates;
            const firstIdx = route.findIndex(c => c.lat === first.coord.lat && c.lng === first.coord.lng);
            const secondIdx = route.findIndex(c => c.lat === second.coord.lat && c.lng === second.coord.lng);
            if (firstIdx === -1 || secondIdx === -1) return;
            
            const travelTime = timeMatrix[first.coord.index][second.coord.index];
            const distance = calculateDistance(
                first.coord.lat, first.coord.lng, second.coord.lat, second.coord.lng
            );
            
            const distanceInfo = document.getElementById('distanceInfo');
            const distanceDetails = document.getElementById('distanceDetails');
            document.getElementById('segmentDescription').textContent = 
                `Route segment from ${first.coord.address.split(',')[0]} to ${second.coord.address.split(',')[0]}`;
            
            distanceDetails.innerHTML = `
                <div class="distance-detail">
                    <div class="distance-label">Travel Time</div>
                    <div class="distance-value">${travelTime.toFixed(1)} min</div>
                </div>
                <div class="distance-detail">
                    <div class="distance-label">Distance</div>
                    <div class="distance-value">${distance.toFixed(1)} km</div>
                </div>
                <div class="distance-detail">
                    <div class="distance-label">Vehicle</div>
                    <div class="distance-value">${first.routeData.vehicle_id}</div>
                </div>
                <div class="distance-detail">
                    <div class="distance-label">Avg Speed</div>
                    <div class="distance-value">${(distance / (travelTime / 60)).toFixed(1)} km/h</div>
                </div>
            `;
            distanceInfo.classList.add('active');
        }
        
        function calculateDistance(lat1, lon1, lat2, lon2) {
            const R = 6371;
            const dLat = (lat2 - lat1) * Math.PI / 180;
            const dLon = (lon2 - lon1) * Math.PI / 180;
            const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                    Math.sin(dLon/2) * Math.sin(dLon/2);
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            return R * c;
        }
        
        function clearMap() {
            directionsRenderers.forEach(renderer => renderer.setMap(null));
            markers.forEach(m => m.marker.setMap(null));
            directionsRenderers = []; markers = []; selectedMarkers = [];
            document.getElementById('distanceInfo').classList.remove('active');
        }

        function highlightStop(element, vehicleId, routeIndex) {
            if (vehicleId !== activeHighlightVehicle) {
                document.querySelectorAll('.route-stop.highlighted').forEach(el => {
                    el.classList.remove('highlighted');
                });
                selectedStopIndices = [];
                activeHighlightVehicle = vehicleId;
            }
            const selectionIndex = selectedStopIndices.indexOf(routeIndex);
            if (selectionIndex > -1) {
                selectedStopIndices.splice(selectionIndex, 1);
                element.classList.remove('highlighted');
            } else {
                selectedStopIndices.push(routeIndex);
                element.classList.add('highlighted');
            }
            updateHighlightPath();
        }

        function updateHighlightPath() {
            if (highlightRenderer) {
                highlightRenderer.setMap(null); highlightRenderer = null;
            }
            for (const vId in mapMarkers) {
                mapMarkers[vId].forEach(m => m.marker.setAnimation(null));
            }
            if (selectedStopIndices.length === 0 || activeHighlightVehicle === null) return;

            selectedStopIndices.forEach(idx => {
                if (mapMarkers[activeHighlightVehicle] && mapMarkers[activeHighlightVehicle][idx]) {
                    mapMarkers[activeHighlightVehicle][idx].marker.setAnimation(google.maps.Animation.BOUNCE);
                }
            });

            const routeData = routesData.find(r => r.vehicle_id === activeHighlightVehicle);
            if (!routeData) return;
            const lastStopIndex = Math.max(...selectedStopIndices);
            const highlightCoords = routeData.coordinates.slice(0, lastStopIndex + 1);
            if (highlightCoords.length < 2) return;

            const origin = { lat: highlightCoords[0].lat, lng: highlightCoords[0].lng };
            const destination = { lat: highlightCoords[highlightCoords.length - 1].lat, lng: highlightCoords[highlightCoords.length - 1].lng };
            const waypoints = highlightCoords.slice(1, -1).map(c => ({
                location: { lat: c.lat, lng: c.lng }, stopover: true
            }));
            
            highlightRenderer = new google.maps.DirectionsRenderer({
                map: map, suppressMarkers: true, preserveViewport: true,
                polylineOptions: {
                    strokeColor: '#e74c3c', strokeWeight: 8,
                    strokeOpacity: 0.9, zIndex: 99
                }
            });

            const request = {
                origin: origin, destination: destination, waypoints: waypoints,
                travelMode: google.maps.TravelMode.DRIVING, optimizeWaypoints: false
            };

            directionsService.route(request, (result, status) => {
                if (status === 'OK') { highlightRenderer.setDirections(result); }
            });
        }
        
        function toggleSection(sectionId) {
            const section = document.getElementById(sectionId);
            const content = document.getElementById(sectionId.replace('-section', '-content'));
            const icon = section.querySelector('.collapsible-icon');
            const isExpanded = sectionStates[sectionId];
            
            if (isExpanded) {
                content.classList.remove('expanded');
                content.classList.add('collapsed');
                icon.classList.remove('expanded');
                sectionStates[sectionId] = false;
            } else {
                content.classList.remove('collapsed');
                content.classList.add('expanded');
                icon.classList.add('expanded');
                sectionStates[sectionId] = true;
            }
        }
        
        function expandAllSections() {
            Object.keys(sectionStates).forEach(sectionId => {
                if (!sectionStates[sectionId]) {
                    toggleSection(sectionId);
                }
            });
        }
        
        function collapseAllSections() {
            Object.keys(sectionStates).forEach(sectionId => {
                if (sectionStates[sectionId]) {
                    toggleSection(sectionId);
                }
            });
        }
        
        function toggleAllSections() {
            const allExpanded = Object.values(sectionStates).every(state => state);
            if (allExpanded) {
                collapseAllSections();
            } else {
                expandAllSections();
            }
        }
    </script>
    
    <script async defer
        src="https://maps.googleapis.com/maps/api/js?key={{google_api_key}}&callback=initMap">
    </script>
</body>
</html>
"""
    
    # Calculate statistics
    avg_wait_time_formatted = f"{avg_wait_time:.1f}" if avg_wait_time > 0 else "0"
    
    # Generate Analytics Section
    analytics_html = generate_analytics_section(static_charts) # Pass charts in
    
    # Generate other sections (existing code)
    pending_section = ""
    if pending_for_report:
        pending_html = f"""
        <div class="pending-orders">
            <h3>⚠️ {len(pending_for_report)} Pending Orders (End of Day)</h3>
            <p>The following orders could not be assigned to the standard fleet:</p>
        """
        for order in pending_for_report:
            order_loc = all_locations[order['index']]['original_address'].split(',')[0]
            pending_html += f"""
            <div class="pending-order">
                <strong>Order #{order['id']}</strong> (Demand: {order['demand']}) - {order_loc}
            </div>
            """
        pending_html += "</div>"
        pending_section = pending_html
    
    vehicle_cards_html = ""
    with state_lock:
        # Use the snapshot of routes from the start of the function
        for v_id in sorted(routes_for_report.keys()):
            route_orders = routes_for_report[v_id]
            if route_orders:
                route_cost = calculate_route_cost(route_orders, time_matrix)
                total_demand = sum(order['demand'] for order in route_orders)
                unique_stops = list(dict.fromkeys([order['index'] for order in route_orders]))
                
                stops_html = ""
                for stop_num, stop_index in enumerate(unique_stops, 1):
                    loc_name = all_locations[stop_index]['original_address'].split(',')[0]
                    stops_html += f"""
                    <div class="route-stop" 
                         onclick="highlightStop(this, {v_id}, {stop_num})">
                        {stop_num}. {loc_name}
                    </div>
                    """
                
                vehicle_cards_html += f"""
                <div class="vehicle-card">
                    <div class="vehicle-header">Vehicle {v_id}</div>
                    {stops_html}
                    <div class="route-metrics">
                        <div class="metric">
                            <div class="metric-value">{len(unique_stops)}</div>
                            <div class="metric-label">Stops</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{total_demand} / {VEHICLE_CAPACITY}</div>
                            <div class="metric-label">Capacity</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{route_cost:.0f}</div>
                            <div class="metric-label">Minutes</div>
                        </div>
                    </div>
                </div>
                """
            else:
                vehicle_cards_html += f"""
                <div class="vehicle-card">
                    <div class="vehicle-header">Vehicle {v_id}</div>
                    <p style="text-align: center; opacity: 0.7; padding: 20px;">No route assigned</p>
                </div>
                """
    
    timeline_html = ""
    for event in simulation_events:
        event_type = event['type']
        event_type_label = event_type.replace('_', ' ').title()
        
        if event_type == 'new_order':
            event_type_class = 'event-new-order'
        elif event_type == 'assignment':
            if event.get('success'):
                event_type_class = 'event-assignment'
                event_type_label = 'Assignment'
            else:
                event_type_class = 'event-rejected'
                event_type_label = 'Assignment Failed'
        elif event_type == 'optimization':
            event_type_class = 'event-optimization'
        elif event_type == 'premium':
            event_type_class = 'event-rejected'
            event_type_label = 'End of Day'
        else:
            event_type_class = 'event-optimization'
        
        timeline_html += f"""
        <div class="timeline-event">
            <div class="event-time">{event['time']}</div>
            <span class="event-type {event_type_class}">{event_type_label}</span>
            <div class="event-details">{event['description']}</div>
        </div>
        """
    
    historical_log_html = ""
    if global_order_assignments_log:
        historical_log_html = f"""
            <p>Shows every individual order assignment that occurred during the day. (Total: {len(global_order_assignments_log)})</p>
            <table class="log-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Order ID</th>
                        <th>Location</th>
                        <th>Demand</th>
                        <th>Assigned Vehicle</th>
                        <th>Method</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for assignment in global_order_assignments_log:
            historical_log_html += f"""
                    <tr>
                        <td><strong>{assignment['timestamp']}</strong></td>
                        <td>#{assignment['order_id']}</td>
                        <td>{assignment['location']}</td>
                        <td>{assignment['demand']}</td>
                        <td><strong>Vehicle {assignment['assigned_vehicle']}</strong></td>
                        <td>{assignment['method']}</td>
                    </tr>
            """
        
        historical_log_html += "</tbody></table>"
    else:
        historical_log_html = "<p>No historical trips were logged.</p>"
    
    # Fill in template
    html_content = html_template.replace('{{start_time}}', format_time(0))
    html_content = html_content.replace('{{end_time}}', format_time((SIMULATION_END_HOUR - SIMULATION_START_HOUR) * 60))
    html_content = html_content.replace('{{total_orders}}', str(total_orders))
    html_content = html_content.replace('{{assigned_orders}}', str(assigned_orders))
    html_content = html_content.replace('{{success_rate}}', f"{success_rate:.1f}")
    html_content = html_content.replace('{{pending_count}}', str(pending_count))
    html_content = html_content.replace('{{active_vehicles}}', str(active_vehicles))
    html_content = html_content.replace('{{total_vehicles}}', str(NUM_VEHICLES))
    html_content = html_content.replace('{{fleet_utilization}}', f"{fleet_utilization:.1f}")
    html_content = html_content.replace('{{optimization_count}}', str(optimization_count))
    html_content = html_content.replace('{{avg_duration}}', str(avg_duration))
    html_content = html_content.replace('{{avg_wait_time}}', avg_wait_time_formatted)
    html_content = html_content.replace('{{pending_orders_section}}', pending_section)
    html_content = html_content.replace('{{vehicle_cards}}', vehicle_cards_html)
    html_content = html_content.replace('{{timeline_events}}', timeline_html)
    html_content = html_content.replace('{{historical_log_section}}', historical_log_html)
    html_content = html_content.replace('{{analytics_section}}', analytics_html)
    html_content = html_content.replace('{{map_routes_json}}', map_routes_json)
    html_content = html_content.replace('{{time_matrix_json}}', time_matrix_json)
    
    # REMOVED JSON data bindings for charts
    
    html_content = html_content.replace('{{google_api_key}}', GOOGLE_MAPS_API_KEY)
    
    # Ensure 'outputs' directory exists
    os.makedirs('outputs', exist_ok=True)
    
    with open(OUTPUT_HTML_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ HTML Dashboard saved to '{OUTPUT_HTML_FILE}'")
    
    try:
        filepath = os.path.abspath(OUTPUT_HTML_FILE)
        webbrowser.open(f"file://{filepath}")
        print("📊 Dashboard opened in your default web browser.")
    except Exception as e:
        print(f"Could not auto-open dashboard. Please open '{OUTPUT_HTML_FILE}' manually.")

def generate_analytics_section(static_charts):
    """Generate the new analytics section with charts and insights"""
    
    # NEW: Cost vs. Optimized Cost Chart (now an <img>)
    if static_charts.get('cost_chart'):
        cost_chart_html = f"""
        <div class="chart-container">
            <div class="chart-title">💸 L1 Cost (Greedy) vs. Optimized Cost (L2/L3)</div>
            <img src="data:image/png;base64,{static_charts['cost_chart']}" class="chart-image" alt="Cost over Time Chart">
        </div>
        """
    else:
        cost_chart_html = "<div class='chart-container'><p>Cost chart data not available.</p></div>"


    # NEW: Wait Time Histogram (now an <img>)
    if static_charts.get('wait_chart'):
        wait_time_chart_html = f"""
        <div class="chart-container">
            <div class="chart-title">⏱️ Order Wait Time Distribution (Time from Order to L1 Assignment)</div>
            <img src="data:image/png;base64,{static_charts['wait_chart']}" class="chart-image" alt="Wait Time Histogram">
        </div>
        """
    else:
        wait_time_chart_html = "<div class='chart-container'><p>Wait time chart data not available.</p></div>"

    # NEW: Pending Orders Chart (now an <img>)
    if static_charts.get('pending_chart'):
        pending_orders_chart_html = f"""
        <div class="chart-container">
            <div class="chart-title">📈 Pending Orders Over Time (System Stress)</div>
            <img src="data:image/png;base64,{static_charts['pending_chart']}" class="chart-image" alt="Pending Orders Chart">
        </div>
        """
    else:
        pending_orders_chart_html = "<div class='chart-container'><p>Pending orders chart data not available.</p></div>"

    
    # Vehicle Utilization Heatmap
    heatmap_html = "<div class='chart-container'><div class='chart-title'>📊 Vehicle Capacity Utilization Heatmap</div>"
    heatmap_html += "<div class='heatmap-container'>"
    
    with state_lock:
        # Use a snapshot for analytics
        analytics_routes = {vid: r[:] for vid, r in current_routes.items()}
        analytics_pending = pending_orders[:]
        
        for v_id in sorted(analytics_routes.keys()):
            route_orders = analytics_routes[v_id]
            if route_orders:
                total_demand = sum(order['demand'] for order in route_orders)
                utilization_pct = (total_demand / VEHICLE_CAPACITY) * 100
                
                # Color based on utilization
                if utilization_pct >= 80:
                    color = '#f87171'  # Red - high utilization
                elif utilization_pct >= 50:
                    color = '#fbbf24'  # Yellow - medium
                else:
                    color = '#4ade80'  # Green - low
                
                heatmap_html += f"""
                <div class="heatmap-cell" style="background: {color};" 
                     title="Vehicle {v_id}: {utilization_pct:.1f}% utilized ({total_demand}/{VEHICLE_CAPACITY})">
                    V{v_id}<br>{utilization_pct:.0f}%
                </div>
                """
            else:
                heatmap_html += f"""
                <div class="heatmap-cell" style="background: #94a3b8;"
                     title="Vehicle {v_id}: Unused">
                    V{v_id}<br>0%
                </div>
                """
    
    heatmap_html += "</div>"
    heatmap_html += """
    <div class="heatmap-legend">
        <span>Utilization:</span>
        <div class="legend-gradient"></div>
        <span>Low (0%) → High (100%)</span>
    </div>
    </div>"""
    
    # Optimization Performance Comparison (MODIFIED)
    if optimization_performance_log:
        comparison_html = "<div class='chart-container'><div class='chart-title'>⚡ Optimization Layer Comparison</div>"
        comparison_html += "<table class='comparison-table'><thead>"
        comparison_html += """
            <tr>
                <th rowspan='2'>Cycle #</th>
                <th rowspan='2'>Time</th>
                <th colspan='4'>Layer 2 (OR-Tools)</th>
                <th colspan='4'>Layer 3 (ALNS)</th>
                <th rowspan='2'>Winner</th>
            </tr>
            <tr>
                <th>Cost</th>
                <th>Trucks</th>
                <th>Distance (km)</th>
                <th>Runtime (s)</th>
                <th>Cost</th>
                <th>Trucks</th>
                <th>Distance (km)</th>
                <th>Runtime (s)</th>
            </tr>
        """
        comparison_html += "</thead><tbody>"
        
        # Show all logs, or last 10 if too many
        logs_to_show = optimization_performance_log[-10:] if len(optimization_performance_log) > 10 else optimization_performance_log
        
        for idx, log_entry in enumerate(logs_to_show, 1):
            comparison_html += f"<tr>"
            comparison_html += f"<td><strong>#{idx}</strong></td>"
            comparison_html += f"<td>{log_entry['time']}</td>"
            # L2 Stats
            comparison_html += f"<td>{log_entry['l2_cost']:.2f}</td>"
            comparison_html += f"<td>{log_entry['l2_trucks']}</td>"
            comparison_html += f"<td>{log_entry['l2_distance']:.1f}</td>"
            comparison_html += f"<td>{log_entry['l2_runtime']:.2f}</td>"
            # L3 Stats
            comparison_html += f"<td>{log_entry['l3_cost']:.2f}</td>"
            comparison_html += f"<td>{log_entry['l3_trucks']}</td>"
            comparison_html += f"<td>{log_entry['l3_distance']:.1f}</td>"
            comparison_html += f"<td>{log_entry['l3_runtime']:.2f}</td>"
            # Winner
            comparison_html += f"<td>{log_entry['winner']} <span class='winner-badge'>✓</span></td>"
            comparison_html += f"</tr>"
        
        comparison_html += "</tbody></table></div>"
    else:
        comparison_html = "<div class='chart-container'><p>No optimization comparison data available.</p></div>"
    
    # Key Insights
    insights_html = "<div class='chart-container'><div class='chart-title'>💡 Key Insights</div>"
    insights_html += "<ul style='line-height: 2; font-size: 1.05em;'>"
    
    # Calculate insights
    with state_lock:
        # Use the same snapshot as above
        used_vehicles = len([r for r in analytics_routes.values() if r])
        unused_vehicles = NUM_VEHICLES - used_vehicles
        
        if unused_vehicles > 0:
            insights_html += f"<li>🚚 <strong>{unused_vehicles} vehicles</strong> remained unused - potential for fleet optimization</li>"
        
        overutilized = sum(1 for route in analytics_routes.values() if route and sum(o['demand'] for o in route) > VEHICLE_CAPACITY * 0.9)
        if overutilized > 0:
            insights_html += f"<li>⚠️ <strong>{overutilized} vehicles</strong> are operating near capacity (>90%) - consider load balancing</li>"
        
        underutilized = sum(1 for route in analytics_routes.values() if route and sum(o['demand'] for o in route) < VEHICLE_CAPACITY * 0.5)
        if underutilized > 0:
            insights_html += f"<li>📉 <strong>{underutilized} vehicles</strong> are underutilized (<50%) - consolidation opportunity</li>"
        
        if analytics_pending:
            insights_html += f"<li>❌ <strong>{len(analytics_pending)} orders</strong> could not be fulfilled - consider expanding fleet or capacity</li>"
        else:
            insights_html += "<li>✅ <strong>All orders</strong> successfully assigned to the standard fleet</li>"
        
        total_assignments = len(global_order_assignments_log)
        l1_assignments = len([log for log in global_order_assignments_log if log['method'] in ['greedy_insert', 'cheapest_insert']])
        if total_assignments > 0:
            l1_percentage = (l1_assignments / total_assignments) * 100
            insights_html += f"<li>⚡ <strong>{l1_percentage:.1f}%</strong> of orders assigned in real-time (Layer 1) - excellent responsiveness</li>"
    
    insights_html += "</ul></div>"
    
    # Order charts as requested
    return cost_chart_html + wait_time_chart_html + pending_orders_chart_html + heatmap_html + comparison_html + insights_html

# NEW: Refactored optimization logic into its own function
def run_optimization_cycle(cycle_name=""):
    """Runs one full optimization cycle (L2 vs L3)"""
    global current_routes, simulation_events, pending_orders, global_order_assignments_log, optimization_performance_log
    
    log_prefix = f"[{cycle_name}]" if cycle_name else ""
    
    with state_lock:
        routes_to_optimize = {vid: r[:] for vid, r in current_routes.items()}
        pending_to_optimize = pending_orders[:]
    
    if not any(routes_to_optimize.values()) and not pending_to_optimize:
        print(f"--- [OPTIMIZER WORKER]{log_prefix} No routes or pending orders. Skipping cycle. ---")
        return # Nothing to do

    print(f"\n{'='*10} [OPTIMIZER WORKER @ {datetime.now().strftime('%I:%M:%S %p')}] Starting {cycle_name} Optimization {'='*10}")
    print(f"Optimizing {sum(len(r) for r in routes_to_optimize.values())} assigned orders and {len(pending_to_optimize)} pending orders.")

    l2_results = {}
    l3_results = {}
    
    def run_layer2():
        print(f"--- [LAYER 2 OR-Tools]{log_prefix} Starting optimization... ---")
        start_time = time.time()
        try:
            opt_routes, unassigned = batch_optimization_vrp(
                current_routes=routes_to_optimize, pending_orders=pending_to_optimize, time_matrix=time_matrix,distance_matrix=distance_matrix,
                num_vehicles=NUM_VEHICLES, vehicle_capacity=VEHICLE_CAPACITY, max_route_duration_mins=MAX_ROUTE_DURATION_MINS,
                variable_cost_per_km=VARIABLE_COST_PER_KM, fixed_cost_per_truck=FIXED_COST_PER_TRUCK
            )
            l2_results['routes'] = opt_routes
            l2_results['unassigned'] = unassigned
            l2_results['error'] = None
        except Exception as e:
            print(f"--- [LAYER 2 OR-Tools]{log_prefix} Error during optimization: {e} ---")
            l2_results['error'] = e
        end_time = time.time()
        l2_results['runtime'] = end_time - start_time
        print(f"--- [LAYER 2 OR-Tools]{log_prefix} Finished in {l2_results['runtime']:.2f} seconds. Found {len(l2_results.get('unassigned',[]))} unassigned. ---")

    def run_layer3():
        print(f"--- [LAYER 3 ALNS]{log_prefix} Starting optimization... ---")
        start_time = time.time()
        try:
            opt_routes, unassigned = run_alns_optimization(
                current_routes_input=routes_to_optimize, pending_orders_input=pending_to_optimize, time_matrix=time_matrix, distance_matrix=distance_matrix,
                num_vehicles=NUM_VEHICLES, vehicle_capacity=VEHICLE_CAPACITY, max_route_duration_mins=MAX_ROUTE_DURATION_MINS , fixed_cost_per_truck=FIXED_COST_PER_TRUCK, variable_cost_per_km=VARIABLE_COST_PER_KM
            )
            l3_results['routes'] = opt_routes
            l3_results['unassigned'] = unassigned
            l3_results['error'] = None
        except Exception as e:
            print(f"--- [LAYER 3 ALNS]{log_prefix} Error during optimization: {e} ---")
            l3_results['error'] = e
        end_time = time.time()
        l3_results['runtime'] = end_time - start_time
        print(f"--- [LAYER 3 ALNS]{log_prefix} Finished in {l3_results['runtime']:.2f} seconds. Found {len(l3_results.get('unassigned',[]))} unassigned. ---")

    thread_l2 = threading.Thread(target=run_layer2)
    thread_l3 = threading.Thread(target=run_layer3)
    
    thread_l2.start()
    thread_l3.start()
    
    thread_l2.join()
    thread_l3.join()

    print(f"--- [OPTIMIZER WORKER]{log_prefix} Both layers completed. Comparing... ---")

    best_solution = None
    best_cost = float('inf')
    selected_layer = "None"
    
    cost_l2 = float('inf')
    trucks_l2 = 0
    dist_l2 = 0
    if l2_results.get('routes') is not None and l2_results.get('error') is None:
        cost_l2, trucks_l2, dist_l2 = calculate_total_fleet_cost(
            l2_results['routes'], distance_matrix,
            FIXED_COST_PER_TRUCK, VARIABLE_COST_PER_KM
        )
        print(f"    L2 (OR-Tools) Result{log_prefix}: Cost={cost_l2:.2f}, Trucks={trucks_l2}, Dist={dist_l2:.2f} km, Unassigned={len(l2_results['unassigned'])}, Time={l2_results['runtime']:.2f}s")
        if cost_l2 < best_cost:
            best_cost = cost_l2
            best_solution = l2_results
            selected_layer = "Layer 2 (OR-Tools)"
    else:
         print(f"    L2 (OR-Tools) Result{log_prefix}: Failed or produced no solution.")

    cost_l3 = float('inf')
    trucks_l3 = 0
    dist_l3 = 0
    if l3_results.get('routes') is not None and l3_results.get('error') is None:
        cost_l3, trucks_l3, dist_l3 = calculate_total_fleet_cost(
            l3_results['routes'], distance_matrix,
            FIXED_COST_PER_TRUCK, VARIABLE_COST_PER_KM
        )
        print(f"    L3 (ALNS) Result{log_prefix}: Cost={cost_l3:.2f}, Trucks={trucks_l3}, Dist={dist_l3:.2f} km, Unassigned={len(l3_results['unassigned'])}, Time={l3_results['runtime']:.2f}s")
        if cost_l3 < best_cost:
            best_cost = cost_l3
            best_solution = l3_results
            selected_layer = "Layer 3 (ALNS)"
        elif cost_l3 == best_cost and len(l3_results.get('unassigned', [])) < len(best_solution.get('unassigned', [])):
             best_solution = l3_results
             selected_layer = "Layer 3 (ALNS) - Tie Break on Unassigned"
    else:
         print(f"    L3 (ALNS) Result{log_prefix}: Failed or produced no solution.")

    # Log optimization performance for analytics (MODIFIED)
    if cost_l2 != float('inf') or cost_l3 != float('inf'): # Log even if one fails
        improvement = 0
        if cost_l2 != float('inf') and cost_l3 != float('inf'):
            improvement = ((max(cost_l2, cost_l3) - min(cost_l2, cost_l3)) / max(cost_l2, cost_l3)) * 100
        
        log_time_str = f"{datetime.now().strftime('%I:%M:%S %p')} {cycle_name}".strip()
        if "FINAL" in cycle_name:
            log_time_str = "Final"
            
        optimization_performance_log.append({
            'time': log_time_str,
            'l2_cost': cost_l2,
            'l2_trucks': trucks_l2,
            'l2_distance': dist_l2,
            'l2_runtime': l2_results.get('runtime', 0),
            'l3_cost': cost_l3,
            'l3_trucks': trucks_l3,
            'l3_distance': dist_l3,
            'l3_runtime': l3_results.get('runtime', 0),
            'winner': selected_layer.split(' ')[0], # 'L2' or 'L3'
            'improvement': improvement
        })

    if best_solution:
        print(f"--- [OPTIMIZER WORKER]{log_prefix} Selected solution from: {selected_layer} with Cost: {best_cost:.2f} ---")
        with state_lock:
            current_routes = best_solution['routes']
            pending_orders = best_solution['unassigned']
            
            # NEW: Log Optimized Cost
            sim_time_now = datetime.now()
            # Ensure simulation_start_time is set before this runs
            minutes_from_start = 0
            if simulation_start_time:
                 # Calculate minutes from simulation start, not real-world time
                 minutes_from_start = (simulation_start_time - simulation_start_time).total_seconds() / 60
                 # This logic is a bit flawed, let's tie it to sim minutes
                 # We'll use the *current* simulation minute if available, or just real time
                 
                 # Re-thinking: The optimizer runs in real-time, so logging
                 # vs real-time elapsed is correct.
                 minutes_from_start = (sim_time_now - simulation_start_time).total_seconds() / 60
            
            cost_over_time_log.append({
                'time': minutes_from_start,
                'type': 'optimization',
                'cost': best_cost # best_cost is already calculated
            })

            log_time = f"{datetime.now().strftime('%I:%M:%S %p')} {cycle_name}".strip()
            if "FINAL" in cycle_name:
                log_time = "Final"
                
            simulation_events.append({
                'type': 'optimization',
                'time': log_time,
                'description': f"[Parallel Opt.{cycle_name}] Selected {selected_layer}. New Cost={best_cost:.2f}. {len(pending_orders)} orders pending."
            })
    else:
        print(f"--- [OPTIMIZER WORKER]{log_prefix} Warning: Neither optimization layer produced a valid solution. State not updated. ---")
        
    print(f"{'='*10} [OPTIMIZER WORKER] {cycle_name} Cycle Finished {'='*10}")


def parallel_optimization_worker():
    """Background thread to run Layer 2 (OR-Tools) and Layer 3 (ALNS) in parallel"""
    
    while simulation_running:
        time.sleep(LAYER_2_INTERVAL_SECONDS)
        
        if not simulation_running:
            break # Exit loop if simulation stopped during sleep
            
        run_optimization_cycle("Periodic")
        
    # --- Simulation has ended, run ONE FINAL cycle ---
    print(f"\n{'='*10} [OPTIMIZER WORKER] Simulation ended. Running FINAL optimization cycle. {'='*10}")
    
    run_optimization_cycle("FINAL")

    # Finally, signal the main thread
    print(f"{'='*10} [OPTIMIZER WORKER] FINAL cycle complete. Signaling main thread. {'='*10}")
    optimizer_finished_event.set() # <<< THE SIGNAL


def run_hybrid_simulation():
    global current_routes, pending_orders, simulation_running, all_locations, time_matrix, simulation_events, simulation_start_time
    global distance_matrix, order_wait_times, pending_orders_history, cost_over_time_log
    print("--- Starting HYBRID DYNAMIC Delivery Simulation (Capacity Aware, Trace-Based) ---")
    
    try:
        with open(TIME_MATRIX_FILE, 'r') as f: 
            data = json.load(f)
        all_locations, time_matrix = data['locations'], data['time_matrix']
        distance_matrix = data['distance_matrix']
        print(f"✅ Master time matrix loaded ({len(time_matrix)}x{len(time_matrix)}).")
        print(f"✅ Master distance matrix loaded ({len(distance_matrix)}x{len(distance_matrix)}).")
        print(f"✅ {len(all_locations)} locations loaded.")
        
        if not all_locations or not time_matrix or not distance_matrix:
             raise ValueError("Loaded data is missing locations, time_matrix, or distance_matrix.")
        if len(time_matrix) != len(all_locations) or len(distance_matrix) != len(all_locations):
             raise ValueError(f"Matrix dimensions ({len(time_matrix)} / {len(distance_matrix)}) don't match location count ({len(all_locations)}).")
        
    except FileNotFoundError:
        print(f"FATAL Error: '{TIME_MATRIX_FILE}' not found. Run the 'build_and_save_matrix_custom.py' script first.")
        return
    except KeyError as e:
        print(f"FATAL Error: Missing key '{e}' in '{TIME_MATRIX_FILE}'. Ensure the file contains 'locations', 'time_matrix', and 'distance_matrix'.")
        return
    except ValueError as e:
         print(f"FATAL Error: Data validation failed - {e}")
         return
    except Exception as e:
        print(f"FATAL Error loading data from '{TIME_MATRIX_FILE}': {e}")
        return
        
    try:
        all_orders_df = pd.read_csv(PREPROCESSED_ORDER_FILE)
        sim_orders_df = all_orders_df[
            all_orders_df['day_of_year'] == SIMULATION_DAY_OF_YEAR
        ].sort_values(by='minute_of_day')
        
        historical_orders = sim_orders_df.to_dict('records')
        historical_orders.reverse()
        print(f"✅ Loaded {len(historical_orders)} orders for simulation day {SIMULATION_DAY_OF_YEAR}.")
        
    except FileNotFoundError:
        print(f"FATAL Error: '{PREPROCESSED_ORDER_FILE}' not found. Run 'preprocess_order_history.py' first.")
        return
    except Exception as e:
        print(f"Error loading orders: {e}")
        return

    # Reset shared state
    current_routes = {i: [] for i in range(NUM_VEHICLES)}
    pending_orders = []
    simulation_events = []
    order_wait_times = {}
    pending_orders_history = []
    cost_over_time_log = []
    global_order_assignments_log = []
    optimization_performance_log = []
    
    simulation_start_time = datetime.now() # Set start time for cost logging
    real_sim_start_time_for_logging = time.time()


    optimizer_thread = threading.Thread(target=parallel_optimization_worker, daemon=True)
    optimizer_thread.start()
    print(f"✅ Parallel Optimizer (L2/L3) thread started (Interval: {LAYER_2_INTERVAL_SECONDS}s).")
    print(f"✅ Simulating {NUM_VEHICLES} vehicles with {VEHICLE_CAPACITY} capacity each.")
    print(f"--- Simulating Day {SIMULATION_DAY_OF_YEAR} from {SIMULATION_START_HOUR}:00 to {SIMULATION_END_HOUR}:00 ---")

    for minute in range(SIMULATION_START_HOUR * 60, SIMULATION_END_HOUR * 60, MINUTES_PER_TICK):
        current_time_str = f"Day {SIMULATION_DAY_OF_YEAR}, {minute//60:02d}:{minute%60:02d}"
        print(f"\n{'='*15} {current_time_str} (Tick: {minute} - {minute + MINUTES_PER_TICK}) {'='*15}")
        
        orders_this_tick = 0
        while historical_orders and historical_orders[-1]['minute_of_day'] < (minute + MINUTES_PER_TICK):
            order_data = historical_orders.pop()
            orders_this_tick += 1
            
            try:
                location_idx = int(order_data['location_index'])
                if location_idx >= len(all_locations):
                    print(f"Warning: Skipping order with invalid location_index {location_idx}")
                    continue
            except ValueError:
                print(f"Warning: Skipping order with non-numeric location_index {order_data['location_index']}")
                continue

            new_order = {
                'id': order_data.get('order_id', f"ord_{order_data['timestamp']}"),
                'index': location_idx,
                'demand': int(order_data['demand']),
                'arrival_minute': minute  # NEW: Track arrival time
            }
            
            with state_lock:
                pending_orders.append(new_order)
            
            order_location = all_locations[new_order['index']]['original_address'].split(',')[0]
            print(f"EVENT: New historical order #{new_order['id']} received for {order_location} (Demand: {new_order['demand']})")
            
            simulation_events.append({
                'type': 'new_order',
                'time': format_time(minute - SIMULATION_START_HOUR * 60),
                'description': f"<strong>Order #{new_order['id']}</strong> (Demand: {new_order['demand']}) received for <strong>{order_location}</strong>"
            })

        if orders_this_tick > 0:
            print(f"Total {orders_this_tick} new orders this tick. Total pending: {len(pending_orders)}")
        
        if pending_orders:
            for order_to_assign in pending_orders[:]:
                if order_to_assign not in pending_orders:
                    continue
                    
                print(f"\n[LAYER 1] Attempting to assign Order #{order_to_assign['id']} (Demand: {order_to_assign['demand']})...")
                
                with state_lock:
                    final_routes, method = assign_new_order_realtime(
                        order_to_assign, 
                        current_routes, 
                        time_matrix,
                        VEHICLE_CAPACITY, 
                        MAX_ROUTE_DURATION_MINS
                    )

                if final_routes:
                    order_location = all_locations[order_to_assign['index']]['original_address'].split(',')[0]
                    print(f"SUCCESS (L1): Order #{order_to_assign['id']} assigned via {method} method.")
                    
                    assigned_vehicle_id = -1
                    newly_assigned_order_id = order_to_assign['id']
                    
                    for v_id, new_route in final_routes.items():
                        new_route_ids = [o['id'] for o in new_route]
                        if newly_assigned_order_id in new_route_ids:
                            assigned_vehicle_id = v_id
                            break

                    # NEW: Calculate wait time
                    wait_time = minute - order_to_assign['arrival_minute']
                    order_wait_times[order_to_assign['id']] = wait_time

                    with state_lock:
                        if assigned_vehicle_id != -1:
                            global_order_assignments_log.append({
                                "timestamp": format_time(minute - SIMULATION_START_HOUR * 60),
                                "order_id": order_to_assign['id'],
                                "demand": order_to_assign['demand'],
                                "location": order_location,
                                "assigned_vehicle": assigned_vehicle_id,
                                "method": method
                            })
                        
                        current_routes = final_routes
                        if order_to_assign in pending_orders:
                            pending_orders.remove(order_to_assign)
                        
                        # NEW: Log L1 cost
                        # We calculate cost based on the state *right after* L1 assignment
                        l1_cost, _, _ = calculate_total_fleet_cost(
                            current_routes, distance_matrix, FIXED_COST_PER_TRUCK, VARIABLE_COST_PER_KM
                        )
                        
                        # Log based on *simulation* minute, not real time
                        cost_over_time_log.append({
                            'time': minute - SIMULATION_START_HOUR * 60, # minutes from sim start
                            'type': 'l1_assignment',
                            'cost': l1_cost
                        })

                    
                    simulation_events.append({
                        'type': 'assignment',
                        'time': format_time(minute - SIMULATION_START_HOUR * 60),
                        'description': f"<span style='color:green;'>✓ ASSIGNED (L1)</span> Order #{order_to_assign['id']} via <strong>{method}</strong>. Dest: {order_location}. Wait: {wait_time} min",
                        'success': True
                    })
                else:
                    print(f"FAILURE (L1): Order #{order_to_assign['id']} could not be assigned. Awaiting Layer 2.")
                    simulation_events.append({
                        'type': 'assignment',
                        'time': format_time(minute - SIMULATION_START_HOUR * 60),
                        'description': f"<span style='color:red;'>✗ FAILED (L1)</span> Could not find immediate fit for Order #{order_to_assign['id']}.",
                        'success': False
                    })
        
        if not pending_orders:
            print("All pending orders assigned.")
        
        # NEW: Log pending orders history at end of tick
        with state_lock:
            pending_count = len(pending_orders)
        
        pending_orders_history.append({
            'time': minute - SIMULATION_START_HOUR * 60, # minutes from sim start
            'count': pending_count
        })
        
        # Log L1 cost *again* at the end of the tick, to get a clean "stepped" line
        with state_lock:
            l1_cost, _, _ = calculate_total_fleet_cost(
                current_routes, distance_matrix, FIXED_COST_PER_TRUCK, VARIABLE_COST_PER_KM
            )
        cost_over_time_log.append({
            'time': minute - SIMULATION_START_HOUR * 60, # minutes from sim start
            'type': 'l1_assignment',
            'cost': l1_cost
        })
            
        time.sleep(0.01) # Reduced sleep time, as sim is now faster

    simulation_running = False
    print("\n--- Dynamic Simulation Ended ---")
    print("Waiting for Parallel Optimizer thread to finish final cycle...")
    
    # --- THIS IS THE FIX ---
    # Wait for the optimizer's signal
    optimizer_finished_event.wait()
    print("...Optimizer's final cycle finished. Proceeding to summary.")
    # --- END OF FIX ---

    if pending_orders:
        print(f"\n--- {len(pending_orders)} orders remained unassigned at end of day ---")
        for order in pending_orders:
            order_location = all_locations[order['index']]['original_address'].split(',')[0]
            print(f"  - Order #{order['id']} (Demand: {order['demand']}) for {order_location}")
        
        simulation_events.append({
            'type': 'premium',
            'time': format_time((SIMULATION_END_HOUR - SIMULATION_START_HOUR) * 60),
            'description': f"<span style='color:orange;'>PREMIUM DEPLOYMENT</span> {len(pending_orders)} orders unassigned. Required premium fleet."
        })
    else:
        print("All orders were assigned to the standard fleet. ✅")

    print("\n" + "="*20 + " FINAL SIMULATION SUMMARY " + "="*20)
    
    total_orders_processed = len([e for e in simulation_events if e['type'] == 'new_order'])
    total_assignments_logged = len(global_order_assignments_log)
    
    # Read the final state *after* the optimizer has finished
    with state_lock:
        final_pending_count = len(pending_orders)
        final_routes_snapshot = current_routes
        
    print(f"Simulation Period: {SIMULATION_START_HOUR}:00 - {SIMULATION_END_HOUR}:00 (Day {SIMULATION_DAY_OF_YEAR})")
    print(f"Fleet Size: {NUM_VEHICLES} vehicles, Capacity: {VEHICLE_CAPACITY} units each")
    print("-" * 60)
    print(f"Total Orders Processed: {total_orders_processed}")
    print(f"Total Assignment Events Logged: {total_assignments_logged}") 
    print(f"Orders Pending at End of Day: {final_pending_count}")
    print("-" * 60)

    # Calculate final cost using the *optimized* routes
    final_cost, final_trucks_used, final_total_distance = calculate_total_fleet_cost(
        final_routes_snapshot, distance_matrix, FIXED_COST_PER_TRUCK, VARIABLE_COST_PER_KM
    )
    
    final_route_durations = [calculate_route_cost(r, time_matrix) for r in final_routes_snapshot.values() if r]
    avg_final_duration = sum(final_route_durations) / len(final_route_durations) if final_route_durations else 0

    print(f"Final Fleet State (Post-Optimization):")
    print(f"  - Vehicles Used: {final_trucks_used} out of {NUM_VEHICLES}")
    print(f"  - Total Distance Traveled: {final_total_distance:.2f} km")
    print(f"  - Average Route Duration (used vehicles): {avg_final_duration:.2f} min")
    print(f"  - Estimated Total Cost: {final_cost:.2f}")
    print("-" * 60)
    
    opt_events = [e for e in simulation_events if e['type'] == 'optimization']
    print(f"Parallel Optimization Cycles Triggered: {len(opt_events)}")

    print("=" * 60)
    
    print("\n--- Generating Enhanced HTML Dashboard (with final optimized data) ---")
    generate_html_report()

if __name__ == "__main__":
    # This assumes you have a 'hybrid_solver_layers.py' file in the same directory
    # with the required functions (assign_new_order_realtime, etc.)
    
    # Mocking the functions if the file doesn't exist, to allow this script to be analyzed
    # In a real run, you would remove this mock setup.
    try:
        # Check if functions are already imported
        assign_new_order_realtime
    except NameError:
        print("Warning: 'hybrid_solver_layers.py' not found or functions not imported.")
        print("Creating mock functions for testing purposes.")
        
        def mock_assign_new_order_realtime(order, routes, *args):
            # Mock: Assign to vehicle 0 if it has < 2 stops
            v_id_to_assign = order['id'] % len(routes)
            if len(routes[v_id_to_assign]) < 5: # Mock capacity check
                new_routes = {vid: r[:] for vid, r in routes.items()}
                new_routes[v_id_to_assign].append(order)
                return new_routes, "cheapest_insert"
            return None, None

        def mock_calculate_route_cost(route, *args):
            return len(route) * 15 # 15 min per stop
        
        def mock_calculate_route_distance(route, *args):
            return len(route) * 10 # 10 km per stop

        def mock_calculate_total_fleet_cost(routes, distance_matrix, fixed_cost, var_cost):
            cost = 0
            trucks = 0
            distance = 0
            for r in routes.values():
                if r:
                    trucks += 1
                    route_dist = mock_calculate_route_distance(r, distance_matrix)
                    cost += fixed_cost
                    cost += route_dist * var_cost
                    distance += route_dist
            return cost, trucks, distance

        def mock_batch_optimization_vrp(current_routes, pending_orders, *args, **kwargs):
            # Mock: Assign all pending to route 1, return current routes for others
            new_routes = {vid: r[:] for vid, r in current_routes.items()}
            all_orders = pending_orders[:]
            for r in new_routes.values():
                all_orders.extend(r)
            
            new_routes_out = {i: [] for i in range(kwargs['num_vehicles'])}
            unassigned = []
            
            v_idx = 0
            for order in all_orders:
                if sum(o['demand'] for o in new_routes_out[v_idx]) + order['demand'] > kwargs['vehicle_capacity']:
                    v_idx += 1
                
                if v_idx >= kwargs['num_vehicles']:
                    unassigned.append(order)
                else:
                    new_routes_out[v_idx].append(order)
            
            time.sleep(1.0) # Mock runtime
            return new_routes_out, unassigned

        def mock_run_alns_optimization(current_routes_input, pending_orders_input, *args, **kwargs):
            # Mock: Assign all pending to route 2
            new_routes = {vid: r[:] for vid, r in current_routes_input.items()}
            all_orders = pending_orders_input[:]
            for r in new_routes.values():
                all_orders.extend(r)
            
            new_routes_out = {i: [] for i in range(kwargs['num_vehicles'])}
            unassigned = []
            
            v_idx = 1 # Start on vehicle 1
            for order in all_orders:
                if sum(o['demand'] for o in new_routes_out[v_idx]) + order['demand'] > kwargs['vehicle_capacity']:
                    v_idx = (v_idx + 1) % kwargs['num_vehicles']
                
                new_routes_out[v_idx].append(order)

            time.sleep(0.8) # Mock runtime (faster)
            return new_routes_out, [] # Mock: ALNS is better, assigns all
        
        # Overwrite with mocks
        assign_new_order_realtime = mock_assign_new_order_realtime
        calculate_route_cost = mock_calculate_route_cost
        calculate_total_fleet_cost = mock_calculate_total_fleet_cost
        run_alns_optimization = mock_run_alns_optimization
        batch_optimization_vrp = mock_batch_optimization_vrp

    run_hybrid_simulation()