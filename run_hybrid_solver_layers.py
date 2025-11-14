import time
import random
import json
import threading
import pandas as pd
from datetime import datetime, timedelta
# We will import the new functions from our updated solver
from hybrid_solver_layers import (
    assign_new_order_realtime, 
    calculate_route_cost, 
    batch_optimization_vrp,
    calculate_total_fleet_cost,
    run_alns_optimization
)

random.seed(42)

# --- Configuration ---
SIMULATION_START_HOUR = 9  # 9:00 AM
SIMULATION_END_HOUR = 22 # 10:00 PM (to catch more orders)
MINUTES_PER_TICK = 10      # Check for new orders every 10 minutes
NUM_VEHICLES = 12
TIME_MATRIX_FILE = 'matrix_data_with_distance.json' # Or 'time_matrix_custom.json'

# --- NEW: Capacity & Real Order Data ---
VEHICLE_CAPACITY = 20         # Max number of "units" per vehicle
MAX_ROUTE_DURATION_MINS = 200 # Max drive time
PREPROCESSED_ORDER_FILE = 'preprocessed_orders.csv'
# --- Select the day you want to simulate ---
# (From your data, Sept 10 2024 is day 254)
SIMULATION_DAY_OF_YEAR = 254 

# --- Other Settings ---
LAYER_2_INTERVAL_SECONDS = 60 # Run background re-optimization every 60s
OUTPUT_HTML_FILE = 'outputs/hybrid_simulation_live_capacity.html'

# --- Cost Parameters ---
FIXED_COST_PER_TRUCK = 5000  # Example cost (e.g., rupees per day)
VARIABLE_COST_PER_KM = 15   # Example cost (e.g., rupees per km for fuel, maintenance)

# --- Distance Matrix (Crucial for Cost Calculation) ---
# We assume you have or can generate a distance matrix similar to your time matrix.
# If not, you'll need to calculate distances (e.g., using Haversine formula on lat/lon).
DISTANCE_MATRIX_FILE = 'distance_matrix.json' # Or load from the same file if combined
distance_matrix = [] # This will be loaded


# --- Shared State ---
# current_routes holds a list of full order objects for each vehicle
# { 0: [order_obj_1, order_obj_2], 1: [order_obj_3], ... }
current_routes = {} 
pending_orders = [] # [order_obj_4, order_obj_5, ...]
state_lock = threading.Lock()
simulation_running = True
simulation_events = []
all_locations = []
time_matrix = []
global_order_assignments_log = [] # <--- ADD THIS NEW LOG
simulation_start_time = None # For real-time sync

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
    lat_key, lng_key = ('latitude', 'longitude') # From geocoded_locations
    if 'lat' in sample_loc: # From other sources
        lat_key, lng_key = 'lat', 'lng'
    
    # Add depot at start
    coords.append({
        'lat': locations[0][lat_key], 'lng': locations[0][lng_key],
        'address': locations[0]['original_address'], 'type': 'depot', 'index': 0
    })
    
    # Get unique stops from the list of orders
    stop_indices = []
    if route_orders:
        stop_indices = list(dict.fromkeys([order['index'] for order in route_orders]))

    # Add all stops
    for idx in stop_indices:
        if idx < len(locations):
            coords.append({
                'lat': locations[idx][lat_key], 'lng': locations[idx][lng_key],
                'address': locations[idx]['original_address'], 'type': 'stop', 'index': idx
            })
        else:
            print(f"Warning: Index {idx} out of bounds for locations.")
            
    # Add depot at end
    coords.append({
        'lat': locations[0][lat_key], 'lng': locations[0][lng_key],
        'address': locations[0]['original_address'], 'type': 'depot', 'index': 0
    })
    return coords

# --- HTML Generation Function ---
# Import these at the top of your file if they aren't already
import webbrowser
import os

# --- NEW: Add your Google Maps API Key ---
# The map will not work without this.
# This was in your old file but missing from the new one.
GOOGLE_MAPS_API_KEY = 'AIzaSyC_hI6BowrJPojeBiRldmuFVf3aqsSRZbg' # Replace with your key

def generate_html_report():
    """Generate comprehensive HTML report of the simulation"""
    
    # --- 1. Prepare Map Route Data ---
    # This part is now adapted to read routes as lists of order objects
    map_routes = []
    try:
        with state_lock:
            for v_id in sorted(current_routes.keys()):
                route_orders = current_routes[v_id] # This is now [order_obj_1, ...]
                if route_orders:
                    # generate_route_coordinates is already updated
                    # to handle a list of order objects
                    coords = generate_route_coordinates(route_orders, all_locations)
                    map_routes.append({
                        'vehicle_id': v_id,
                        'coordinates': coords,
                        'color': ['#667eea', '#28a745', '#ffc107', '#dc3545'][v_id % 4]
                    })
    except Exception as e:
        print(f"Warning: Could not generate map routes: {e}")
        print(f"Sample location data: {all_locations[0] if all_locations else 'No locations'}")
        map_routes = []
    
    # --- 2. Serialize Data for JavaScript ---
    map_routes_json = json.dumps(map_routes)
    time_matrix_json = json.dumps(time_matrix)
    
    # --- 3. Define the HTML Template ---
    # This template is updated to show 3 metrics per vehicle
    # and includes all necessary CSS/JS from before.
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hybrid Delivery Simulation - Live Dashboard (Capacity Aware)</title>
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
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
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
        .content { padding: 30px; }
        .section { margin-bottom: 40px; }
        .section-title {
            font-size: 1.8em; color: #667eea; margin-bottom: 20px;
            padding-bottom: 10px; border-bottom: 3px solid #667eea;
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
            /* UPDATED: 3 columns for new metrics */
            grid-template-columns: repeat(3, 1fr);
            gap: 10px; margin-top: 15px;
        }
        .metric {
            background: rgba(255,255,255,0.2); padding: 10px;
            border-radius: 6px; text-align: center;
        }
        .metric-value { font-size: 1.3em; font-weight: bold; } /* Slightly smaller for 3 cols */
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

        /* --- NEW STYLES FOR ASSIGNMENT LOG --- */
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

        /* --- COLLAPSIBLE SECTIONS STYLES --- */
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
        
        .collapsible-content.expanded {
            max-height: 2000px;
            padding: 25px;
        }
        
        .collapsible-content.collapsed {
            max-height: 0;
            padding: 0 25px;
        }
        
        /* Central Controls */
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
        
        /* Section-specific styling */
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
        
        /* Responsive design */
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
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚚 Hybrid Delivery Simulation Dashboard</h1>
            <p>Real-time Vehicle Routing with Capacity Constraints (Trace-Based)</p>
            <p><strong>Simulation Period:</strong> {{start_time}} - {{end_time}}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Orders</div>
                <div class="stat-value">{{total_orders}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Orders Assigned</div>
                <div class="stat-value">{{assigned_orders}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Pending (End of Day)</div>
                <div class="stat-value">{{pending_count}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Active Vehicles (Final)</div>
                <div class="stat-value">{{active_vehicles}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Layer 2 Optimizations</div>
                <div class="stat-value">{{optimization_count}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Final Route</div>
                <div class="stat-value">{{avg_duration}} min</div>
            </div>
        </div>
        
        <div class="content">
            {{pending_orders_section}}
            
            <!-- Central Controls -->
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
            
            <div class="section">
                <h2 class="section-title">🗺️ Interactive Route Map (Final State at 10:00 PM)</h2>
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
                    <h4>📍 Route Segment Information</h4>
                    <p id="segmentDescription">Select two consecutive stops on a route to see travel details.</p>
                    <div class="distance-details" id="distanceDetails"></div>
                </div>
            </div>

            <!-- Historical Order Assignment Log Section -->
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

            <!-- Final Fleet Status Section -->
            <div class="collapsible-section" id="fleet-status-section">
                <div class="collapsible-header" onclick="toggleSection('fleet-status-section')">
                    <h3 class="collapsible-title">
                        <span class="section-icon">🚚</span>
                        Final Fleet Status (Snapshot at 10:00 PM)
                    </h3>
                    <span class="collapsible-icon">▼</span>
                </div>
                <div class="collapsible-content fleet-status-content" id="fleet-status-content">
                    <div class="vehicle-grid">
                        {{vehicle_cards}}
                    </div>
                </div>
            </div>
            
            {{premium_section}} 
            
            <!-- Simulation Event Timeline Section -->
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
        // Route data from Python
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
        
        // Collapsible sections state
        let sectionStates = {
            'historical-log-section': false,
            'fleet-status-section': false,
            'timeline-section': false
        };
        
        function initMap() {
            const depotLocation = routesData.length > 0 && routesData[0].coordinates.length > 0
                ? { lat: routesData[0].coordinates[0].lat, lng: routesData[0].coordinates[0].lng }
                : { lat: 22.7196, lng: 75.8577 }; // Default
            
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
            const R = 6371; // km
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
        
        // Collapsible sections functionality
        function toggleSection(sectionId) {
            const section = document.getElementById(sectionId);
            const content = document.getElementById(sectionId.replace('-section', '-content'));
            const icon = section.querySelector('.collapsible-icon');
            const isExpanded = sectionStates[sectionId];
            
            if (isExpanded) {
                // Collapse
                content.classList.remove('expanded');
                content.classList.add('collapsed');
                icon.classList.remove('expanded');
                sectionStates[sectionId] = false;
            } else {
                // Expand
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
        
        // Initialize sections as collapsed on page load
        document.addEventListener('DOMContentLoaded', function() {
            Object.keys(sectionStates).forEach(sectionId => {
                const content = document.getElementById(sectionId.replace('-section', '-content'));
                if (content) {
                    content.classList.add('collapsed');
                }
            });
        });
    </script>
    
    <script async defer
        src="https://maps.googleapis.com/maps/api/js?key={{google_api_key}}&callback=initMap">
    </script>
</body>
</html>
"""
    
    # --- 4. Calculate Final Statistics ---
    total_orders = len([e for e in simulation_events if e['type'] == 'new_order'])
    assigned_orders = len([e for e in simulation_events if e['type'] == 'assignment' and e.get('success')])
    optimization_count = len([e for e in simulation_events if e['type'] == 'optimization'])
    pending_count = len(pending_orders)
    
    with state_lock:
        active_vehicles = len([r for r in current_routes.values() if r])
        avg_duration = 0
        if active_vehicles > 0:
            # Pass the list of order objects directly to the new cost function
            total_duration = sum([calculate_route_cost(r, time_matrix) for r in current_routes.values() if r])
            avg_duration = int(total_duration / active_vehicles) if total_duration > 0 else 0
    
    # --- 5. Generate Dynamic HTML Snippets ---
    
    # A. Pending Orders Section (Capacity-Aware)
    pending_section = ""
    if pending_orders:
        pending_html = f"""
        <div class="pending-orders">
            <h3>⚠️ {len(pending_orders)} Pending Orders (End of Day)</h3>
            <p>The following orders could not be assigned to the standard fleet:</p>
        """
        # pending_orders now contains order objects
        for order in pending_orders:
            order_loc = all_locations[order['index']]['original_address'].split(',')[0]
            pending_html += f"""
            <div class="pending-order">
                <strong>Order #{order['id']}</strong> (Demand: {order['demand']}) - {order_loc}
            </div>
            """
        pending_html += "</div>"
        pending_section = pending_html
    
    # B. Vehicle Cards Section (Capacity-Aware)
    vehicle_cards_html = ""
    with state_lock:
        for v_id in sorted(current_routes.keys()):
            route_orders = current_routes[v_id] # List of order objects
            if route_orders:
                # Pass list of order objects to cost function
                route_cost = calculate_route_cost(route_orders, time_matrix)
                
                # NEW: Calculate total demand and unique stops
                total_demand = sum(order['demand'] for order in route_orders)
                unique_stops = list(dict.fromkeys([order['index'] for order in route_orders]))
                
                stops_html = ""
                # Iterate over unique stops to build the list
                # This matches what generate_route_coordinates does
                for stop_num, stop_index in enumerate(unique_stops, 1):
                    loc_name = all_locations[stop_index]['original_address'].split(',')[0]
                    # stop_num is the 1-based index (1, 2, 3...)
                    # This is what highlightStop JS function expects as 'routeIndex'
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
    
    # C. Premium Section (Simplified)
    # The new script just logs a 'premium' event, it doesn't solve
    # a separate route. This section is no longer needed as
    # 'pending_orders_section' and the timeline cover it.
    premium_section = "" 
    
    # D. Timeline Section (Adapted for new event types)
    timeline_html = ""
    for event in simulation_events:
        event_type = event['type']
        event_type_label = event_type.replace('_', ' ').title()
        
        # Assign CSS class based on event type
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
            event_type_class = 'event-rejected' # Use warning/rejected color
            event_type_label = 'End of Day'
        else:
            event_type_class = 'event-optimization' # Default
        
        # The 'description' field now contains its own HTML styling
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
        
        # Use the new log
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
    # --- 6. Fill in Template ---
    html_content = html_template.replace('{{start_time}}', format_time(0))
    html_content = html_content.replace('{{end_time}}', format_time((SIMULATION_END_HOUR - SIMULATION_START_HOUR) * 60))
    html_content = html_content.replace('{{total_orders}}', str(total_orders))
    html_content = html_content.replace('{{assigned_orders}}', str(assigned_orders))
    html_content = html_content.replace('{{pending_count}}', str(pending_count))
    html_content = html_content.replace('{{active_vehicles}}', str(active_vehicles))
    html_content = html_content.replace('{{optimization_count}}', str(optimization_count))
    html_content = html_content.replace('{{avg_duration}}', str(avg_duration))
    html_content = html_content.replace('{{pending_orders_section}}', pending_section)
    html_content = html_content.replace('{{vehicle_cards}}', vehicle_cards_html)
    html_content = html_content.replace('{{premium_section}}', premium_section)
    html_content = html_content.replace('{{timeline_events}}', timeline_html)

    # --- ADD THIS NEW LINE ---
    html_content = html_content.replace('{{historical_log_section}}', historical_log_html)


    html_content = html_content.replace('{{map_routes_json}}', map_routes_json)
    html_content = html_content.replace('{{time_matrix_json}}', time_matrix_json)
    html_content = html_content.replace('{{google_api_key}}', GOOGLE_MAPS_API_KEY)
    
    # --- 7. Save and Open File ---
    with open(OUTPUT_HTML_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ HTML Dashboard saved to '{OUTPUT_HTML_FILE}'")
    
    try:
        filepath = os.path.abspath(OUTPUT_HTML_FILE)
        webbrowser.open(f"file://{filepath}")
        print("📊 Dashboard opened in your default web browser.")
    except Exception as e:
        print(f"Could not auto-open dashboard. Please open '{OUTPUT_HTML_FILE}' manually.")

# --- Parallel Optimization Worker Thread ---
def parallel_optimization_worker():
    """
    Background thread to run Layer 2 (OR-Tools) and Layer 3 (ALNS)
    in parallel periodically and select the best solution based on total cost.
    """
    global current_routes, simulation_events, pending_orders, global_order_assignments_log
    
    while simulation_running:
        time.sleep(LAYER_2_INTERVAL_SECONDS)
        
        # --- Get current state ---
        with state_lock:
            routes_to_optimize = {vid: r[:] for vid, r in current_routes.items()}
            pending_to_optimize = pending_orders[:]
        
        if not any(routes_to_optimize.values()) and not pending_to_optimize:
            # print("--- [OPTIMIZER WORKER] No routes or pending orders to optimize. Skipping cycle. ---")
            continue

        print(f"\n{'='*10} [OPTIMIZER WORKER @ {datetime.now().strftime('%I:%M:%S %p')}] Starting Parallel Optimization {'='*10}")
        print(f"Optimizing {sum(len(r) for r in routes_to_optimize.values())} assigned orders and {len(pending_to_optimize)} pending orders.")

        # --- Setup threads ---
        l2_results = {}
        l3_results = {}
        
        # Target function for Layer 2 (OR-Tools) thread
        def run_layer2():
            print("--- [LAYER 2 OR-Tools] Starting optimization... ---")
            start_time = time.time()
            try:
                # Use the function from hybrid_solver_new
                opt_routes, unassigned = batch_optimization_vrp(
                    routes_to_optimize, pending_to_optimize, time_matrix,
                    NUM_VEHICLES, VEHICLE_CAPACITY, MAX_ROUTE_DURATION_MINS
                )
                l2_results['routes'] = opt_routes
                l2_results['unassigned'] = unassigned
                l2_results['error'] = None
            except Exception as e:
                print(f"--- [LAYER 2 OR-Tools] Error during optimization: {e} ---")
                l2_results['error'] = e
            end_time = time.time()
            l2_results['runtime'] = end_time - start_time
            print(f"--- [LAYER 2 OR-Tools] Finished in {l2_results['runtime']:.2f} seconds. Found {len(l2_results.get('unassigned',[]))} unassigned. ---")

        # Target function for Layer 3 (ALNS) thread
        def run_layer3():
            print("--- [LAYER 3 ALNS] Starting optimization... ---")
            start_time = time.time()
            try:
                # Use the function from hybrid_solver_new
                opt_routes, unassigned = run_alns_optimization( # Using the placeholder
                    routes_to_optimize, pending_to_optimize, time_matrix, distance_matrix, # Pass distance_matrix
                    NUM_VEHICLES, VEHICLE_CAPACITY, MAX_ROUTE_DURATION_MINS
                    # Add your ALNS params here if needed
                )
                l3_results['routes'] = opt_routes
                l3_results['unassigned'] = unassigned
                l3_results['error'] = None
            except Exception as e:
                print(f"--- [LAYER 3 ALNS] Error during optimization: {e} ---")
                l3_results['error'] = e
            end_time = time.time()
            l3_results['runtime'] = end_time - start_time
            print(f"--- [LAYER 3 ALNS] Finished in {l3_results['runtime']:.2f} seconds. Found {len(l3_results.get('unassigned',[]))} unassigned. ---")

        # --- Run threads in parallel ---
        thread_l2 = threading.Thread(target=run_layer2)
        thread_l3 = threading.Thread(target=run_layer3)
        
        thread_l2.start()
        thread_l3.start()
        
        thread_l2.join() # Wait for L2 to finish
        thread_l3.join() # Wait for L3 to finish

        print("--- [OPTIMIZER WORKER] Both optimization layers completed. Comparing results... ---")

        # --- Compare results and select the best ---
        best_solution = None
        best_cost = float('inf')
        selected_layer = "None"
        
        # Calculate cost for Layer 2 solution
        cost_l2 = float('inf')
        trucks_l2 = 0
        dist_l2 = 0
        if l2_results.get('routes') is not None and l2_results.get('error') is None:
            cost_l2, trucks_l2, dist_l2 = calculate_total_fleet_cost(
                l2_results['routes'], distance_matrix,
                FIXED_COST_PER_TRUCK, VARIABLE_COST_PER_KM
            )
            print(f"    L2 (OR-Tools) Result: Cost={cost_l2:.2f}, Trucks={trucks_l2}, Dist={dist_l2:.2f} km, Unassigned={len(l2_results['unassigned'])}, Time={l2_results['runtime']:.2f}s")
            if cost_l2 < best_cost:
                best_cost = cost_l2
                best_solution = l2_results
                selected_layer = "Layer 2 (OR-Tools)"
        else:
             print(f"    L2 (OR-Tools) Result: Failed or produced no solution.")

        # Calculate cost for Layer 3 solution
        cost_l3 = float('inf')
        trucks_l3 = 0
        dist_l3 = 0
        if l3_results.get('routes') is not None and l3_results.get('error') is None:
            cost_l3, trucks_l3, dist_l3 = calculate_total_fleet_cost(
                l3_results['routes'], distance_matrix,
                FIXED_COST_PER_TRUCK, VARIABLE_COST_PER_KM
            )
            print(f"    L3 (ALNS) Result: Cost={cost_l3:.2f}, Trucks={trucks_l3}, Dist={dist_l3:.2f} km, Unassigned={len(l3_results['unassigned'])}, Time={l3_results['runtime']:.2f}s")
             # --- Selection Logic ---
             # Choose L3 if its cost is strictly better than L2's best cost so far
            if cost_l3 < best_cost:
                best_cost = cost_l3
                best_solution = l3_results
                selected_layer = "Layer 3 (ALNS)"
            # Optional Tie-breaking (e.g., prefer fewer unassigned if costs are equal)
            elif cost_l3 == best_cost and len(l3_results['unassigned']) < len(best_solution['unassigned']):
                 best_solution = l3_results
                 selected_layer = "Layer 3 (ALNS) - Tie Break on Unassigned"

        else:
             print(f"    L3 (ALNS) Result: Failed or produced no solution.")


        # --- Update global state with the best solution found ---
        if best_solution:
            print(f"--- [OPTIMIZER WORKER] Selected solution from: {selected_layer} with Cost: {best_cost:.2f} ---")
            with state_lock:
                current_routes = best_solution['routes']
                pending_orders = best_solution['unassigned']
                
                # Log the optimization event (maybe add which layer won)
                log_time = datetime.now().strftime("%I:%M:%S %p")
                simulation_events.append({
                    'type': 'optimization',
                    'time': log_time,
                    'description': f"[Parallel Opt.] Selected {selected_layer}. New Cost={best_cost:.2f}. {len(pending_orders)} orders pending."
                })
        else:
            print("--- [OPTIMIZER WORKER] Warning: Neither optimization layer produced a valid solution. State not updated. ---")
            
        print(f"{'='*10} [OPTIMIZER WORKER] Cycle Finished {'='*10}")
            
# --- Main Simulation Function ---
def run_hybrid_simulation():
    global current_routes, pending_orders, simulation_running, all_locations, time_matrix, simulation_events, simulation_start_time
    global distance_matrix
    print("--- Starting HYBRID DYNAMIC Delivery Simulation (Capacity Aware, Trace-Based) ---")
    
    # 1. Load the "Map" (Time Matrix & Locations)
    try:
        with open(TIME_MATRIX_FILE, 'r') as f: 
            data = json.load(f)
        all_locations, time_matrix = data['locations'], data['time_matrix']
        distance_matrix = data['distance_matrix']
        print(f"✅ Master time matrix loaded ({len(time_matrix)}x{len(time_matrix)}).")
        print(f"✅ Master distance matrix loaded ({len(distance_matrix)}x{len(distance_matrix)}).")
        print(f"✅ {len(all_locations)} locations loaded.")
        
        # Basic validation (Check if matrices match location count)
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
        
    # 2. Load the "Event Script" (Historical Orders)
    try:
        all_orders_df = pd.read_csv(PREPROCESSED_ORDER_FILE)
        # Filter for only the day we want to simulate
        sim_orders_df = all_orders_df[
            all_orders_df['day_of_year'] == SIMULATION_DAY_OF_YEAR
        ].sort_values(by='minute_of_day')
        
        historical_orders = sim_orders_df.to_dict('records')
        historical_orders.reverse() # Reverse so we can .pop() from the end
        print(f"✅ Loaded {len(historical_orders)} orders for simulation day {SIMULATION_DAY_OF_YEAR}.")
        
    except FileNotFoundError:
        print(f"FATAL Error: '{PREPROCESSED_ORDER_FILE}' not found. Run 'preprocess_order_history.py' first.")
        return
    except Exception as e:
        print(f"Error loading orders: {e}")
        return

    # 3. Initialize Simulation State
    current_routes = {i: [] for i in range(NUM_VEHICLES)}
    pending_orders = []
    simulation_events = []
    simulation_start_time = datetime.now() # Real-world start time

    # 4. Start Layer 2 Background Thread
    # layer2_thread = threading.Thread(target=layer2_worker, daemon=True)
    # layer2_thread.start()
    optimizer_thread = threading.Thread(target=parallel_optimization_worker, daemon=True) # New line
    optimizer_thread.start()
    # print(f"✅ Layer 2 background optimization thread started (Interval: {LAYER_2_INTERVAL_SECONDS}s).")
    print(f"✅ Parallel Optimizer (L2/L3) thread started (Interval: {LAYER_2_INTERVAL_SECONDS}s).") # New log message
    print(f"✅ Simulating {NUM_VEHICLES} vehicles with {VEHICLE_CAPACITY} capacity each.")
    print(f"--- Simulating Day {SIMULATION_DAY_OF_YEAR} from {SIMULATION_START_HOUR}:00 to {SIMULATION_END_HOUR}:00 ---")

    # --- Main Simulation Loop (Layer 1) ---
    for minute in range(SIMULATION_START_HOUR * 60, SIMULATION_END_HOUR * 60, MINUTES_PER_TICK):
        current_time_str = f"Day {SIMULATION_DAY_OF_YEAR}, {minute//60:02d}:{minute%60:02d}"
        print(f"\n{'='*15} {current_time_str} (Tick: {minute} - {minute + MINUTES_PER_TICK}) {'='*15}")
        
        # --- A: Check for new orders from our REAL data ---
        orders_this_tick = 0
        while historical_orders and historical_orders[-1]['minute_of_day'] < (minute + MINUTES_PER_TICK):
            order_data = historical_orders.pop()
            orders_this_tick += 1
            
            # Ensure index is an integer and valid
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
                'demand': int(order_data['demand'])
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
        
        # --- B: Try to assign pending orders (Layer 1) ---
        if pending_orders:
            # We iterate on a copy because we'll be removing items
            for order_to_assign in pending_orders[:]:
                if order_to_assign not in pending_orders:
                    continue # Was assigned by L2 in the background
                    
                print(f"\n[LAYER 1] Attempting to assign Order #{order_to_assign['id']} (Demand: {order_to_assign['demand']})...")
                routes_before_assignment = {}
                # with state_lock:
                #     # Get a deep copy of the routes *before* the solver runs
                #     routes_before_assignment = {vid: r[:] for vid, r in current_routes.items()}
                with state_lock:
                    # Pass the full order object and capacity constraints
                    final_routes, method = assign_new_order_realtime(
                        order_to_assign, 
                        current_routes, 
                        time_matrix,
                        VEHICLE_CAPACITY, 
                        MAX_ROUTE_DURATION_MINS
                    )

                if final_routes: # If assignment was successful
                    order_location = all_locations[order_to_assign['index']]['original_address'].split(',')[0]
                    print(f"SUCCESS (L1): Order #{order_to_assign['id']} assigned via {method} method.")
                    
                    # --- NEW LOGIC: FIND WHICH VEHICLE CHANGED ---
                    assigned_vehicle_id = -1
                    newly_assigned_order_id = order_to_assign['id']
                    
                    for v_id, new_route in final_routes.items():
                        new_route_ids = [o['id'] for o in new_route]
                        if newly_assigned_order_id in new_route_ids:
                            assigned_vehicle_id = v_id
                            break

                    with state_lock:
                        # --- ADD TO NEW ASSIGNMENT LOG ---
                        if assigned_vehicle_id != -1:
                            global_order_assignments_log.append({
                                "timestamp": format_time(minute - SIMULATION_START_HOUR * 60),
                                "order_id": order_to_assign['id'],
                                "demand": order_to_assign['demand'],
                                "location": order_location,
                                "assigned_vehicle": assigned_vehicle_id,
                                "method": method
                            })
                        
                        # Set the new routes
                        current_routes = final_routes
                        if order_to_assign in pending_orders:
                            pending_orders.remove(order_to_assign) # Remove from pending
                    
                    simulation_events.append({
                        'type': 'assignment',
                        'time': format_time(minute - SIMULATION_START_HOUR * 60),
                        'description': f"<span style='color:green;'>✓ ASSIGNED (L1)</span> Order #{order_to_assign['id']} via <strong>{method}</strong>. Dest: {order_location}",
                        'success': True
                    })
                else:
                    # If final_routes is None, assignment failed
                    print(f"FAILURE (L1): Order #{order_to_assign['id']} could not be assigned. Awaiting Layer 2.")
                    simulation_events.append({
                        'type': 'assignment',
                        'time': format_time(minute - SIMULATION_START_HOUR * 60),
                        'description': f"<span style='color:red;'>✗ FAILED (L1)</span> Could not find immediate fit for Order #{order_to_assign['id']}.",
                        'success': False
                    })
        
        if not pending_orders:
            print("All pending orders assigned.")
            
        # Control simulation speed
        time.sleep(0.1) # Faster simulation

    # --- End of Day Processing ---
    simulation_running = False
    print("\n--- Dynamic Simulation Ended ---")
    print("Waiting for Layer 2 thread to finish...")
    time.sleep(LAYER_2_INTERVAL_SECONDS + 2) # Wait for L2 thread to stop
    
    # Final check for any remaining pending orders
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
    
    # --- End of Day Processing ---
    simulation_running = False
    print("\n--- Dynamic Simulation Ended ---")
    print("Waiting for Parallel Optimizer thread to finish last cycle...")
    time.sleep(LAYER_2_INTERVAL_SECONDS + 5) # Wait a bit longer for potentially slower ALNS

    # --- FINAL SIMULATION SUMMARY ---
    print("\n" + "="*20 + " FINAL SIMULATION SUMMARY " + "="*20)
    
    total_orders_processed = len([e for e in simulation_events if e['type'] == 'new_order'])
    # Count assignments differently now from the assignment log
    total_assignments_logged = len(global_order_assignments_log)
    final_pending_count = len(pending_orders)
    
    print(f"Simulation Period: {SIMULATION_START_HOUR}:00 - {SIMULATION_END_HOUR}:00 (Day {SIMULATION_DAY_OF_YEAR})")
    print(f"Fleet Size: {NUM_VEHICLES} vehicles, Capacity: {VEHICLE_CAPACITY} units each")
    print("-" * 60)
    print(f"Total Orders Processed: {total_orders_processed}")
    # Note: L1 might assign, then L2/L3 reassigns. This log counts each assignment event.
    print(f"Total Assignment Events Logged: {total_assignments_logged}") 
    print(f"Orders Pending at End of Day: {final_pending_count}")
    print("-" * 60)

    # Calculate final state metrics
    final_cost, final_trucks_used, final_total_distance = calculate_total_fleet_cost(
        current_routes, distance_matrix, FIXED_COST_PER_TRUCK, VARIABLE_COST_PER_KM
    )
    
    # Calculate final average duration ONLY for routes used
    final_route_durations = [calculate_route_cost(r, time_matrix) for r in current_routes.values() if r]
    avg_final_duration = sum(final_route_durations) / len(final_route_durations) if final_route_durations else 0

    print(f"Final Fleet State:")
    print(f"  - Vehicles Used: {final_trucks_used} out of {NUM_VEHICLES}")
    print(f"  - Total Distance Traveled: {final_total_distance:.2f} km")
    print(f"  - Average Route Duration (used vehicles): {avg_final_duration:.2f} min")
    print(f"  - Estimated Total Cost: {final_cost:.2f}")
    print("-" * 60)
    
    # Count optimization runs
    opt_events = [e for e in simulation_events if e['type'] == 'optimization']
    print(f"Parallel Optimization Cycles Triggered: {len(opt_events)}")
    # Could add more detail here later by parsing description if needed

    print("=" * 60)
    
    # Generate final HTML report (as before)
    print("\n--- Generating HTML Dashboard ---")
    generate_html_report()
if __name__ == "__main__":
    run_hybrid_simulation()

