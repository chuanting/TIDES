import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers

    return c * r


def calculate_distances_and_connectivity(base_stations):
    """
    Calculate distances and connectivity between all pairs of base stations
    Returns two dictionaries with (id1, id2) as keys and distances/connectivity as values
    """
    distances = {}
    connectivity = {}
    n = len(base_stations)

    for i in range(n):
        for j in range(i + 1, n):  # Only calculate upper triangle to avoid duplicates
            bs1 = base_stations.iloc[i]
            bs2 = base_stations.iloc[j]

            dist = haversine_distance(bs1['latitude'], bs1['longitude'],
                                      bs2['latitude'], bs2['longitude'])

            # Store distance with base station IDs as key
            key = (bs1['station_id'], bs2['station_id'])
            distances[key] = dist

            # Calculate connectivity as inverse of distance
            if dist == 0:
                connectivity[key] = float('inf')  # Infinite connectivity
                # connectivity[key] = 1.0
            else:
                connectivity[key] = 1.0 / dist

    return distances, connectivity


def create_connectivity_matrix(base_stations, connectivity):
    """
    Create a connectivity matrix for all base stations
    """
    n = len(base_stations)
    ids = base_stations['station_id'].tolist()
    id_to_index = {id: i for i, id in enumerate(ids)}

    # Initialize matrix with zeros
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, 1.0)

    # Fill the matrix with connectivity values
    for (id1, id2), conn in connectivity.items():
        i, j = id_to_index[id1], id_to_index[id2]
        matrix[i, j] = conn
        matrix[j, i] = conn  # Matrix is symmetric

    return matrix, ids


def visualize_base_stations(base_stations, connectivity=None, top_connections=20):
    """
    Visualize base stations on a map with their strongest connections
    """
    plt.figure(figsize=(12, 10))

    # Plot base stations
    plt.scatter(base_stations['longitude'], base_stations['latitude'],
                s=50, c='blue', alpha=0.7, label='Base Stations')

    # If connectivity is provided, draw lines between the most connected stations
    if connectivity is not None:
        # Sort connectivity by value (descending)
        sorted_conn = sorted(connectivity.items(), key=lambda x: x[1], reverse=True)

        # Draw lines for top connections
        for i, ((id1, id2), conn) in enumerate(sorted_conn[:top_connections]):
            # Get coordinates
            bs1 = base_stations[base_stations['station_id'] == id1].iloc[0]
            bs2 = base_stations[base_stations['station_id'] == id2].iloc[0]

            # Draw line with transparency based on connectivity
            alpha = min(conn / max(c for _, c in sorted_conn[:top_connections]) * 0.9, 0.9)
            plt.plot([bs1['longitude'], bs2['longitude']],
                     [bs1['latitude'], bs2['latitude']],
                     'r-', alpha=alpha, linewidth=1)

    plt.title('5G Base Station Locations and Connectivity')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def analyze_base_station_connectivity(file_path):
    """
    Analyze base station connectivity from CSV file
    """
    # Load data
    base_stations = pd.read_csv(file_path)
    # Calculate distances and connectivity
    distances, connectivity = calculate_distances_and_connectivity(base_stations)
    # print(f"Calculated distances and connectivity between {len(distances)} pairs of base stations")

    # Create connectivity matrix
    conn_matrix, ids = create_connectivity_matrix(base_stations, connectivity)

    return base_stations, distances, connectivity, conn_matrix, ids


# Usage example
if __name__ == "__main__":
    file_path = "../datasets/hw/bs_label.csv"
    base_stations, distances, connectivity, conn_matrix, ids = analyze_base_station_connectivity(file_path, 1)
    print(conn_matrix, ids)

    # To access connectivity between specific base stations (e.g., ID 1 and ID 2)
    # Note: Ensure the key is ordered with the lower ID first
    station_pair = tuple(sorted([1, 2]))
    if station_pair in connectivity:
        print(
            f"\nConnectivity between stations {station_pair[0]} and {station_pair[1]}: {connectivity[station_pair]:.6f}")
