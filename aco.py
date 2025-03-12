import numpy as np

# Matriz de adyacencia
adj_matrix = np.array([
    [0, 2, 4, 0, 0, 0],  # A
    [2, 0, 1, 7, 0, 0],  # B
    [4, 1, 0, 0, 3, 0],  # C
    [0, 7, 0, 0, 2, 5],  # D
    [0, 0, 3, 2, 0, 6],  # E
    [0, 0, 0, 5, 6, 0]   # F
])

n_nodes = adj_matrix.shape[0]
node_letters = ['A', 'B', 'C', 'D', 'E', 'F']

# Inicialización de feromonas
pheromone = np.ones((n_nodes, n_nodes)) * 0.1
np.fill_diagonal(pheromone, 0)

# Parámetros del algoritmo ACO
alpha = 1
beta = 2
evaporation_rate = 0.5
n_ants = 10
n_iterations = 100

# Información heurística (1/distancia)
eta = 1 / (adj_matrix + np.eye(n_nodes))
eta[adj_matrix == 0] = 0  # Asegurar que no haya infinitos en nodos desconectados

# Algoritmo ACO para ruta de A a F
start_node = 0  # A
end_node = 5    # F

best_path = None
best_distance = float('inf')

for iteration in range(n_iterations):
    all_paths = []
    all_distances = []
    
    for ant in range(n_ants):
        current_node = start_node
        path = [current_node]
        visited = set([current_node])
        distance = 0
        
        while current_node != end_node:
            unvisited = [n for n in range(n_nodes) if n not in visited]
            
            # Solo considerar nodos conectados (distancia > 0)
            connected_nodes = [n for n in unvisited if adj_matrix[current_node, n] > 0]
            
            if not connected_nodes:  # Si no hay nodos conectados, terminar
                break
            
            probabilities = np.zeros(len(connected_nodes))
            for i, node in enumerate(connected_nodes):
                probabilities[i] = pheromone[current_node, node]**alpha * eta[current_node, node]**beta
            
            if probabilities.sum() == 0:  # Si no hay probabilidades válidas, terminar
                break
            
            probabilities /= probabilities.sum()
            
            next_node = np.random.choice(connected_nodes, p=probabilities)
            path.append(next_node)
            visited.add(next_node)
            distance += adj_matrix[current_node][next_node]
            current_node = next_node
        
        if current_node == end_node:  # Solo considerar rutas que llegan a F
            all_paths.append(path)
            all_distances.append(distance)
            
            if distance < best_distance:
                best_distance = distance
                best_path = path

    # Actualización de feromonas
    pheromone *= (1 - evaporation_rate)
    
    for path, dist in zip(all_paths, all_distances):
        for i in range(len(path)-1):
            pheromone[path[i], path[i+1]] += 1/dist
            pheromone[path[i+1], path[i]] += 1/dist

# Formatear resultado
if best_path:
    formatted_path = ' → '.join([node_letters[node] for node in best_path])
    print(f"Mejor ruta de A a F: {formatted_path}")
    print(f"Distancia total: {best_distance}")
else:
    print("No se encontró una ruta válida de A a F.")