import torch
import networkx as nx
from pathlib import Path

# Assuming you have the paths defined properly
root_dir = Path('../../../atsp_n5900')  # Root directory for instances
train_file = root_dir / 'train.txt'  # The file containing the instance filenames

# Read the instance filenames from train.txt
with open(train_file, 'r') as f:
    instances = f.readlines()
    instances = [x.strip() for x in instances]  # Remove newlines and spaces

# Prepare tensors to store the adjacency matrices and tour costs
# Assuming num_instances is the number of instances in train.txt and num_nodes is from each instance graph
num_instances = len(instances)
example_graph = nx.read_gpickle(root_dir / instances[0])  # Reading the first graph for size reference
num_nodes = len(example_graph.nodes)  # Assuming all graphs have the same number of nodes
print(num_nodes)
# Initialize tensors
adjacency_matrices = torch.zeros((num_instances, num_nodes, num_nodes))
tour_costs = torch.zeros(num_instances)

# Iterate over all instances
for i, instance_file in enumerate(instances):
    # Construct the file path and load the graph using read_gpickle
    graph = nx.read_gpickle(root_dir / instance_file)

    # Extract adjacency matrix
    adj_matrix = nx.to_numpy_array(graph)
    
    # Store adjacency matrix in the tensor
    adjacency_matrices[i] = torch.tensor(adj_matrix)
    
    # Calculate the tour cost by summing the edges with the 'in_solution' attribute
    tour_cost = 0
    for u, v, data in graph.edges(data=True):
        if data.get('in_solution', False):  # Check if the edge is part of the solution
            tour_cost += data.get('weight', 0)  # Add the edge weight to the tour cost

    # Store the tour cost in the tensor
    tour_costs[i] = tour_cost
print(tour_costs)

# Now `adjacency_matrices` holds the adjacency matrices for all instances
# and `tour_costs` holds the corresponding tour costs.
