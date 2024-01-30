import math
import random

import networkx as nx
import specifications as s
import random as rand
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import collections

from matplotlib import colors
from matplotlib import colorbar


#Generate node positions as an array of arrays, with each element being an np.array that has the xy-coordinates of a node.
def generate_nodes():

    #The node coordinates are generated uniformly from R, according to specifications
    node_positions = np.array([(rand.uniform(-s.R, s.R), rand.uniform(-s.R, s.R)) for i in range(s.nodes)])
    return node_positions


#A matrix that contains the distance between all node pairs.
def distance_matrix(positions: np.ndarray):
    distance_matrix = np.zeros((s.nodes, s.nodes))

    for i in range(s.nodes):
        for j in range(i+1, s.nodes):
            #Calculates Euclidean distance between two nodes.
            distance_ij = sc.spatial.distance.euclidean(positions[i], positions[j], w=None)
            distance_matrix[i,j] = distance_ij

    #This makes the distance matrix symmetrical (and is done due to consistency. Later calculations only use the upper triangle of the matrices.)
    distance_matrix = distance_matrix +distance_matrix.T - np.diag(np.diag(distance_matrix))

    return distance_matrix

def adjacency_matrix_with_weights_erdos_renyi(probability: float, distances: np.ndarray):
    adjacency_matrix = np.zeros((s.nodes, s.nodes), dtype=float)
    for i in range(s.nodes):
        for j in range(i + 1, s.nodes):
            random_value = rand.random()
            distance_ij = distances[i, j]

            if random_value < probability:
                adjacency_matrix[i, j] = calculate_capacity_of_edge(distance_ij)

    adjacency_matrix = adjacency_matrix + adjacency_matrix.T - np.diag(np.diag(adjacency_matrix))

    return adjacency_matrix





#The Waxman-networks adjacency matrix. Contains the matrix form of all edge capacities
def adjacency_matrix_with_weights_waxman(distances: np.ndarray):

    #Adjacency matrix for the Waxman-network
    adjacency_matrix = np.zeros((s.nodes, s.nodes), dtype=float)


    for i in range(s.nodes):
        for j in range(i + 1, s.nodes):
            #For each node pair, get the distance between them
            distance_ij = distances[i,j]
            #Initialize random value between [0,1]
            random_value = rand.random()
            #Get the Waxman-probability for forming an edge between the current node pair
            waxman_probability = np.exp(-distance_ij / (s.alfa * s.L))

            #The random value is generated between [0,1]. If the Waxman-probability is large, the random value can be on a larger interval.
            if random_value < waxman_probability:
                adjacency_matrix[i,j] = calculate_capacity_of_edge(distance_ij)

    #Symmetrizes the adjacency matrix
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T - np.diag(np.diag(adjacency_matrix))
    return adjacency_matrix



#The networkx representation of the graph
def generate_waxman_or_erdos_renyi_network(node_positions: np.ndarray, adjacency_matrix: np.ndarray, distance_matrix: np.ndarray):

        #Create an empty networkx-graph
        graph  = nx.Graph()
        #Generate all nodes with their positions
        for i in range(s.nodes):
            graph.add_node(i, coordinates=(node_positions[i]))


        #Connect them with precalculate distances and capacities for edges. Only create edge if they are adjacent
        for i in range(s.nodes):
            for j in range(i +1, s.nodes):
                if adjacency_matrix[i,j] != 0:
                    graph.add_edge(i,j, capacity = adjacency_matrix[i,j], distance = distance_matrix[i,j])

        return graph


#THIS NETWORK IS CREATED IN A FUNDAMENTALLY DIFFERING WAY FROM WAXMAN NETWORKS!
#WHERE THE WAXMAN EDGES CAN BE IMMEDIATELY CALCULATED BASED ON NODE LOCATIONS, THE SCALEFREE NETWORK
#ADDS EACH NODE TO THE GRAPH SEPARATELY, CHECKING ALL PRE-EXISTING NODES IN THE GRAPH AND CONNECTS TO M NODES (M
#SPECIFIED IN FILE AND THEORY.) THE CHOICE OF WHICH M NODES THE ADDED NODE CONNECTS TO COMES FROM A WEIGHTED RANDOM SAMPLE
def generate_scalefree_network(node_positions: np.ndarray, distance_matrix: np.ndarray):

    #Create graph-object
    graph = nx.Graph()
    #Create list, where the value of the i:th element corresponds to the degree of the i:th node in the network. The amount of elements is known beforehand
    #from the network size.
    node_degrees = np.zeros(s.nodes)

    #Generate all nodes
    for i in range(s.nodes):
        graph.add_node(i, coordinates=(node_positions[i]))

    #Initially none of the nodes are connected.
    adjacency_matrix = np.zeros((s.nodes, s.nodes))


    #Generate initial connections according to parameter m. All initial nodes are connected to each other
    for i in range(s.initial_nodecount):
        for j in range(i + 1, s.initial_nodecount):
            #Set adjacency matrix element to capacity of the edge, the formula for capacity comes from Quantum complex network transitions.
            adjacency_matrix[i,j] = calculate_capacity_of_edge(distance_matrix[i,j])
            #Add an edge to the network-object.
            graph.add_edge(i,j, capacity = adjacency_matrix[i,j], distance = distance_matrix[i,j])


        #Sets the i:th element of the node degree list to the degree of the i:th node.
        np.put(node_degrees, [i], [graph.degree[i]])


    #Next, go through the rest of the generated nodes and connect them to m other nodes THAT ARE ALREADY IN THE NETWORK.

    #Create the weights for added node and pre-existing nodes, going from m +1 to N
    #This loop is for the node to be added to the network
    for i in range(s.initial_nodecount, s.nodes):

        #The number of connection choices increase as the graph grows. Note that only m nodes are still chosen to be connected to, but as the
        #network grows there are more choices to choose the m from.
        weights = np.zeros(i)

        #This loop is for the nodes already in the network.
        for j in range(i):
            #The concept of weight for a node pair is as follows: As a new node is added, a weight is calculated for each pair of that the new node
            #and pre-existing nodes in the network. The weight is proportional to the degree of the pre-existing node and inversely proportional to the
            #distance between the pre-existing node and node to be added. The m largest weighted pairs are chosed as the connections the new node forms in the network.
            # This type of weighting creates hubs in the network, where there is a cluster of
            #nodes that have the highest degree, while the rest of the nodes have a low degree.

            #Calculate the weight of the node
            weight = graph.degree[j] / distance_matrix[j,i]
            #Put the j:th node's weight into the j:th position of the weights-array.
            np.put(weights, [j], weight)


        #Normalization of the weights, mandatory for np.random.choice
        sum_of_weights = np.sum(weights)
        normalized_weights = np.divide(weights, sum_of_weights)


        #Get i nodes from the weights-array.
        nodes = np.array(graph.nodes)[:len(weights)]
        #Contains the node-numbers that the new node will connect to.
        sampled_items = np.random.choice(nodes, size=s.initial_nodecount, p=normalized_weights, replace=False)
        #Add edges according to the choices of samples nodes
        for node in sampled_items:
            adjacency_matrix[node, i] = calculate_capacity_of_edge(distance_matrix[node,i])
            graph.add_edge(node, i, capacity = adjacency_matrix[node, i], distance = distance_matrix[node, i])
    return graph

#As the name suggests, calculates the capacity of edge according to Quantum complex network transitions
def calculate_capacity_of_edge(distance: float):
    return -math.log2(1 - 10 ** (-s.gamma * distance))

def calculate_capacity_and_graph_distance_matrices(graph: nx.Graph):
    fields = ['Graph distance', 'End-to-end capacity']
    filename = f"{s.graph_type}_{s.nodes}_node_graph.csv"
    rows = []

    #Initialize capacities to be 0 and graph distances between nodes to be infinite
    end_to_end_capacity_matrix = np.zeros((s.nodes, s.nodes), dtype=float)
    graph_distance_matrix = np.empty((s.nodes, s.nodes), dtype=float)
    graph_distance_matrix[:] = np.inf



    for start_node in range(s.nodes):
        for end_node in range(start_node + 1, s.nodes):
            try:
                capacity = nx.minimum_cut(graph, start_node, end_node)[0]
                end_to_end_capacity_matrix[start_node, end_node] = capacity


            except:
                print("Raising exception for capacity")

            try:

                shortest_path = nx.shortest_path_length(graph, source=start_node, target=end_node, weight='distance',
               method='dijkstra')
                graph_distance_matrix[start_node, end_node] = shortest_path
            except:
                pass

        rows.append([graph_distance_matrix[start_node, end_node], end_to_end_capacity_matrix[start_node, end_node]])

    with open(filename, 'w', newline="") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
         #writing the fields
        csvwriter.writerow(fields)
        #writing the rows
        csvwriter.writerows(rows)

    end_to_end_capacity_matrix = end_to_end_capacity_matrix + end_to_end_capacity_matrix.T - np.diag(np.diag(end_to_end_capacity_matrix))
    return  graph_distance_matrix, end_to_end_capacity_matrix


def calculate_average_capacity(end_to_end_capacity_matrix: np.ndarray):
    sum = 0
    count = 0
    for i in range(s.nodes):
        for j in range(i + 1, s.nodes):
            sum += end_to_end_capacity_matrix[i,j]
            count += 1
    return sum/ count

#Gets the upper triangle of the graph distance matrix and end-to-end-capacity matrix, used for plotting the data
def create_plot_data_graph_distance_capacity(graph_distance_matrix: np.ndarray, end_to_end_capacity_matrix:np.ndarray):

    graph_distance_data = graph_distance_matrix[np.triu_indices(s.nodes, k=1)]
    end_to_end_capacity_data = end_to_end_capacity_matrix[np.triu_indices(s.nodes, k=1)]

    return graph_distance_data, end_to_end_capacity_data
#Creates a scale-free network.

def create_plot(graph_distance_data: np.ndarray, end_to_end_capacity_data:np.ndarray, average_capacity: float):
    filename = str(s.nodes) + "node graph" + s.graph_type + ".png"
    plt.scatter(graph_distance_data, end_to_end_capacity_data, 6)
    plt.title(r'$\langle \mathcal{C}  \rangle$ = ' + str(round(average_capacity, 7)) + '\n N =' + str(s.nodes))
    plt.xlabel("Graph distance $d_{G}$")
    plt.ylabel("$\mathcal{C}$")
    plt.savefig(filename)
    plt.show()
    plt.draw()



def create_degree_distribution_histogram(graph: nx.Graph):
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')
    plt.xticks(list(map(int, deg)))
    plt.title(f"{s.graph_type}-network degree distribution with ".capitalize() + str(s.nodes) + f" nodes")
    plt.ylabel("Node count")
    plt.xlabel("Degree")
    plt.show()

    plt.savefig(f"{s.graph_type}_{s.nodes}_nodes_degree_distribution.png")


def draw_graph(graph: nx.Graph, node_positions: np.ndarray):

        #Weights for edges
        capacities = np.array([graph[u][v]['capacity'] for u, v in graph.edges])


        #Get smallest and largest capacity
        min_capacity = np.min(capacities)
        max_capacity = np.max(capacities)

        scaled_capacities = np.divide(capacities - min_capacity, max_capacity - min_capacity)


        # Apply a logarithmic scale to make the small differences more noticeable
        log_scaled_capacities = np.array([np.log10(1 + capacity)*10**9 for capacity in scaled_capacities])


        norm = colors.Normalize(vmin=min_capacity, vmax=max_capacity)

        cmap = plt.cm.seismic

        edge_colors = [cmap(norm(capacity)) for capacity in log_scaled_capacities]

        fig, ax = plt.subplots()


        plt.title(f"{s.graph_type}-network with {s.nodes} nodes".capitalize())

        nx.draw(graph, node_positions, with_labels=False, node_size = 20, edge_cmap = cmap, edge_color = edge_colors, ax = ax)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin =min_capacity, vmax=max_capacity))
        sm._A = []

        plt.colorbar(sm, label = "Log-scaled edge capacities")

        font2 = {'family': 'serif', 'color': 'black', 'size': 15}

        plt.axis('on')
        plt.ylabel("$\Omega$", fontdict = font2)
        plt.ylim(-s.R, s.R)
        plt.xlim(-s.R, s.R)
        plt.show()


