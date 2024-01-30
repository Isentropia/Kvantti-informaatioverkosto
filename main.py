import functions as f
import networkx as nx
import matplotlib.pyplot as plt
import specifications as s


def main():

    node_positions = f.generate_nodes()
    distance_matrix = f.distance_matrix(node_positions)

    if s.graph_type == "waxman":
        adjacency_matrix = f.adjacency_matrix_with_weights_waxman(distance_matrix)
        graph = f.generate_waxman_or_erdos_renyi_network(node_positions, adjacency_matrix, distance_matrix)
    elif s.graph_type == "scalefree":
        graph = f.generate_scalefree_network(node_positions, distance_matrix)

    elif s.graph_type == "erdos_renyi":
        adjacency_matrix = f.adjacency_matrix_with_weights_erdos_renyi(distance_matrix)
        graph = f.generate_waxman_or_erdos_renyi_network(node_positions, adjacency_matrix, distance_matrix)


    graph_distance_matrix, end_to_end_capacity_matrix = f.calculate_capacity_and_graph_distance_matrices(graph)
    average_capacity = f.calculate_average_capacity(end_to_end_capacity_matrix)

    graph_distance_data, end_to_end_capacity_data = f.create_plot_data_graph_distance_capacity(graph_distance_matrix, end_to_end_capacity_matrix)

   #Visualization for graph and degree distribution
    f.create_plot(graph_distance_data, end_to_end_capacity_data, average_capacity)
    f.create_degree_distribution_histogram(graph)

    f.draw_graph(graph, node_positions)

if __name__ == '__main__':
    main()
