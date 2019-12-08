#network of states
#Alan Balu

#import statements
import numpy as np, math
import matplotlib.pyplot as plt
import pandas as pd

from pprint import pprint

import networkx as nx
import matplotlib.pyplot as plt
import community


def main():
	print("main")

	network_df = pd.read_csv('network_df.csv' , sep=',', encoding='latin1')
	states_df = pd.concat([network_df["STATE_ABBR"]], axis = 1)

	states_dict = states_df.to_dict('index')

	new_states_dict = {}

	counter = 0
	for item in states_dict:
		new_states_dict[counter] = states_dict.get(item).get("STATE_ABBR")
		counter += 1

	#pprint(new_states_dict)

	merged_graph = nx.Graph()
	with open('final_network.csv', 'r') as f:
		junk = f.readline()
		for line in f:
			data = line.strip().split(',')
			#source = int(data[0])
			#target = int(data[1])
			source = data[0]
			target = data[1]
			weight = float(data[2])

			if (weight > 0.8):      #0.6 works well for clustering, but higher is better for modularity score and seeing stronger relationships
				merged_graph.add_edge(source, target, weight=weight)

    #VISUALIZE NETWORK
	nx.draw(merged_graph, node_size=9, edge_size = 1, node_color='b', with_labels = True, font_color = "purple")
	plt.show()

	#--------------------------------------------------

	degree_sequence = [d for n, d in merged_graph.degree()] # degree sequence
	print(degree_sequence)

	plt.hist(degree_sequence, bins = 10)
	plt.ylabel('Count')
	plt.xlabel('Degree of Nodes')

	plt.show()
	plt.clf()

	plt.scatter(list(merged_graph.nodes), degree_sequence)
	ax = plt.axes()
	ax.tick_params(which="both", bottom=False, left=True, labelsize = 7)
	plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
	plt.ylabel('Degree')
	plt.xlabel('States')

	plt.show()
	plt.clf()


	print("density of subgraph: ", nx.density(merged_graph))
	# Compute and print number of nodes and edges
	nbr_nodes = len(merged_graph.nodes())
	nbr_edges = len(merged_graph.edges())

	print("Number of nodes:", nbr_nodes)
	print("Number of edges:", nbr_edges)

	#-------------------------------------------------

	nbr_components = nx.number_connected_components(merged_graph)
	print("Number of connected components:", nbr_components)
	print("number of triangles: \n")
	print(nx.triangles(merged_graph).values())

	#-----------------------------------------------

	triangles = 0
	for value in nx.triangles(merged_graph).values():
		triangles += value

	print("num tri: ", triangles/3)

	#--------------------------------------------

	#calculate degree centrality for all nodes
	centralities = nx.degree_centrality(merged_graph)
	values = [centralities[node] for node in merged_graph.nodes()]

	print("degree centrality average: ", sum(values)/len(values))


	#plot graph with degree centrality as colors
	plt.clf()
	nx.draw(merged_graph, cmap = plt.get_cmap('coolwarm'), node_color = values, node_size=13, with_labels=True, font_size = 9)
	plt.show()
	#plt.savefig("degree_centrality.png")
	plt.clf()

	#calculate degree centrality for all nodes
	centralities = nx.betweenness_centrality(merged_graph)
	values = [centralities[node] for node in merged_graph.nodes()]

	print("betweenness centrality average: ", sum(values)/len(values))

	#plot graph with degree centrality as colors
	plt.clf()
	nx.draw(merged_graph, cmap = plt.get_cmap('coolwarm'), node_color = values, node_size=13, with_labels=True, font_size = 9)
	plt.show()
	#plt.savefig("betweeness_centrality.png")
	plt.clf()

	#--------------------------------------

	test = [merged_graph.subgraph(c) for c in nx.connected_components(merged_graph)]
	maximum = max(len(c.nodes()) for c in test)

	print("largest connected component: ")
	print("number of nodes: ", maximum)

	#determine which connected component is the largest and print that
	max_len = 0
	largest_CC = []
	for item in test:
		if (len(item.nodes()) > max_len):
			max_len = len(item.nodes())
			largest_CC = item.nodes()

	print(largest_CC)

	#---------------------------------------------

	#####################
	# Clustering
	#####################
	# Conduct modularity clustering
	# Create an unweighted version of G because modularity works only on graphs with non-negative edge weights
	# for u, v in undirected_SG.edges():
	#     unweighted_SG.add_edge(u, v)
	partition = community.best_partition(merged_graph)

	# Print clusters (You will get a list of each node with the cluster you are in)
	print()
	print("Clusters")
	print(partition)

	things = [partition.get(item) for item in partition]
	print("number of clusters: ", max(things)+1)
	print("average clustering coefficient: ", nx.average_clustering(merged_graph))

	# Get the values for the clusters and select the node color based on the cluster value
	values = [partition.get(node) for node in merged_graph.nodes()]
	print(values)
	nx.draw(merged_graph, cmap = plt.get_cmap('jet'), node_color = values, node_size=13, with_labels=True, font_size = 9)
	plt.show()

	# Determine the final modularity value of the network
	modValue = community.modularity(partition, merged_graph)
	print("modularity: {}".format(modValue))

	plt.scatter(list(merged_graph.nodes), degree_sequence, cmap = 'rainbow', c=values)
	ax = plt.axes()
	ax.tick_params(which="both", bottom=False, left=True, labelsize = 7)
	plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
	plt.ylabel('Degree')
	plt.xlabel('States')

	plt.show()
	plt.clf()


if __name__ == '__main__':
	main()
