import networkx as nx
import networkx as nx
import json
import matplotlib.pyplot as plt
from networkx.algorithms import community
import pickle
import numpy as np
import argparse
import os.path
from os import path

def find_node_type(id_val):
	nodes = data["nodes"]
	for i in nodes:
		if i["id"] == id_val:
			if i["entity_type"] == 'class':
				return 1
			else:
				return 0
	print ("Something is wrong with this")

def find_node(id_val):
	"""
	Finding the label of graph nodes given the node id
	Input: ID of node
	Output: Label of node
	"""
	nodes = data["nodes"]
	for i in nodes:
		if i["id"] == id_val:
			return i["label"]
	print ("Something is wrong")

def find_node_id(name):
	"""
	Finding the id of graph nodes given the node name
	Input: ID of node
	Output: Label of node
	"""
	nodes = data["nodes"]
	for i in nodes:
		if i["label"] == name:
			return i["id"]
	print ("Something is worng")



def check_community(i,j, comm):
	flag = 0
	for k in comm:
		if i in k and j in k:
			flag = 1
	return flag


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--clusterFile')
	parser.add_argument('--outFile')
	args = parser.parse_args()



	graph_file = args.clusterFile

	with open(graph_file) as json_file:
	    data = json.load(json_file)

	for i in data["edges"]:
		if i["type"] == "inter_class_connections":
			data_read = i["relationship"]
			break

	g = nx.DiGraph() #Directed Graph
	for i in data_read:
		g.add_edge(find_node(i["properties"]["start"]), find_node(i["properties"]["end"]))


	B = nx.directed_modularity_matrix(g)
	node_list = g.nodes()


	comm = []
	count = 0
	comm_data = data["clusters"]
	for i in comm_data:
		temp = []
		for j in i["nodes"]:
			if find_node_type(j):
				temp.append(find_node(j))
		comm.append(temp)
		count += len(temp)

	print (count)
	""" Calculating modularity score """
	modularity_score = 0
	for i,i_node in enumerate(node_list):
		for j,j_node in enumerate(node_list):
			value = check_community(i_node,j_node, comm)
			if value == 1:
				modularity_score += B.item((i,j))

	number_of_edges = g.number_of_edges()
	score = modularity_score/number_of_edges

	if path.exists(args.outFile):
		with open(args.outFile, 'r') as json_file:
			data = json.load(json_file)
	else:
		data = {}

	data["modularity"] = {"score":score, "range":"[-1,1]"}

	with open(args.outFile, 'w') as f:
		json.dump(data, f)

	print ("modularity",score)
