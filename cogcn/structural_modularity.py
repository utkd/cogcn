import json
import numpy as np
import argparse
import os.path
from os import path
import distance

def structural_modularity(clusters, edges):
    """ Structural modularity quality """
    result = clusters
    # scoh: structural cohesiveness of a service
    scoh = [] 
    
    # scop: coupling between service 
    scop = np.empty([len(result), len(result)], dtype=float)

    for _, value in result.items():
        n_cls = len(value['classes']) 
        mu = 0
        for i in range(n_cls-1):
            for j in range(i, n_cls):
                c1 = value['classes'][i]
                c2 = value['classes'][j]
                if c1 + "--" + c2 in edges  or c2 +"--" + c1 in edges :
                    mu += 1 
        scoh.append(mu * 1.0/(n_cls * n_cls))
     
    for key1, value1 in result.items():
        for key2, value2 in result.items():
            sigma = 0
            if key1 != key2:
                c_i = value1['classes']
                c_j = value2['classes'] 
                for i in range(len(c_i)):
                    for j in range(len(c_j)):
                        c1 = c_i[i]
                        c2 = c_j[j]
                        if c1 + "--" + c2 in edges or c2 + "--" + c1 in edges:
                            sigma += 1 
                scop[int(key1)][int(key2)] = sigma * 1.0 / (2 * len(c_i) * len(c_j))
    """Cohesion""" 
    p1 = sum(scoh) * 1.0 / len(scoh)

    """Coupling"""
    p2 = 0
    for i in range(len(scop)):
        for j in range(len(scop[0])):
            if i != j:
                p2 += scop[i][j]
        
    if len(scop) == 1:
        p2 = 0
    else:
        p2 = p2 / len(scop) / (len(scop) - 1) * 2
    

    smq = p1 - p2
    return smq

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
            return i["label"].split(".")[-1]
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
            data_edge = i["relationship"]
            break
    edge = {}
    for i in data_edge:
        key = find_node(i["properties"]["start"])+"--"+find_node(i["properties"]["end"])
        edge[key] = int(i["frequency"])

    cluster_process = data["clusters"]
    cluster_groups = []
    for i in cluster_process:
        temp = []
        for j in i["nodes"]:
            if find_node_type(j):
                temp.append(find_node(j))
        cluster_groups.append(temp)

    clusters = {}
    for i,j in enumerate(cluster_groups):
        temp = {}
        temp['classes'] = j
        clusters[str(i)] = temp
    score = structural_modularity(clusters, edge)

    if path.exists(args.outFile):
        with open(args.outFile, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = {}

    data["structural_modularity"] = {"score":score, "range":"[-1,1]"}

    with open(args.outFile, 'w') as f:
        json.dump(data, f)

    print ("Structural Modularity",score)