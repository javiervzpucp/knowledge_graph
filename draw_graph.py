# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:45:43 2024

@author: jveraz
"""

import json
import networkx as nx
import matplotlib.pyplot as plt

# Abrir el archivo JSON
with open("paucar_graph.json", "r") as file:
    graph_data = json.load(file)

# Crear un grafo vacío
nx_graph = nx.Graph()  # Usa nx.DiGraph() si el grafo es dirigido

# Agregar nodos al grafo
for node in graph_data["nodes"]:
    node_id = node["id"]  # Identificador del nodo
    node_properties = node["properties"]  # Propiedades del nodo
    nx_graph.add_node(node_id, **node_properties)

# Agregar aristas (relaciones) al grafo
for edge in graph_data["edges"]:
    source = edge["source"]  # Nodo origen
    target = edge["target"]  # Nodo destino
    edge_type = edge["type"]  # Tipo de relación
    edge_properties = edge["properties"]  # Propiedades de la relación
    nx_graph.add_edge(source, target, type=edge_type, **edge_properties)
    
# Procesar el componente más grande del grafo
#Gcc = sorted(nx.connected_components(nx_graph), key=len, reverse=True)
#nx_graph = nx_graph.subgraph(Gcc[0])

# Layout del grafo
pos = nx.kamada_kawai_layout(nx_graph)

# Figura
plt.figure(figsize=(10, 10))

# Dibujar aristas
nx.draw_networkx_edges(nx_graph, pos, edge_color="k", width=0.75)

# Dibujar nodos
nx.draw_networkx_nodes(nx_graph, pos, node_size=500, node_color="gold")

# Crear etiquetas personalizadas
labels = {node: nx_graph.nodes[node].get("id", str(node)) for node in nx_graph.nodes()}

# Dibujar etiquetas
nx.draw_networkx_labels(nx_graph, pos, labels=labels, font_size=4, font_weight="bold")

# Dibujar etiquetas
nx.draw_networkx_labels(nx_graph, pos, font_size=4, font_weight="bold")

# Guardar la imagen
plt.savefig("graph.png", format="PNG")
plt.show()
