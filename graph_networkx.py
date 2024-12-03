# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:00:25 2024

@author: jveraz
"""

import os
from dotenv import load_dotenv
#from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
import networkx as nx
import json
import matplotlib.pyplot as plt

#############################
## Conexiones con las APIs ##
#############################

# Cargar variables del archivo .env en desarrollo
load_dotenv()

# Acceder a la clave API
api_key = os.getenv("OPENAI_API_KEY")

# LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4")

############
## Textos ##
############
    
# Leer el archivo 'video.txt' y procesar el texto
with open("archivos/video_processed.txt", "r", encoding="utf-8") as file:
    lines = file.read()

###########
## Grafo ##
###########

# documentos
documents = [Document(page_content=lines)]

# Initialize LLMGraphTransformer and create graph documents
graph_transformer = LLMGraphTransformer(llm=llm)
graph_documents = graph_transformer.convert_to_graph_documents(documents)

#print(f"Nodes:{graph_documents[0].nodes}")
#print(f"Relationships:{graph_documents[0].relationships}")

# Create a NetworkX graph and add nodes and edges with content
nx_graph = nx.Graph()

# Loop through graph_documents to add nodes and edges with 'content'
for doc in graph_documents:
    for node in doc.nodes:
        node_id = dict(node)["id"]  # Access 'id' directly
        content = dict(node)["type"]  # Access 'content' directly
        #attributes = dict()#dict(node)["attributes"]  # Access 'attributes' directly
        #attributes["content"] = content  # Ensure 'content' is stored in attributes
        nx_graph.add_node(node_id, content=content)

for doc in graph_documents:    
    for edge in doc.relationships:
        edge_dict = vars(edge) if not isinstance(edge, dict) else edge  # Convert to dict if necessary
        source = str(dict(edge_dict["source"])["id"])  # Convert source to string
        target = str(dict(edge_dict["target"])["id"])  # Convert target to string
        print(source,target)
        tipo = edge_dict["type"]
        nx_graph.add_edge(source, target, content=tipo)

# Save the graph with nodes and edges to JSON
data = nx.node_link_data(nx_graph)
with open("archivos/saved_graph.json", "w") as file:
    json.dump(data, file)


# leemos
with open("archivos/saved_graph.json", "r") as file:
    data = json.load(file)

# de vuelta a networkx
nx_graph = nx.node_link_graph(data)

# layout
pos = nx.spring_layout(nx_graph)  # Spring layout positions

# figura
plt.figure(figsize=(10, 10))  # Set the figure size

# aristas
nx.draw_networkx_edges(nx_graph, pos, edge_color="k", width=0.75)

# nodos
nx.draw_networkx_nodes(nx_graph, pos, node_size=500, node_color="gold")

# labels
nx.draw_networkx_labels(nx_graph, pos, font_size=7, font_weight="bold")

# guardamos
plt.savefig("archivos/graph.png", format="PNG")
plt.show()  

