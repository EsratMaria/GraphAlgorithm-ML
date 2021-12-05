import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from py2neo import Graph, Node, Relationship
import IPython.display
import nxneo4j as netneo
from neo4j import GraphDatabase
import pandas as pd
import networkx as nx
import numpy as np
import csv

# my-db credentials
password = '1234'
user = 'neo4j'
uri = 'bolt://localhost:7687'

driver = GraphDatabase.driver(uri=uri, auth=(user, password))

my_got = "C:\\Users\\Esrat Maria\\Desktop\\my_got.csv"
dataset = pd.read_csv(my_got)

# print(dataset.head())

# preparing data for ML model
data = dataset.dropna()
X = data[['allegiances', 'nobility', 'has_dead_rels', 'culture', 'house',
          'gender', 'has_Allegiance', 'mother', 'father', 'spouse', 'heir', 'data_poor']]
X = pd.get_dummies(X)
y = data['is_alive']


# building ML model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
classifier = RandomForestClassifier(
    n_estimators=50, criterion='entropy', random_state=42, max_depth=8)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))

# Trying to make a better model by using graph algorithms

# query = """
# LOAD CSV WITH HEADERS FROM 'file:///my_got.csv' AS line
# WITH line
# WHERE line.house IS NOT NULL
# MERGE(person:Person {Name: line.name})
# MERGE(house:House {Name: line.house})
# MERGE(person) - [:belongs_to] -> (house)
# """

#_to_neo4jDB = graph.run(query)

print("-------------------------------------------")

G = netneo.Graph(driver)
G.delete_all()
G.load_got()
G.identifier_property = 'name'
G.relationship_type = '*'
G.node_label = 'Character'

# Graph Algorithm 1
# the most influential characters
# PageRank Algorithm

# query = '''
# CALL gds.pageRank.stream('prGraph')
# YIELD nodeId, score
# RETURN gds.util.asNode(nodeId).Name AS name, score as pageRank
# ORDER BY pageRank DESC
# limit 1962
# '''

# _to_neo4jDB = graph.run(query).to_data_frame()


pagerank = netneo.pagerank(G)
sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
for character, score in sorted_pagerank[:5]:
    print(character, score)

print("--------------------------------------------------------------")

_to_df = pd.DataFrame(list(pagerank.items()), columns=['name', 'pagerank'])
print(_to_df.head())

print("--------------------------------------------------------------")

merge_pagerank_data_with_csv = data.merge(_to_df, on='name', how='left')
print(merge_pagerank_data_with_csv.head())

print("--------------------------------------------------------------")

# Graph Algorithm 2
# calculating Closeness centrality

closeness_centrality = netneo.closeness_centrality(G)

_cc_to_df = pd.DataFrame(list(closeness_centrality.items()), columns=[
                         'name', 'closeness centrality'])
print(_cc_to_df.head())

print("--------------------------------------------------------------")

merge_cc_data_with_csv = merge_pagerank_data_with_csv.merge(
    _cc_to_df, on='name', how='left')
print(merge_cc_data_with_csv.head())

print("--------------------------------------------------------------")

# checking the ML model again to see if the accuracy got any better

merge_cc_data_with_csv = merge_cc_data_with_csv.fillna(
    merge_cc_data_with_csv.mean())

X = merge_cc_data_with_csv[['allegiances', 'nobility', 'has_dead_rels', 'culture', 'house',
                            'gender', 'has_Allegiance', 'mother', 'father', 'spouse', 'heir', 'data_poor', 'pagerank', 'closeness centrality']]
X = pd.get_dummies(X)
y = data['is_alive']


# building ML model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.37, random_state=42)
classifier = RandomForestClassifier(
    n_estimators=50, criterion='entropy', random_state=42, max_depth=8)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))

print("--------------------------------------------------------------")
