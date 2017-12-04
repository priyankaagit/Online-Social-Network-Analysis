import networkx as nx
from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import urllib.request

def jaccard_wt(graph, node):
    """
  The weighted jaccard score, defined above.
  Args:
    graph....a networkx graph
    node.....a node to score potential new edges for.
  Returns:
    A list of ((node, ni), score) tuples, representing the 
              score assigned to edge (node, ni)
              (note the edge order)
  """

    edges = graph.edges()
    neighbors = set(graph.neighbors(node))
    scores = []
    for n in graph.nodes():
        s = 0
        neighbors2 = set(graph.neighbors(n))
        intersect = set(neighbors & neighbors2)
        for l in list(graph.degree(list(intersect)).values()):
            if l!=0:
                s = s + 1/l
        if sum(list(graph.degree(list(neighbors)).values())) !=0 and sum(list(graph.degree(list(neighbors2)).values())):
            scores.append(((node,n),s /((1/sum(list(graph.degree(list(neighbors)).values()))) + 
                                       (1/sum(list(graph.degree(list(neighbors2)).values()))))))    
        
    return sorted(scores, key=lambda x: x[1], reverse=True)
        
    pass