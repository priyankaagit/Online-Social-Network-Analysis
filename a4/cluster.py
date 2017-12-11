from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import urllib.request


def approximate_betweenness(graph):

	eb = nx.edge_betweenness_centrality(graph)
	return sorted(eb.items(), key=lambda x: x[1], reverse=True)
	pass
	
def partition_girvan_newman(graph):
    """
    Use your approximate_betweenness implementation to partition a graph.
    Unlike in class, here you will not implement this recursively. Instead,
    just remove edges until more than one component is created, then return
    those components.
    That is, compute the approximate betweenness of all edges, and remove
    them until multiple comonents are created.

    You only need to compute the betweenness once.
    If there are ties in edge betweenness, break by edge name (e.g.,
    (('A', 'B'), 1.0) comes before (('B', 'C'), 1.0)).

    Note: the original graph variable should not be modified. Instead,
    make a copy of the original graph prior to removing edges.
    See the Graph.copy method https://networkx.github.io/documentation/development/reference/generated/networkx.Graph.copy.html
    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A list of networkx Graph objects, one per partition.

    >>> components = partition_girvan_newman(example_graph(), 5)
    >>> components = sorted(components, key=lambda x: sorted(x.nodes())[0])
    >>> sorted(components[0].nodes())
    ['A', 'B', 'C']
    >>> sorted(components[1].nodes())
    ['D', 'E', 'F', 'G']
    """
    ###TODO
    graph_copy = graph.copy()
    appx_betweenness = approximate_betweenness(graph)
    components = [c for c in nx.connected_component_subgraphs(graph_copy)]
    while len(components) == 1:
        edge_remove =  max(appx_betweenness, key=lambda x: x[1])
        graph_copy.remove_edge(*edge_remove[0])
        appx_betweenness.remove(edge_remove)
        components = [c for c in nx.connected_component_subgraphs(graph_copy)]  
    result = [c for c in components]
    return sorted(result, key=lambda x: sorted(x.nodes())[0])
    pass

def read_graph():
    """ Read 'follower.txt' into a networkx **undirected** graph.
    Returns:
      A networkx undirected graph.
    """
    return nx.read_edgelist('follower.txt', delimiter='\t')

def get_subgraph(graph, min_degree):
    """Return a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    We'll use this in the main method to prune the original graph.

    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.

    >>> subgraph = get_subgraph(example_graph(), 3)
    >>> sorted(subgraph.nodes())
    ['B', 'D', 'F']
    >>> len(subgraph.edges())
    2
    """
    ###TODO
    node = graph.nodes()
    delnode = []
    for n in node:
        if nx.degree(graph,n) < min_degree:
            delnode.append(n)
    for d in delnode:
        graph.remove_node(d)
    return graph
    pass

def write_file(clusters,subgraph):  
    file = open('cluster.txt','w')
    file.write(str(len(clusters)))
    file.write("\n")
    file.write(str(subgraph.order()/len(clusters)))
    file.close()

def main():
    graph = read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    clusters = partition_girvan_newman(subgraph)	
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    write_file(clusters,subgraph)
    print('Communitues created, run the next script classify.py')
    
if __name__ == '__main__':
    main()
