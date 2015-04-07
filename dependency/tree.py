import networkx as nx
from collections import Counter
from pathlib import Path
import cPickle as pickle
from mynlp.ling_util.bracket import convert_bracket_for_token
from mynlp.dependency.stanford_output_parser import parse_output

class NodeNotFoundError(Exception):
    pass

class DependencyTree(object):
    """
    Data wrapper for dependency tree in format of graph as well as the edge to label mapping

    >>> from mynlp.dependency.stanford_output_parser import (parse_output, Node)
    >>> t = DependencyTree.from_path('test_data/depparse_output1.out').next()
    >>> t.get_node('Objectives', 1)
    Objectives(NNS)-1
    >>> len(t.tokens())
    25
    >>> t.tokens()
    ['Objectives', 'of', 'AL', 'QAEDA', ':', 'Support', 'God', "'s", 'religion', ',', 'establishment', 'of', 'Islamic', 'rule', ',', 'and', 'restoration', 'of', 'the', 'Islamic', 'Caliphate', ',', 'God', 'willing', '.']
    >>> t.nl2n
    {(Objectives(NNS)-1, 'dep'): religion(NN)-9, (religion(NN)-9, 'poss'): God(NNP)-7, (Caliphate(NN)-21, 'det'): the(DT)-19, (Objectives(NNS)-1, 'prep_of'): QAEDA(NNP)-4, (Caliphate(NN)-21, 'amod'): Islamic(JJ)-20, (restoration(NN)-17, 'prep_of'): Caliphate(NN)-21, (God(NNP)-7, 'nn'): Support(NN)-6, (religion(NN)-9, 'punct'): ,(,)-10, (QAEDA(NNP)-4, 'nn'): AL(NNP)-3, (Objectives(NNS)-1, 'punct'): :(:)-5, (God(NNP)-23, 'amod'): willing(JJ)-24, (rule(NN)-14, 'amod'): Islamic(JJ)-13, (religion(NN)-9, 'conj_and'): establishment(NN)-11, (establishment(NN)-11, 'prep_of'): rule(NN)-14, (religion(NN)-9, 'appos'): God(NNP)-23, (ROOT-0, 'root'): Objectives(NNS)-1}
    >>> t.get_node_by_path_labels(t.get_node('Objectives', 1), ('dep', 'conj_and', 'prep_of'))
    rule(NN)-14
    >>> t.get_node_by_path_labels(t.get_node('Objectives', 1), ('dep', 'conj_and', 'punct'))   
    Traceback (most recent call last):
    NodeNotFoundError: node not found under path ('dep', 'conj_and', 'punct') from Objectives(NNS)-1

    >>> objective = Node('Objectives', 1, 'NNS')
    >>> the = Node('the', 19, 'DT')
    >>> t.get_path(objective, the)
    ('dep', 'conj_and', 'prep_of', 'det')
    >>> print t.get_path(the, objective)
    None
    >>> print t.get_path(the, Node('random_node', 10001, 'WQR'))
    None

    # children test
    >>> religion = t.get_node('religion', 9)
    >>> t.children(religion)
    [restoration(NN)-17, God(NNP)-23, ,(,)-22, God(NNP)-7, ,(,)-15, ,(,)-10, establishment(NN)-11]
    >>> t.children_by_edge_labels(religion, ('poss', 'conj_and'))
    [restoration(NN)-17, God(NNP)-7, establishment(NN)-11]
    """
    def __init__(self, g, e2l, all_nodes):
        self.g = g
        #edge to label
        self.e2l = e2l
        #source node + edge label => target node
        self.nl2n = {}
        for (s,e), l in self.e2l.items():
            self.nl2n[(s,l)] = e

        self.n2nl = {}
        for (s,e), l in self.e2l.items():
            self.n2nl[s] = (e, l)
            
        self.all_nodes = all_nodes
        self.wp2node = {(convert_bracket_for_token(n.token), n.index): n
                        for n in all_nodes}
        
    @classmethod
    def from_path(cls, path):
        for o in  parse_output(open(path)):
            yield cls.to_graph(o.nodes, o.edges)

    @classmethod
    def to_graph(cls, nodes, edges):
        """
        Convert the dependency tree in format of nodes and eedges to a networkx directed graph
        as well as the edge to label mapping(as networkx graph cannot handle edge label)
        
        >>> from mynlp.dependency.stanford_output_parser import parse_output
        >>> o = parse_output(open('test_data/depparse_output1.out')).next()
        >>> t = DependencyTree.to_graph(o.nodes, o.edges)
        >>> len(t.g.nodes()) # preposition words such as in, of are omitted
        21
        >>> t.g.nodes()
        [QAEDA(NNP)-4, AL(NNP)-3, Islamic(JJ)-13, rule(NN)-14, Caliphate(NN)-21, the(DT)-19, :(:)-5, .(.)-25, restoration(NN)-17, ,(,)-22, Support(NN)-6, God(NNP)-7, ,(,)-15, ,(,)-10, establishment(NN)-11, religion(NN)-9, ROOT-0, God(NNP)-23, Objectives(NNS)-1, Islamic(JJ)-20, willing(JJ)-24]
        >>> len(t.g.edges())
        20
        >>> t.g.edges()
        [(QAEDA(NNP)-4, AL(NNP)-3), (rule(NN)-14, Islamic(JJ)-13), (Caliphate(NN)-21, the(DT)-19), (Caliphate(NN)-21, Islamic(JJ)-20), (restoration(NN)-17, Caliphate(NN)-21), (God(NNP)-7, Support(NN)-6), (establishment(NN)-11, rule(NN)-14), (religion(NN)-9, restoration(NN)-17), (religion(NN)-9, God(NNP)-23), (religion(NN)-9, ,(,)-22), (religion(NN)-9, God(NNP)-7), (religion(NN)-9, ,(,)-15), (religion(NN)-9, ,(,)-10), (religion(NN)-9, establishment(NN)-11), (ROOT-0, Objectives(NNS)-1), (God(NNP)-23, willing(JJ)-24), (Objectives(NNS)-1, QAEDA(NNP)-4), (Objectives(NNS)-1, :(:)-5), (Objectives(NNS)-1, .(.)-25), (Objectives(NNS)-1, religion(NN)-9)]
        >>> t.e2l    
        {(QAEDA(NNP)-4, AL(NNP)-3): 'nn', (Objectives(NNS)-1, QAEDA(NNP)-4): 'prep_of', (religion(NN)-9, ,(,)-15): 'punct', (religion(NN)-9, restoration(NN)-17): 'conj_and', (religion(NN)-9, God(NNP)-7): 'poss', (restoration(NN)-17, Caliphate(NN)-21): 'prep_of', (Objectives(NNS)-1, .(.)-25): 'punct', (establishment(NN)-11, rule(NN)-14): 'prep_of', (God(NNP)-23, willing(JJ)-24): 'amod', (Objectives(NNS)-1, religion(NN)-9): 'dep', (Caliphate(NN)-21, Islamic(JJ)-20): 'amod', (ROOT-0, Objectives(NNS)-1): 'root', (religion(NN)-9, ,(,)-22): 'punct', (religion(NN)-9, God(NNP)-23): 'appos', (religion(NN)-9, establishment(NN)-11): 'conj_and', (rule(NN)-14, Islamic(JJ)-13): 'amod', (religion(NN)-9, ,(,)-10): 'punct', (God(NNP)-7, Support(NN)-6): 'nn', (Objectives(NNS)-1, :(:)-5): 'punct', (Caliphate(NN)-21, the(DT)-19): 'det'}
        """
        g = nx.DiGraph()
        
        e2l = {}
        for f, t, l in edges:
            g.add_edge(f, t)
            e2l[(f,t)] = l
        return cls(g, e2l, nodes)
       
    def get_node_by_path_labels(self, start_node, labels):
        cur = start_node
        for l in labels:
            try:
                cur = self.nl2n[(cur, l)]
            except KeyError:
                raise NodeNotFoundError("node not found under path %r from %r" %(labels, start_node))
        return cur

    def get_node(self, token, index):
        try:
            return self.wp2node[(token, index)]
        except IndexError:
            raise NodeNotFoundError((token, index))

    def children(self, node):
        return self.g[node].keys()
        
    def children_by_edge_labels(self, node, labels):
        children = self.children(node)
        return [c for c in children 
                if self.e2l[(node, c)] in labels]

    def tokens(self):
        tokens = [convert_bracket_for_token(n.token) for n in self.all_nodes]
        if tokens[0] == 'ROOT':
            return tokens[1:]
        else:
            return tokens

    def get_path(self, src, dest):
        """Get the path from src to dest in dependency parse tree
        """
        nodes = self.g.nodes()
        if src not in nodes or dest not in nodes:
            return None

        try:
            path = nx.shortest_path(self.g, src, dest)
        except nx.NetworkXNoPath:
            return None

        labels = []
        for i in xrange(len(path)-1):    
            labels.append(self.e2l[(path[i], path[i+1])])
        return tuple(labels)
