import codecs 
import networkx as nx
import cPickle as pickle

from collections import Counter
from pathlib import Path

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
    Objectives-1
    >>> len(t.tokens())
    25
    >>> t.tokens()
    [u'Objectives', u'of', u'AL', u'QAEDA', u':', u'Support', u'God', u"'s", u'religion', u',', u'establishment', u'of', u'Islamic', u'rule', u',', u'and', u'restoration', u'of', u'the', u'Islamic', u'Caliphate', u',', u'God', u'willing', u'.']
    >>> t.nl2n
    {(restoration-17, u'prep_of'): Caliphate-21, (Objectives-1, u'prep_of'): QAEDA-4, (Caliphate-21, u'amod'): Islamic-20, (establishment-11, u'prep_of'): rule-14, (Caliphate-21, u'det'): the-19, (religion-9, u'punct'): ,-10, (Objectives-1, u'dep'): religion-9, (God-7, u'nn'): Support-6, (Objectives-1, u'punct'): .-25, (rule-14, u'amod'): Islamic-13, (religion-9, u'appos'): God-23, (religion-9, u'poss'): God-7, (religion-9, u'conj_and'): establishment-11, (God-23, u'amod'): willing-24, (ROOT-0, u'root'): Objectives-1, (QAEDA-4, u'nn'): AL-3}
    >>> t.get_node_by_path_labels(t.get_node('Objectives', 1), ('dep', 'conj_and', 'prep_of'))
    rule-14
    >>> t.get_node_by_path_labels(t.get_node('Objectives', 1), ('dep', 'conj_and', 'punct'))   
    Traceback (most recent call last):
    NodeNotFoundError: node not found under path ('dep', 'conj_and', 'punct') from Objectives-1

    >>> objective = Node('Objectives', 1, 'NNS')
    >>> the = Node('the', 19, 'DT')
    >>> t.get_path(objective, the)
    (u'dep', u'conj_and', u'prep_of', u'det')
    >>> print t.get_path(the, objective)
    None
    >>> print t.get_path(the, Node('random_node', 10001, 'WQR'))
    None

    # children test
    >>> religion = t.get_node('religion', 9)
    >>> t.children(religion)
    [establishment-11, restoration-17, ,-10, God-7, ,-22, ,-15, God-23]
    >>> t.children_by_edge_labels(religion, ('poss', 'conj_and'))
    [establishment-11, restoration-17, God-7]
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
        self.nodes = all_nodes
        self.wp2node = {(convert_bracket_for_token(n.token), n.index): n
                        for n in all_nodes}
        
    @classmethod
    def from_path(cls, path):
        for o in  parse_output(codecs.open(path, 'r', 'utf8')):
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
        [establishment-11, Support-6, rule-14, .-25, religion-9, ,-10, God-7, Islamic-13, ,-15, Islamic-20, restoration-17, God-23, willing-24, the-19, AL-3, ROOT-0, Objectives-1, Caliphate-21, ,-22, QAEDA-4, :-5]
        >>> len(t.g.edges())
        20
        >>> t.g.edges()
        [(establishment-11, rule-14), (rule-14, Islamic-13), (religion-9, establishment-11), (religion-9, restoration-17), (religion-9, ,-10), (religion-9, God-7), (religion-9, ,-22), (religion-9, ,-15), (religion-9, God-23), (God-7, Support-6), (restoration-17, Caliphate-21), (God-23, willing-24), (ROOT-0, Objectives-1), (Objectives-1, religion-9), (Objectives-1, .-25), (Objectives-1, QAEDA-4), (Objectives-1, :-5), (Caliphate-21, the-19), (Caliphate-21, Islamic-20), (QAEDA-4, AL-3)]
        >>> t.e2l    
        {(Objectives-1, :-5): 'punct', (religion-9, restoration-17): 'conj_and', (Objectives-1, .-25): 'punct', (God-23, willing-24): 'amod', (Objectives-1, religion-9): 'dep', (religion-9, ,-22): 'punct', (QAEDA-4, AL-3): 'nn', (religion-9, ,-15): 'punct', (ROOT-0, Objectives-1): 'root', (Caliphate-21, Islamic-20): 'amod', (establishment-11, rule-14): 'prep_of', (Objectives-1, QAEDA-4): 'prep_of', (religion-9, God-23): 'appos', (religion-9, ,-10): 'punct', (God-7, Support-6): 'nn', (religion-9, establishment-11): 'conj_and', (religion-9, God-7): 'poss', (restoration-17, Caliphate-21): 'prep_of', (Caliphate-21, the-19): 'det', (rule-14, Islamic-13): 'amod'}
        """
        g = nx.DiGraph()
        
        e2l = {}
        for f, t, l in edges:
            g.add_edge(f, t)
            e2l[(f,t)] = l
        return cls(g, e2l, nodes)
       
    def get_node_by_path_labels(self, start_node, labels):
        # TODO: label path can lead to multiple ending nodes
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
