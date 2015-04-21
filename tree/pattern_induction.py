# Induce patterns given tree and nodes
from nltk.tree import Tree
from search import search_by_exact_string_matching
from regexp import (MatchAllNode, TreeRegexp)

def produce_patterns(tree, words):
    matched_nodes = set()
    for word in words:
        matched_nodes |= set(search_by_exact_string_matching(tree, word))

    def subs_node(node, matched_nodes, anchor_nodes):
        """substitute the matched_ndoes with MatchAllNode
        """
        if isinstance(node, Tree):
            if node in matched_nodes:
                new_node = MatchAllNode()
                anchor_nodes.append(new_node)
                return new_node
            else:
                return TreeRegexp(node.label(), [subs_node(n, matched_nodes, anchor_nodes) for n in node])
        else:
            return node
        
    anchor_nodes = []
    subs_node(tree, matched_nodes, anchor_nodes)

    patterns = set()
    for node in anchor_nodes:
        def noun_node_count(node):
            return len(list(node.subtrees(lambda n: hasattr(n, 'label') and node!=n and n.label().startswith('N'))))
            
        this_noun_count = noun_node_count(node)
        current = node
        
        # get the lowest parent node that has
        # 1. label N*
        # 2. more N* nodes than that the anchor nodes have
        while True:
            parent = current.parent()
            if parent:
                parent_noun_count = noun_node_count(parent)
                if parent_noun_count > this_noun_count:
                    patterns.add(parent)
                    break
                current = parent
            else:
                break
        
    return list(patterns)
