from nltk.tree import Tree
from mynlp.tree.regexp import MatchAllNode

def search_by_exact_string_matching(tree, string):
    """search the parent node that contains the string"""
    def aux(node):
        if isinstance(node, Tree):
            node_string = ' '.join(node.leaves())
            if node_string == string:
                return [node]
            elif string in node_string:
                matched_nodes = []
                for child in node:
                    matched_nodes.extend(aux(child))
                return matched_nodes
        return []
            
    return aux(tree)

def search_by_tree_regexp(tree, tree_regexp):
    if tree_regexp == tree:
        return [tree]
    elif isinstance(tree, Tree):
        results = []
        for child in tree:
            results.extend(search_by_tree_regexp(child, tree_regexp))
        return results
    else:
        return []

def findall_by_tree_regexp(tree, tree_regexp):
    def aux(tree, regexp, results):
        if isinstance(regexp, MatchAllNode):
            results.append(tree)
        elif isinstance(tree, Tree):
            for child, regexp_child in zip(tree, regexp):
                aux(child, regexp_child, results)

    results = []
    for t in search_by_tree_regexp(tree, tree_regexp):
        aux(t, tree_regexp, results)
                
    return results
