from nltk.tree import Tree

def search_by_exact_string_matching(tree, string):
    """search the parent node that contains the string"""
    def aux(node):
        node_string = ' '.join(node.leaves())
        if node_string == string:
            return [node]
        elif string in node_string:
            matched_nodes = []
            for child in node:
                matched_nodes.extend(aux(child))
            return matched_nodes
        else:
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
