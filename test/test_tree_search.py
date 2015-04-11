from nltk.tree import ParentedTree
from nose.tools import assert_equal
from mynlp.tree.search import (search_by_exact_string_matching, search_by_tree_regexp)
from mynlp.tree.regexp import (TreeRegexp, MatchAllNode)

def test_exact_match():
    tree = ParentedTree.fromstring('(S (NP (DT the) (JJ big) (NN cat)) (VP bit) (NP (DT a) (NN cat)))')
    node = search_by_exact_string_matching(tree, 'cat')
    assert_equal(len(node), 2)
    assert_equal(node[0], ParentedTree.fromstring('(NN cat)'))

    node = search_by_exact_string_matching(tree, 'a cat')
    assert_equal(len(node), 1)
    assert_equal(node[0], ParentedTree.fromstring('(NP (DT a) (NN cat))'))

def test_regexp():
    tree = ParentedTree.fromstring('(S (NP (DT the) (JJ big) (NN dog)) (VP bit) (NP (DT a) (NN cat)))')
    
    regexp = TreeRegexp('NP', [TreeRegexp('DT', ['the']), TreeRegexp('JJ', ['big']), TreeRegexp('NN', [MatchAllNode()])])
    nodes = search_by_tree_regexp(tree, regexp)
    assert_equal(len(nodes), 1)
    assert_equal(nodes[0], 
                 ParentedTree.fromstring('(NP (DT the) (JJ big) (NN dog))'))
    
    regexp = TreeRegexp('NN', [MatchAllNode()])
    nodes = search_by_tree_regexp(tree, regexp)
    assert_equal(len(nodes), 2)
    assert_equal(nodes[0], ParentedTree.fromstring('(NN dog)'))
    assert_equal(nodes[1], ParentedTree.fromstring('(NN cat)'))

