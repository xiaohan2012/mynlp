from nltk.tree import (ImmutableParentedTree, ParentedTree)
from nose.tools import assert_equal
from mynlp.tree.regexp import (TreeRegexp, MatchAllNode)
from mynlp.tree.pattern_induction import produce_patterns

def test_single_word_case():
    tree = ImmutableParentedTree.fromstring("""(ROOT
  (S
    (NP
      (NP (JJ Strong) (NN programming) (NNS skills))
      (PP (IN in)
        (NP (NNP Java))))
    (VP (VBP are)
      (VP (VBN required)))
    (. .)))
    """)
    words = ["Java"]
    expected = TreeRegexp('NP', 
                          [TreeRegexp('NP', 
                                      [TreeRegexp('JJ', ['Strong']), TreeRegexp('NN', ['programming']), TreeRegexp('NNS', ['skills'])]), 
                           TreeRegexp('PP', [TreeRegexp('IN', ['in']), TreeRegexp('NP', [MatchAllNode()])])])

    actual = produce_patterns(tree, words)
    assert_equal(len(actual), 1)
    assert_equal(actual[0], expected)

def test_parallel_words():
    tree = ImmutableParentedTree.fromstring("""
(ROOT
  (NP
    (NP
      (NP
        (QP (IN At) (JJS least) (CD 3))
        (NNS years) (NN experience))
      (PP (IN in)
        (NP (NN programming))))
    (PP (IN in)
      (NP (NNP Java)
        (CC and)
        (NNP C++)))))
    """)
    words = ["Java", "C++", "Python", "Closure"]
    expected = TreeRegexp('NP', [TreeRegexp('NP', [TreeRegexp('NP', [TreeRegexp('QP', [TreeRegexp('IN', ['At']), TreeRegexp('JJS', ['least']), TreeRegexp('CD', ['3'])]), TreeRegexp('NNS', ['years']), TreeRegexp('NN', ['experience'])]), TreeRegexp('PP', [TreeRegexp('IN', ['in']), TreeRegexp('NP', [TreeRegexp('NN', ['programming'])])])]), TreeRegexp('PP', [TreeRegexp('IN', ['in']), TreeRegexp('NP', [MatchAllNode(), TreeRegexp('CC', ['and']), MatchAllNode()])])])
    actual = produce_patterns(tree, words)
    assert_equal(len(actual), 1)
    assert_equal(actual[0], expected)
