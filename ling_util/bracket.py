"""
Linguistic utility functions
"""
from nltk.tree import Tree

mapping = dict(zip('-LRB- -RRB- -RSB- -RSB- -LCB- -RCB-'.split(), '( ) [ ] { }'.split()))

def convert_bracket_for_token(tok):
    """
    >>> convert_bracket_for_token('feng')
    'feng'
    >>> convert_bracket_for_token('-LRB-')
    '('
    """
    if tok in mapping:
        return mapping[tok]
    else:
        return tok

def convert_brackets(tree):
    """convert the bracket notation back to the original

    >>> t = Tree('FRAG', [Tree('PP', [Tree('IN', ['In']), Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['name'])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('NP', [Tree('NNP', ['Allah'])]), Tree(',', [',']), Tree('NP', [Tree('JJS', ['Most']), Tree('NNS', ['Gracious'])]), Tree(',', [','])])])])]), Tree('NP', [Tree('NP', [Tree('JJS', ['Most'])]), Tree('NP', [Tree('NP', [Tree('NNP', ['Merciful']), Tree('.', ['.'])]), Tree('PRN', [Tree('-LRB-', ['-LRB-']), Tree('NP', [Tree('NP', [Tree('NNP', ['T.C'])]), Tree(':', [':']), Tree('NP', [Tree('NP', [Tree('NN', ['verse'])]), Tree('PP', [Tree('IN', ['from']), Tree('NP', [Tree('DT', ['the']), Tree('NNP', ['Koran'])])])])]), Tree('-RRB-', ['-RRB-'])])])])])
    >>> convert_brackets(t)
    Tree('FRAG', [Tree('PP', [Tree('IN', ['In']), Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['name'])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('NP', [Tree('NNP', ['Allah'])]), Tree(',', [',']), Tree('NP', [Tree('JJS', ['Most']), Tree('NNS', ['Gracious'])]), Tree(',', [','])])])])]), Tree('NP', [Tree('NP', [Tree('JJS', ['Most'])]), Tree('NP', [Tree('NP', [Tree('NNP', ['Merciful']), Tree('.', ['.'])]), Tree('PRN', [Tree('-LRB-', ['(']), Tree('NP', [Tree('NP', [Tree('NNP', ['T.C'])]), Tree(':', [':']), Tree('NP', [Tree('NP', [Tree('NN', ['verse'])]), Tree('PP', [Tree('IN', ['from']), Tree('NP', [Tree('DT', ['the']), Tree('NNP', ['Koran'])])])])]), Tree('-RRB-', [')'])])])])])
    """
    def aux(t):
        if isinstance(t, basestring):
            if t in mapping:
                return mapping[t]
            else:
                return t
        else:
            return Tree(t.label(), [convert_brackets(subtree) for subtree in t])
            
    return aux(tree)

