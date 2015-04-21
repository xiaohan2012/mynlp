from nltk.tree import (Tree, ImmutableParentedTree)

class MatchAllNode(ImmutableParentedTree):
    def __init__(self):
        super(MatchAllNode, self).__init__("*", ["*"])
        
    def __eq__(self, other):
        return True

class BaseRegexp(ImmutableParentedTree):
    pass

class TreeRegexp(BaseRegexp):
    def __eq__(self, other):
        return (isinstance(self, Tree) and 
                isinstance(other,Tree) and 
                (self._label, list(self)) == (other._label, list(other)))
