from collections import defaultdict
import operator

class InvalidTransition(Exception):
    pass

class PathExist(Exception):
    pass
    
class Trie(object):
    """
    Trie implementation using dictionary

    >>> t = Trie()
    >>> t.add_paths(['bar', 'baz', 'barz'], last_value_func = lambda o: "_end_")
    >>> t.add_paths(['bar'], last_value_func = lambda o: "_end_")# doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    PathExist: Path 'bar' exist
    >>> t.valid_input()
    ['b']
    >>> t.add_paths(['foo'], last_value_func = lambda o: "_end_")
    >>> t.valid_input()
    ['b', 'f']
    >>> t.take('b')
    >>> print t.terminal_values()
    None
    >>> t.take('a')
    >>> t.take('r')
    >>> t.matching_paths()
    [(['b', 'a', 'r'], '_end_'), (['b', 'a', 'r', 'z'], '_end_')]
    >>> t.terminal_values()
    '_end_'
    >>> t.take('a') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    InvalidTransition: "Invalid input 'a'. Valid are ['z']...
    >>> t.reset()
    >>> t.take('f')
    >>> t.matching_paths()
    [(['f', 'o', 'o'], '_end_')]
    >>> t.has_path('bar')
    True
    >>> t.has_path('ba')
    False
    >>> t.has_path('blah')
    False
    """
    
    def __init__(self, last_key = None):
        root = {}
        self.root = root
        self.last_key = last_key
        self.state = root
        self.path = []
        
    def add_paths(self, iter_list, last_value_func):
        assert callable(last_value_func)
        
        for iterable in iter_list:
            if self.has_path(iterable):
                raise PathExist("Path %r exist" %(iterable))

            cur_dict = self.root
            for o in iterable:
                cur_dict = cur_dict.setdefault(o, {})

            self._update_last_key_value(cur_dict, last_value_func(iterable))

    def _update_last_key_value(self, cur_dict, value):
        """decides how it behaves when value is appended at the leaf node"""
        cur_dict[self.last_key] = value
        
    def valid_input(self):
        return [k for k in self.state if k is not self.last_key]
        
    def take(self, key):
        if key in self.state:
            self.path.append(key)
            self.state = self.state[key]
        else:
            raise InvalidTransition("\"Invalid input %r. Valid are %r. Path are %r\"" %(key, self.valid_input(), self.path))
        
    def matching_paths(self):
        """
        Return the matching paths of the current state
        """
        def aux(state, path):
            paths = []
            for key in state:
                if key == self.last_key:
                    paths.append((path, state[key]))
                else:
                    paths += aux(state[key], path + [key])
            return paths
                
        return aux(self.state, self.path)

    def reset(self):
        self.state = self.root
        self.path = []

    def has_path(self, iterable):
        current = self.root
        for i in iterable:
            try:
                current = current[i]
            except KeyError:
                return False

        return self.last_key in current

    def __getitem__(self, key):
        return self.state[key]

    def terminal_values(self):
        return self.state.get(self.last_key)

class MultiSetTrie(Trie):
    """
    Trie implementation that allows the single path leading to multiple end node

    >>> t = MultiSetTrie()
    >>> t.add_paths(['bar', 'baz', 'barz'], values = ['bar1', 'baz', 'barz'])
    >>> t.add_paths(['bar'], values = ['bar2'])
    >>> t.add_paths(['foo'], values = ['foo'])
    >>> t.valid_input()
    ['b', 'f']
    >>> t.take('b')
    >>> t.take('a')
    >>> t.take('r')
    >>> t.terminal_values()
    ['bar1', 'bar2']
    """
    
    def __init__(self, *args, **kwargs):
        super(MultiSetTrie, self).__init__(*args, **kwargs)

    def add_paths(self, iter_list, values):
        assert len(iter_list) == len(values)
        
        for value, iterable in zip(values, iter_list):
            cur_dict = self.root
            for o in iterable:
                cur_dict = cur_dict.setdefault(o, defaultdict(list))
            cur_dict[self.last_key].append(value)

class EntityMatchingTrie(Trie):
    """
    Associate each string with a list of entities.
    
    For example: `cs` can be mapped to `computer science` or `counter strike`
    
    >>> t = EntityMatchingTrie()
    >>> t.add_mention('cs', set(['computer science', 'counter strike']))
    >>> t.add_mention('csd', set(['computer science department']))
    >>> t.add_mention('user assistance and performance support', set(['asdf']))
    >>> t.proceed('csd')
    >>> t.matched_keywords()
    set(['computer science department'])
    >>> t.reset()
    >>> t.proceed('user')
    >>> print t.matched_keywords()
    None
    """

    def add_mention(self, mention, entities):
        self.add_paths([mention], lambda o: entities)

    def proceed(self, s):
        for c in s:
            self.take(c)

    def matched_keywords(self):
        # wierd order
        return self.terminal_values()


class StringMatchingTrie(Trie):
    """
    More methods that proceed with `string` by wrapping a series of `character` state transitions.
    
    >>> t = StringMatchingTrie()
    >>> t.add_string('cs')
    >>> t.add_string('csd')
    >>> t.add_string('user assistance and performance support')
    >>> t.proceed('csd')
    >>> t.matched_string()
    'csd'
    >>> t.reset()
    >>> t.proceed('user')
    >>> print t.matched_string()
    None
    """

    def add_string(self, string):
        self.add_paths([string], lambda o: string)

    def proceed(self, s):
        for c in s:
            self.take(c)

    def matched_string(self):
        return self.terminal_values()
