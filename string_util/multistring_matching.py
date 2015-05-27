from mynlp.data_structure.trie import (StringMatchingTrie, InvalidTransition)

class MultiStringMatcher(object):
    """
    Greedily return all the occurrences of a list of target strings in a given string.

    If two target matched substrings overlap, return the first occurring one(so it's called greedy).

    If the occur at the same starting position, return the longest one.
    
    >>> matcher = MultiStringMatcher(['Google', 'Apple Inc.', 'Windows', 'Windows Phone'])
    >>> s = 'Google is cool and Apple Inc. cool also Windows Phone'
    >>> matcher.search(s)
    ['Google', 'Apple Inc.', 'Windows Phone']
    """
    def __init__(self, strings):
        trie = StringMatchingTrie()
        for s in strings:
            trie.add_string(s)
        self._trie = trie

    def search(self, string, 
               tokenizer=lambda s: s.split()):
        tokens = tokenizer(string)
        ans = []
        i = 0
        while i < len(tokens):
            longest_matching_token = None
            for j in xrange(i, len(tokens)):
                word = tokens[j]
                if j > i:
                    word = " " + word # prepend space
                try:
                    self._trie.proceed(word)
                except InvalidTransition:
                    self._trie.reset()
                    break

                kws = self._trie.matched_string()
                if kws:
                    longest_matching_token = kws


            if longest_matching_token:
                ans.append(longest_matching_token)
                self._trie.reset()
                i = j+1
            else:
                i += 1

        return ans
