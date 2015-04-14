import simplejson
import sys
import codecs
import cPickle as pickle
from pathlib import Path

from mynlp.preprocess.tok import Tokenizer
from mynlp.data_structure.trie import (InvalidTransition, MultiSetTrie)

tokenizer = Tokenizer()

class FreebaseAnnotator(object):
    """Annotate the entities in the sentence
    >>> cur_dir = str(Path(__file__).resolve().parent.parent)
    >>> FREEBASE_DUMP_PATH = cur_dir + '/test/data/freebase.dump'
    >>> FREEBASE_RAW_PATH = cur_dir + '/test/data/freebase'
    >>> annotator = FreebaseAnnotator(FREEBASE_DUMP_PATH, FREEBASE_RAW_PATH)
    >>> anns = annotator.annotate('Viacom Inc. and Apple Computers released a new phone .'.split())
    >>> len(anns)
    3
    >>> import os
    >>> os.remove(FREEBASE_DUMP_PATH)
    """
    def __init__(self, kb_path, kb_raw_path=None):
        if not Path(kb_path).exists():
            if not Path(kb_raw_path).exists():
                raise IOError('%s does not exist' % kb_raw_path)
            else:
                sys.stderr.write('processing raw knowledge base\n')
                obj = self._process_kb_raw(kb_raw_path)
                pickle.dump(obj, open(kb_path, 'w'))
            self._trie = obj
        else:
            sys.stderr.write('loading knowledge base\n')
            self._trie = pickle.load(open(kb_path))
        
    def _process_kb_raw(self, path):
        self._trie = MultiSetTrie()
        with codecs.open(path, 'r', 'utf8') as f:
            for l in f:
                item = simplejson.loads(l)
                paths = [alias.split()
                         for alias in item["/common/topic/alias"]]
                values = [item] * len(paths)
                self._trie.add_paths(paths, values)
        return self._trie
        
    def annotate(self, tokens):
        """
        Input: 
        tokens: list of string

        Return:
        the annotations: list of ((start_token_index, end_token_index), {annotation info})

        """
        ans = []
        for i in xrange(len(tokens)):
            for j in xrange(i, len(tokens)):
                try:
                    self._trie.take(tokens[j])
                except InvalidTransition:
                    self._trie.reset()
                    break
                values = self._trie.terminal_values()
                if values:
                    for value in values:
                        ans.append(((i,j), value))
        
        return ans


