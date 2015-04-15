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
    >>> anns = annotator.annotate('Google and Samsung strike patent cross-licensing deal'.split())
    >>> print [a['name'] for _,a in anns]
    [u'Google', u'Samsung Electronics']
    >>> import os
    >>> os.remove(FREEBASE_DUMP_PATH)
    """
    DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent) + '/models/freebase.pkl'
    def __init__(self, kb_path = DEFAULT_MODEL_PATH, kb_raw_path=None):
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
            for i,l in enumerate(f):
                if (i+1)%1000 == 0:
                    print "%d processed" % (i+1)
                item = simplejson.loads(l)
                names = item["/common/topic/alias"] + [item["name"]]
                paths = [alias.split()
                         for alias in names]
                values = [item] * len(paths)
                self._trie.add_paths(paths, values)
        return self._trie
        
    def annotate(self, tokens):
        """
        Input: 
        tokens: list of string

        Return:
        the annotations: list of ((start_token_index, end_token_index), {annotation info})

        Note: one annotation contains another, then the longer one is kept
        """
        ans = []
        last_terminal_value = None
        for i in xrange(len(tokens)):
            for j in xrange(i, len(tokens)):
                try:
                    self._trie.take(tokens[j])
                except InvalidTransition:
                    self._trie.reset()
                    if last_terminal_value:
                        for value in last_terminal_value:
                            ans.append(((i, j-1), value))
                        last_terminal_value = None
                    break
                last_terminal_value = self._trie.terminal_values()
        return ans

if __name__ == "__main__":
    FreebaseAnnotator('models/freebase.pkl','/cs/taatto/home/hxiao/product_classification/resources/freebase')
