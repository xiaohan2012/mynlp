import sys

TOKENIZE = "tokenize"
DIGITALIZE = "digit"
LOWERCASE = "lower"
WORD_LENGTH_FILTER = "wordlen"
STEM = "stem"

ALL_PIPELINE_NAMES = [TOKENIZE, WORD_LENGTH_FILTER,
                      DIGITALIZE, LOWERCASE, STEM]
DEFAULT_PIPELINE_NAMES = [TOKENIZE, WORD_LENGTH_FILTER,
                          DIGITALIZE, LOWERCASE]

_pipelines = {p: None
              for p in ALL_PIPELINE_NAMES}


def is_loaded(name):
    global _pipelines
    return _pipelines[name] is not None


def load_pipeline(name):
    """
    >>> is_loaded('digit')
    False
    >>> load_pipeline("digit") # doctest: +ELLIPSIS
    <digit.Digitalizer object at ...>
    >>> is_loaded("digit")
    True
    """
    global _pipelines
    if not is_loaded(name):
        if name == TOKENIZE:
            from tok import Tokenizer
            obj = Tokenizer()
        elif name == DIGITALIZE:
            from digit import Digitalizer
            obj = Digitalizer()
        elif name == LOWERCASE:
            from case import Lowercaser
            obj = Lowercaser()
        elif name == STEM:
            from stem import Stemmer
            obj = Stemmer()
        elif name == WORD_LENGTH_FILTER:
            from word_len import WordLengthFilter
            obj = WordLengthFilter()
        else:
            raise ValueError("Invalid pipeline name '%s'" % (name))
        
        _pipelines[name] = obj
        sys.stderr.write("%s loaded\n" % (name))
        return obj
    else:
        return _pipelines[name]

    
def transform(stuff, pipelines=DEFAULT_PIPELINE_NAMES):
    """
    >>> transform(["A1", "a12", "C3", "a"],
    ... pipelines = ["lower", "wordlen", "digit"])
    ['aDIGIT', 'aDIGITDIGIT', 'cDIGIT']
    >>> transform("A1 a12 C3", pipelines = ["lower", "digit"])
    'aDIGIT aDIGITDIGIT cDIGIT'
    >>> transform("Giving it to me generously", pipelines = ["tokenize", "stem"])
    [u'give', 'it', 'to', 'me', u'generous']
    """
    global _pipelines
    for name in pipelines:
        p = load_pipeline(name)
        stuff = p.transform(stuff)
    return stuff

    
