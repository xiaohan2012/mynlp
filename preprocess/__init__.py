import sys

DIGITALIZE = "digit"
LOWERCASE = "lower"

ALL_PIPELINE_NAMES = [DIGITALIZE, LOWERCASE]
DEFAULT_PIPELINE_NAMES = [DIGITALIZE, LOWERCASE]

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
        if name == DIGITALIZE:
            from digit import Digitalizer
            obj = Digitalizer()
        elif name == LOWERCASE:
            from case import Lowercaser
            obj = Lowercaser()
        else:
            raise ValueError("Invalid pipeline name")
        
        _pipelines[name] = obj
        sys.stderr.write("%s loaded\n" %(name))
        return obj
    else:
        return _pipelines[name]
    
def transform(stuff, pipelines = DEFAULT_PIPELINE_NAMES):
    """
    
    >>> transform(["A1", "a12", "C3"], pipelines = ["lower", "digit"])
    ['aDIGIT', 'aDIGITDIGIT', 'cDIGIT']
    >>> transform("A1 a12 C3", pipelines = ["lower", "digit"])
    'aDIGIT aDIGITDIGIT cDIGIT'
    """
    global _pipelines
    for name in pipelines:
        p = load_pipeline(name)
        stuff = p.transform(stuff)
    return stuff

    
