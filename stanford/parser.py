import nltk
from nltk.parse.stanford import StanfordParser


nltk.internals.config_java("/cs/fs/home/hxiao/software/jre1.8.0_31/bin/java")
nltk.internals.config_java(options='-Xmx4096M')


class ConstituencyParser(StanfordParser):
    """
    Wrapper over nltk.parse.stanford.StanfordParser
    
    >>> parser = ConstituencyParser()
    >>> list(parser.raw_parse("the quick brown fox jumps over the lazy dog"))
    [Tree('ROOT', [Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['quick']), Tree('JJ', ['brown']), Tree('NN', ['fox'])]), Tree('NP', [Tree('NP', [Tree('NNS', ['jumps'])]), Tree('PP', [Tree('IN', ['over']), Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['lazy']), Tree('NN', ['dog'])])])])])])]
    """
    def __init__(self,
                 path_to_jar = "/cs/fs/home/hxiao/code/stanford-parser-full-2015-01-30/stanford-parser.jar",
                 path_to_models_jar = "/cs/fs/home/hxiao/code/stanford-parser-full-2015-01-30/stanford-parser-3.5.1-models.jar",
                 model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz", **kwargs):
        super(ConstituencyParser, self).__init__(path_to_jar, path_to_models_jar,
                                                 model_path, **kwargs)

