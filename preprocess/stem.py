from base import Base
from nltk.stem.snowball import SnowballStemmer

class Stemmer(Base):
    """
    Stem the input

    >>> s = Stemmer()
    >>> s.transform(["giving", "it", "to", "me", "generously"])
    [u'give', 'it', 'to', 'me', u'generous']
    """
    def __init__(self):
        self.stemmer = SnowballStemmer("english")
        
    def list_transform(self, sent):
        return map(lambda w: self.stemmer.stem(w), sent)
