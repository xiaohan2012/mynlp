from base import Base

class Lowercaser(Base):
    """
    Make the input into lowercase form

    >>> l = Lowercaser()
    >>> l.transform(["A", "bc", "De"])
    ['a', 'bc', 'de']
    >>> l.transform("AbcDe")
    'abcde'
    """
    def string_transform(self, sent):
        return sent.lower()
        
    def list_transform(self, sent):
        return map(lambda w: w.lower(), sent)
