from base import Base

from nltk.tokenize import word_tokenize

class Tokenizer(Base):
    """
    >>> s = '''Good muffins cost $3.88 in New York.  Please buy me two of them. Thanks.'''
    >>> t = Tokenizer()
    >>> t.transform(s)
    ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.', 'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']
    """
    def string_transform(self, sent):
        return word_tokenize(sent)
