from base import Base


class WordLengthFilter(Base):
    """
    Make the input into lowercase form

    >>> l = WordLengthFilter(min_len=1, max_len=3)
    >>> l.transform(["", "1", "12", "123", "1234"])
    ['1', '12', '123']
    """        
    def __init__(self, min_len=2, max_len=35):
        self.min_ = min_len
        self.max_ = max_len
        super(WordLengthFilter, self).__init__()

    def list_transform(self, tokens):
        return filter(lambda t: 
                      (len(t) >= self.min_ and
                       len(t) <= self.max_),
                      tokens)
