import re
from base import Base


class Digitalizer(Base):
    """
    Transform digit to certain token
    
    >>> d = Digitalizer('EA')
    >>> d.transform(['12', 'feng12', '1feng2', '12feng'])
    ['EAEA', 'fengEAEA', 'EAfengEA', 'EAEAfeng']
    >>> d.transform('12 feng12 1feng2 12feng')
    'EAEA fengEAEA EAfengEA EAEAfeng'
    """
    _regexp = re.compile(r"\d")
    
    def __init__(self, replacement = "DIGIT"):
        self.replacement = replacement

    def string_transform(self, sent):
        return Digitalizer._regexp.sub(self.replacement, sent)

    def list_transform(self, sent):
        return map(lambda w: Digitalizer._regexp.sub(self.replacement, w), sent)
 
