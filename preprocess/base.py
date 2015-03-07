class Base(object):
    """
    Base class for preprocessor
    """
    def string_transform(self, sent):
        raise NotImplementedError

    def list_transform(self, sent):
        raise NotImplementedError
        
    def transform(self, sent, *args, **kwargs):
        if isinstance(sent, basestring):
            return self.string_transform(sent, *args, **kwargs)
        elif isinstance(sent, list):
            return self.list_transform(sent, *args, **kwargs)
        else:
            raise ValueError("Accept only list and string for now")
