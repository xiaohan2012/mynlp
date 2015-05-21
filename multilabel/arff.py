import csv
import datetime
import io
import os
import shlex

from collections import namedtuple

COMMENT = '%'
SPECIAL = '@'
RELATION = '@relation'
ATTRIBUTE = '@attribute'
DATA = '@data'


def _str_remove_quotes(obj):
    quotes = obj[0] + obj[-1]
    if quotes in ('""', "''"):
        return str(obj[1:-1])
    else:
        return obj


def decode_java_date_format(fmt):
    # http://docs.oracle.com/javase/1.4.2/docs/api/java/text/SimpleDateFormat.html
    # http://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior
    specs = {
        'G': None,
        'y': {1: '%y', 4: '%Y'},
        'M': {1: '%m', 3: '%b', 4: '%B'},
        'w': {1: '%W'},
        'W': None,
        'D': {1: '%j'},
        'd': {1: '%d'},
        'F': {1: '%w'},
        'E': {1: '%A'},
        'a': {1: '%p'},
        'H': {1: '%H'},
        'k': None,
        'K': None,
        'h': {1: '%I'},
        'm': {1: '%M'},
        's': {1: '%S'},
        'S': {1: '%f'},
        'z': {1: '%Z'},
        'Z': {1: '%z'},
    }

    error = 'Date format "%s" is not implemented.'

    def _find_num(spec, num):
        for n in sorted(spec, reverse=True):
            if num >= n:
                return spec[n]

    def _decode(letters):
        if letters[0] in specs:
            spec = specs[letters[0]]
            if spec is None:
                raise NotImplementedError(error % letters)
            p = _find_num(spec, len(letters))
            if p is None:
                raise NotImplementedError(error % letters)
            return p
        return letters

    s = ''
    letters = ''
    skip = None
    skip_letters = set('"\'')
    for c in fmt:
        if c in skip_letters:
            if skip == c:
                skip = None
                letters = ''
                continue
            else:
                skip = c
        if skip and skip != c:
            s += c
        elif letters == '':
            letters = c
        elif c == letters[0]:
            letters += c
        else:
            s += _decode(letters)
            letters = c
    if letters:
        s += _decode(letters)
    return s


def GenerateRowBase(field_names):
    """
    Rows should behave like so:
        * list(row) should give the values in order
        * row['class'] should get the column named 'class'
        * row[i] should get the i-th column
        * row.balls should get the column named 'balls'
    """
    class Row:
        def __init__(self, *values):
            # iter access
            self._values = list(values)
            # names access
            self._data = dict(zip(field_names, self._values))
            # numbered order access
            self._data.update(enumerate(self._values))

        def __getattr__(self, key):
            if key in self._data:
                return self._data[key]
            else:
                return object.__getattr__(self, key)

        def __getitem__(self, key):
            return self._data[key]

        def __repr__(self):
            return '<Row(%s)>' % ','.join([repr(i) for i in self._values])

        def __iter__(self):
            return iter(self._values)

        def __len__(self):
            return len(self._values)

    return Row


class _ParsedNominal:
    '''Parses and validates the arff enum'''
    def __init__(self, name, type_text):
        self.name = name
        self.type_text = type_text
        values_str = type_text.strip('{} ')
        self.enum = _csv_split(values_str)
        self.enum = [opt.strip(', \'"') for opt in self.enum]

    def parse(self, text):
        if text.strip('\'"') in self.enum:
            return text
        else:
            raise ValueError("'%s' is not in {%s}" % (text, self.enum))


class _SimpleTypeParser(object):
    def __init__(self, name, type_text, type_args):
        self.name = name
        self.type_text = type_text
        self.type = ARFF_TYPES[type_text]
        self.args = type_args

    def parse(self, text):
        return self.type(text)


class _DateTypeParser(_SimpleTypeParser):
    def __init__(self, *args, **kwargs):
        super(_DateTypeParser, self).__init__(*args, **kwargs)
        self.fmt = decode_java_date_format(self.args)

    def parse(self, text):
        return datetime.datetime.strptime(text, self.fmt)


ARFF_TYPES = {
    'numeric': float,
    'integer': int,
    'real': float,
    'string': _str_remove_quotes,
    'date': _DateTypeParser,
}

PYTHON_TYPES = {
    float: 'real',
    int: 'integer',
    str: 'string',
    bool: '{True, False}',
    datetime.datetime: 'date "yyyy-MM-dd\'T\'HH:mm:ss"',
}

DEFAULT_REPRS = {
    datetime.datetime: lambda x: x.isoformat()
}

try:
    import numpy
    PYTHON_TYPES[numpy.float64] = 'real'
    PYTHON_TYPES[numpy.int64] = 'integer'
except ImportError:
    pass

try:
    PYTHON_TYPES[long] = 'integer'
    DEFAULT_REPRS[long] = str
except NameError:
    pass

try:
    PYTHON_TYPES[unicode] = 'string'
    DEFAULT_REPRS[unicode] = lambda x: x.encode('utf-8')
except NameError:
    pass


# python2/3 compatible unicode
def _u(text):
    if str == bytes:
        return text.decode('utf-8')
    else:
        # python 3
        return text


def _csv_split(line):
    return next(csv.reader([line]))


def _sparse_split(line):
    lc = ''
    key = val = ''
    split = []
    quote = None
    quotes = set('"\'')
    delim = ' '

    def append(delim, key, val):
        if delim == ' ':
            return ',', val, ''
        else:
            split.append((key, val))
            return ' ', '', ''

    for c in line:
        if quote is None and c in quotes:
            if lc != c:
                val += c
            quote = c
        elif quote:
            val += c
            if quote == c:
                quote = None
        elif c == delim and val:
            delim, key, val = append(delim, key, val)
        elif c != ' ':
            val += c
        lc = c
    append(',', key, val)
    return split


class _RowParser:
    def __init__(self, fields):
        self.fields = fields
        #self.tuple = namedtuple('Row', [f.name for f in fields])
        self.rowgen = GenerateRowBase([f.name for f in fields])

    def parse(self, row):
        values = []
        for f, item in zip(self.fields, row):
            if item.strip() == '?':
                values.append(None)
            else:
                values.append(f.parse(item))

        return self.rowgen(*values)


def loads(text):
    if bytes == str:
        if not isinstance(text, unicode):
            raise ValueError('arff.loads works with unicode strings only')
    else:
        if not isinstance(text, str):
            raise ValueError('arff.loads works with strings only')
    lines_iterator = io.StringIO(text)
    for item in Reader(lines_iterator):
        yield item


def load(fname):
    with open(fname, 'r') as fhand:
        for item in Reader(fhand):
            yield item


class Reader(object):
    def __init__(self, lines_iterator):
        self.lines_iterator = lines_iterator
        self.arfftypes = dict(ARFF_TYPES)

    def __iter__(self):
        lines_iterator = self.lines_iterator
        fields = []

        for line in lines_iterator:
            if line.startswith(COMMENT):
                continue

            if line.lower().startswith(DATA):
                break

            if line.lower().startswith(RELATION):
                _, relation = line.split()
                self.relation = relation

            if line.lower().startswith(ATTRIBUTE):
                space_separated = line.split(' ', 2)
                name = space_separated[1]
                field_type_text = space_separated[2].strip()
                fields.append(self._field_type(name, field_type_text))

        self.fields = fields

        # data
        row_parser = _RowParser(fields)
        for line in lines_iterator:
            if line.startswith(COMMENT):
                continue
            if line.startswith('{'):
                row = self._sparse_split(line)
            else:
                row = _csv_split(line)
            typed_row = row_parser.parse(row)
            yield typed_row

    def _sparse_split(self, line):
        row = []
        line = line.strip().lstrip('{').rstrip('}')
        values = dict(_sparse_split(line))
        for i in range(len(self.fields)):
            row.append(values.get(str(i), '?'))
        return row


    def _field_type(self, name, type_text):
        if type_text.startswith('{'):
            return _ParsedNominal(name, type_text)

        if ' ' in type_text:
            type_text, type_args = type_text.split(' ', 2)
        else:
            type_args = None

        type_text = type_text.strip()
        if type_text in self.arfftypes:
            at = self.arfftypes[type_text]
            if isinstance(at, type) and issubclass(at, _SimpleTypeParser):
                TypeParser = at
            else:
                TypeParser = _SimpleTypeParser
            return TypeParser(name, type_text, type_args)

        raise ValueError("Unrecognized attribute type: %s" % type_text)


def _convert_row(row):
    items = [repr(item) for item in row]
    return ','.join(items)


def dumps(*args, **kwargs):
    items = []
    rows_gen = (row for row in dump_lines(*args, **kwargs))
    return _u(os.linesep).join(rows_gen)


def dump_lines(row_iterator, relation='untitled', names=None):
    w = _LineWriter(relation, names)
    for row in row_iterator:
        for line in w.generate_lines(row):
            yield line


def dump(fname, row_iterator, relation='untitled', names=None):
    w = Writer(fname, relation, names)
    for row in row_iterator:
        w.write(row)
    w.close()


class _LineWriter:
    def __init__(self, relation='untitled', names=None):
        self.relation = relation
        self.names = names
        self._first_row = True
        self.pytypes = dict(PYTHON_TYPES)

    def generate_header(self, row):
        yield "%s %s" % (RELATION, self.relation)

        if self.names is None:
            self.names = ['attr%d' % i for i in range(len(row))]

        for name, item in zip(self.names, row):
            if isinstance(name, (tuple, list)):
                name, ftype = name
            else:
                item_type = type(item)
                if item_type not in self.pytypes:
                    raise ValueError("Unknown type: %s" % item_type)
                ftype = self.pytypes[item_type]

            yield "%s %s %s" % (ATTRIBUTE, name, ftype)

        yield DATA

    def generate_lines(self, row):
        if self._first_row:
            self._first_row = False
            for line in self.generate_header(row):
                yield line

        yield self._convert_row(row)

    def _convert_obj(self, obj):
        if obj is None:
            return '?'
        typ = type(obj)
        if typ in DEFAULT_REPRS:
            return DEFAULT_REPRS[typ](obj)
        else:
            return repr(obj)

    def _convert_row(self, row):
        items = [self._convert_obj(item) for item in row]
        return ','.join(items)


class Writer(_LineWriter):
    def __init__(self, fname, relation='untitled', names=None):
        self.fhand = open(fname, 'wb')
        _LineWriter.__init__(self, relation, names)

    def write(self, row):
        for line in self.generate_lines(row):
            line = line + os.linesep
            self.fhand.write(line.encode('utf-8'))

    def close(self):
        self.fhand.close()
