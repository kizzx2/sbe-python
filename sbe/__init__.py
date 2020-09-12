import enum
import struct
from dataclasses import dataclass, field
from typing import Dict, List, NewType, Optional, TextIO, Union

import lxml
import lxml.etree


class PrimitiveType(enum.Enum):
    CHAR = 'char'
    UINT8 = 'uint8'
    UINT16 = 'uint16'
    UINT32 = 'uint32'
    UINT64 = 'uint64'
    INT8 = 'int8'
    INT16 = 'int16'
    INT32 = 'int32'
    INT64 = 'int64'
    FLOAT = 'float'
    DOUBLE = 'double'


class SetEncodingType(enum.Enum):
    UINT8 = 'uint8'
    UINT16 = 'uint16'
    UINT32 = 'uint32'
    UINT64 = 'uint64'


class EnumEncodingType(enum.Enum):
    UINT8 = 'uint8'
    CHAR = 'char'


FormatString = NewType('FormatString', str)


FORMAT = dict(zip(PrimitiveType.__members__.values(), "cBHIQbhiqfd"))
FORMAT_TO_TYPE = {v: k for k, v in FORMAT.items()}
FORMAT_SIZES = {k: struct.calcsize(v) for k, v in FORMAT.items()}
PRIMITIVE_TYPES = set(x.value for x in PrimitiveType.__members__.values())
ENUM_ENCODING_TYPES = set(x.value for x in EnumEncodingType.__members__.values())
SET_ENCODING_TYPES = set(x.value for x in SetEncodingType.__members__.values())


class Presence(enum.Enum):
    CONSTANT = 'cosntant'
    REQUIRED = 'required'
    OPTIONAL = 'optional'


class CharacterEncoding(enum.Enum):
    ASCII = 'ASCII'


@dataclass
class UnpackedValue:
    value: dict
    size: int

    __slots__ = ('value', 'size')

    def __repr__(self):
        return self.value.__repr__()


@dataclass
class DecodedMessage:
    message_name: str
    header: dict
    value: dict

    __slots__ = ('message_name', 'header', 'value')


@dataclass(init=False)
class Type:
    __slots__ = (
        'name', 'primitiveType', 'presence', 'semanticType',
        'description', 'length', 'characterEncoding')

    name: str
    primitiveType: PrimitiveType
    presence: Presence
    semanticType: Optional[str]
    description: Optional[str]
    length: int
    characterEncoding: Optional[CharacterEncoding]

    def __init__(self, name: str, primitiveType: PrimitiveType):
        super().__init__()
        self.name = name
        self.primitiveType = primitiveType
        self.presence = Presence.REQUIRED
        self.length = 1

    def __repr__(self):
        rv = self.name + " ("
        rv += self.primitiveType.value
        if rv == PrimitiveType.CHAR or self.length > 1:
            rv += f"[{self.length}]"
        rv += ")"
        return rv


@dataclass
class EnumValue:
    name: str
    value: str

    __slots__ = ('name', 'value')

    def __repr__(self):
        return (self.name, self.value).__repr__()


@dataclass
class Pointer:
    offset: int
    value: Union[FormatString, Dict[str, 'Pointer']]
    size: int
    is_group: bool
    enum: Optional['Enum']

    __slots__ = ('offset', 'value', 'size', 'is_group', 'enum')

    def __init__(self, offset: int, value: Union[FormatString, Dict[str, 'Pointer']], size: int):
        self.offset = offset
        self.value = value
        self.size = size
        self.is_group = False
        self.enum = None

    def return_value(self, buf: memoryview, offset: int, _parent: Optional['WrappedComposite']):
        start = self.offset + offset
        end = start + self.size
        rv = buf[start:end].cast(self.value)[0]

        if self.enum:
            return self.enum.find_name_by_value(rv.decode("ascii") if isinstance(rv, bytes) else str(rv))
        elif self.value.endswith("s"):
            return rv.replace(b"\x00", b"").decode('ascii', errors='ignore').strip()

        return rv

    def unpack(self, buf: memoryview):
        return buf[self.offset:self.offset+self.size].cast(self.value)[0]

    def __repr__(self):
        if self.enum:
            return f"{self.enum.name}@{self.offset}"
        elif self.value in FORMAT_TO_TYPE:
            return f"{FORMAT_TO_TYPE[self.value].value}@{self.offset}"
        else:
            return f"{self.value}@{self.offset}"


@dataclass
class SetChoice:
    name: str
    value: str

    __slots__ = ('name', 'value')

    def __repr__(self):
        return (self.name, self.value).__repr__()


@dataclass
class Enum:
    name: str
    encodingType: Union[EnumEncodingType, Type]
    presence = Presence.REQUIRED
    semanticType: Optional[str] = None
    description: Optional[str] = None
    valid_values: List[EnumValue] = field(default_factory=list)

    def find_value_by_name(self, name: str) -> str:
        return int(next(x for x in self.valid_values if x.name == name).value)

    def find_name_by_value(self, val: str) -> str:
        return next(x for x in self.valid_values if x.value == val).name

    def __repr__(self):
        return f"<Enum '{self.name}'>"


@dataclass
class Composite:
    name: str
    types: List[Union['Composite', Type]] = field(default_factory=list)
    description: Optional[str] = None

    def __repr__(self):
        return f"<Composite '{self.name}'>"


@dataclass
class Set:
    name: str
    encodingType: Union[SetEncodingType, Type]
    presence = Presence.REQUIRED
    semanticType: Optional[str] = None
    description: Optional[str] = None
    choices: List[SetChoice] = field(default_factory=list)

    def __repr__(self):
        return f"<Set '{self.name}'>"


@dataclass(init=False)
class Field:
    name: str
    id: str
    type: Union[PrimitiveType, str]
    description: Optional[str]
    sinceVersion: int

    def __init__(self, name: str, id: str, type: Union[PrimitiveType, str]):
        self.name = name
        self.id = id
        self.type = type
        self.description = None
        self.sinceVersion = 0

    __slots__ = ('name', 'id', 'type', 'description', 'sinceVersion')

    def __repr__(self):
        if isinstance(self.type, PrimitiveType):
            return f"<{self.name} ({self.type.value})>"
        else:
            return f"<{self.name} ({self.type})>"


@dataclass
class Group:
    name: str
    id: str
    dimensionType: Composite
    description: Optional[str] = None
    fields: List[Union['Group', Field]] = field(default_factory=list)


@dataclass
class Message:
    name: str
    id: int
    description: Optional[str] = None
    fields: List[Union[Group, Field]] = field(default_factory=list, repr=False)


@dataclass(init=False)
class WrappedComposite:
    name: str
    pointers: Dict[str, Union[Pointer, 'WrappedComposite', 'WrappedGroup']]
    buf: Optional[memoryview]
    offset: int
    schema: Optional['Schema']
    hydrated: bool

    def __init__(
        self,
        name: str,
        pointers:  Dict[str, Union[Pointer, 'WrappedComposite', 'WrappedGroup']],
        buf: Optional[memoryview],
        offset: int,
        schema: Optional['Schema'] = None,
    ):
        self.name = name
        self.pointers = pointers
        self.buf = buf
        self.offset = offset
        self.schema = schema

    __slots__ = ('name', 'pointers', 'buf', 'offset', 'schema', 'hydrated')

    def hydrate(self, offset: int):
        cursor = offset
        for k, p in self.pointers.items():
            seen_group = False
            if isinstance(p, WrappedGroup):
                p1 = WrappedGroup(
                    self.buf, self.name, cursor, p.pointers, self,
                    p.num_in_group_pointer, p.block_length_pointer)

                p1.buf = self.buf
                p1.offset = cursor
                p1.hydrate()
                self.pointers[k] = p1
                cursor += p1.numInGroup * p1.blockLength
                seen_group = True
            else:
                assert not seen_group

    def get_raw_pointer(self, key):
        p = self.pointers[key]
        p.enum = None
        return Pointer(self.offset + p.offset, p.value, p.size)

    def __getitem__(self, key):
        return self.pointers[key].return_value(self.buf, self.offset, self)

    def __repr__(self):
        return f"<WrappedComposite '{self.name}'>"


@dataclass
class WrappedMessage:
    buf: memoryview
    header: WrappedComposite
    body: Optional[WrappedComposite]

    __slots__ = ('buf', 'header', 'body')


@dataclass(init=False)
class WrappedGroup:
    buf: Optional[memoryview]
    name: str
    offset: int
    pointers: Dict[str, Union[Pointer, 'WrappedGroup']]
    schema: Optional['Schema']

    num_in_group_pointer: Pointer
    block_length_pointer: Pointer

    numInGroup: int
    blockLength: int

    __slots__ = (
        'buf', 'name', 'offset', 'pointers', 'schema',
        'num_in_group_pointer', 'block_length_pointer', 'numInGroup',
        'blockLength'
    )

    def __init__(
        self,
        buf: Optional[memoryview],
        name: str,
        offset: int,
        pointers: Dict[str, Union[Pointer, 'WrappedGroup']],
        schema: Optional['Schema'],
        num_in_group_pointer: Pointer,
        block_length_pointer: Pointer,
    ):
        self.buf = buf
        self.name = name
        self.offset = offset
        self.pointers = pointers
        self.schema = schema
        self.num_in_group_pointer = num_in_group_pointer
        self.block_length_pointer = block_length_pointer
        self.numInGroup = 0
        self.blockLength = 0

    def return_value(self, _buf: memoryview, _offset: int, _schema: Optional['Schema']):
        return self

    def hydrate(self):
        start = self.offset + self.num_in_group_pointer.offset
        end = start + self.num_in_group_pointer.size
        self.numInGroup = self.buf[start:end].cast(self.num_in_group_pointer.value)[0]

        start = self.offset + self.block_length_pointer.offset
        end = start + self.block_length_pointer.size
        self.blockLength = self.buf[start:end].cast(self.block_length_pointer.value)[0]

    def __getitem__(self, i: int):
        offset = self.offset + self.blockLength * i
        return WrappedComposite(f"{self.name}[{i}]", self.pointers, self.buf, offset)

    def __repr__(self):
        return f"<WrappedGroup '{self.name}' numInGroup={self.numInGroup}>"

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.numInGroup


@dataclass
class Cursor:
    val: int
    __slots__ = ('val',)


@dataclass
class Schema:
    id: int
    version: int
    types: Dict[str, Union[Composite, Type]] = field(default_factory=dict)
    messages: Dict[int, Message] = field(default_factory=dict)

    header_wrapper: WrappedComposite = None
    header_size: int = None
    message_wrappers: Dict[int, WrappedComposite] = field(default_factory=dict)

    @classmethod
    def parse(cls, f: TextIO) -> 'Schema':
        rv = _parse_schema(f)
        rv.create_wrappers()
        return rv

    def encode(self, message: Message, obj: dict, header: Optional[dict] = None) -> bytes:
        if header is None:
            header = {}

        fmts = []
        vals = []
        cursor = Cursor(0)

        _walk_fields_encode(self, message.fields, obj, fmts, vals, cursor)
        fmt = "<" + ''.join(fmts)

        header = {
            **header,
            'templateId': message.id,
            'schemaId': int(self.id),
            'version': int(self.version),
            'blockLength': cursor.val,
        }

        return b''.join([
            _pack_composite(self, self.types['messageHeader'], header),
            struct.pack(fmt, *vals)
        ])

    def decode_header(self, buffer: Union[bytes, memoryview]) -> dict:
        buffer = memoryview(buffer)
        return _unpack_composite(self, self.types['messageHeader'], buffer).value

    def decode(self, buffer: Union[bytes, memoryview]) -> dict:
        buffer = memoryview(buffer)

        header = _unpack_composite(self, self.types['messageHeader'], buffer)
        body_offset = header.size

        m = self.messages[header.value['templateId']]

        cursor = Cursor(0)
        format_str_parts = []
        for f in m.fields:
            if isinstance(f, Group):
                assert cursor.val <= header.value['blockLength']
                if cursor.val < header.value['blockLength']:
                    format_str_parts.append(str(header.value['blockLength'] - cursor.val) + 'x')
                cursor.val = header.value['blockLength']
            format_str_parts.append(_unpack_format(self, f, '', buffer[body_offset:], cursor))
        format_str = '<' + ''.join(format_str_parts)

        body_size = struct.calcsize(format_str)

        rv = {}
        vals = struct.unpack(format_str, buffer[body_offset:body_offset+body_size])
        _walk_fields_decode(self, rv, m.fields, vals, Cursor(0))
        return DecodedMessage(m.name, header.value, rv)

    def wrap(self, buf: Union[bytes, memoryview], header_only=False) -> WrappedMessage:
        header = WrappedComposite('messageHeader', self.header_wrapper.pointers, buf, 0, self)
        if header_only:
            return WrappedMessage(buf, header, None)

        m = self.messages[header['templateId']]
        body = WrappedComposite(
            m.name, self.message_wrappers[header['templateId']].pointers,
            buf, self.header_size, self)
        body.hydrate(self.header_size + header['blockLength'])

        return WrappedMessage(buf, header, body)

    def create_wrappers(self) -> WrappedMessage:
        cursor = Cursor(0)

        pointers = {}
        _walk_fields_wrap_composite(self, pointers, self.types['messageHeader'], cursor)
        self.header_wrapper = WrappedComposite('messageHeader', pointers, None, 0)
        self.header_size = struct.calcsize(_unpack_format(self, self.types['messageHeader']))

        for i, m in self.messages.items():
            pointers = {}
            cursor = Cursor(0)
            _walk_fields_wrap(self, pointers, m.fields, cursor)
            self.message_wrappers[i] = WrappedComposite(m.name, pointers, None, 0)


def _unpack_format(
    schema: Schema,
    type_: Union[Field, Group, PrimitiveType, Type, Set, Enum, Composite],
    prefix='<', buffer=None, buffer_cursor=None
):
    if isinstance(type_, PrimitiveType):
        if buffer_cursor:
            buffer_cursor.val += FORMAT_SIZES[type_]
        return prefix + FORMAT[type_]

    elif isinstance(type_, Field):
        return _unpack_format(schema, type_.type, '', buffer, buffer_cursor)

    elif isinstance(type_, Group):
        dimension = _unpack_composite(schema, type_.dimensionType, buffer[buffer_cursor.val:])
        buffer_cursor.val += dimension.size

        rv = _unpack_format(schema, type_.dimensionType, '')
        for _ in range(dimension.value['numInGroup']):
            cursor0 = buffer_cursor.val
            rv += ''.join(_unpack_format(schema, f, '', buffer, buffer_cursor) for f in type_.fields)
            assert buffer_cursor.val <= cursor0 + dimension.value['blockLength']
            if buffer_cursor.val < cursor0 + dimension.value['blockLength']:
                rv += str(cursor0 + dimension.value['blockLength'] - buffer_cursor.val) + "x"
            buffer_cursor.val = cursor0 + dimension.value['blockLength']

        return rv

    elif isinstance(type_, Type):
        if type_.primitiveType == PrimitiveType.CHAR:
            if buffer_cursor:
                buffer_cursor.val += type_.length
            return prefix + f"{type_.length}s"
        else:
            if buffer_cursor:
                buffer_cursor.val += FORMAT_SIZES[type_.primitiveType]
            return prefix + FORMAT[type_.primitiveType]

    elif isinstance(type_, (Set, Enum)):
        if type_.encodingType.value in PRIMITIVE_TYPES:
            if buffer_cursor:
                buffer_cursor.val += FORMAT_SIZES[PrimitiveType(type_.encodingType.value)]
            return prefix + FORMAT[PrimitiveType(type_.encodingType.value)]
        return _unpack_format(schema, type_.encodingType, '', buffer, buffer_cursor)

    elif isinstance(type_, Composite):
        return prefix + ''.join(_unpack_format(schema, t, '', buffer, buffer_cursor) for t in type_.types)


def _pack_format(_schema: Schema, composite: Composite):
    fmt = []
    for t in composite.types:
        if t.length > 1 and t.primitiveType == PrimitiveType.CHAR:
            fmt.append(str(t.length) + 's')
        else:
            fmt.append(FORMAT[t.primitiveType])

    return ''.join(fmt)


def _pack_composite(_schema: Schema, composite: Composite, obj: dict):
    fmt = []
    vals = []
    for t in composite.types:
        v = obj.get(t.name)

        if v is None:
            if t.primitiveType == PrimitiveType.CHAR:
                if t.length > 1:
                    v = ''
                else:
                    v = '0'

            else:
                v = 0

        if t.length > 1 and t.primitiveType == PrimitiveType.CHAR:
            fmt.append(str(t.length) + 's')
            vals.append(v.encode())
        elif t.primitiveType == PrimitiveType.CHAR:
            fmt.append(FORMAT[t.primitiveType])
            vals.append(v.encode())
        else:
            fmt.append(FORMAT[t.primitiveType])
            vals.append(v)

    return struct.pack('<' + ''.join(fmt), *vals)


def _unpack_composite(schema: Schema, composite: Composite, buffer: memoryview):
    fmt = _unpack_format(schema, composite)
    size = struct.calcsize(fmt)
    vals = struct.unpack(fmt, buffer[:size])

    rv = {}
    for t, v in zip(composite.types, vals):
        rv[t.name] = _prettify_type(schema, t, v)

    return UnpackedValue(rv, size)


def _prettify_type(_schema: Schema, t: Type, v):
    if t.primitiveType == PrimitiveType.CHAR and t.characterEncoding == CharacterEncoding.ASCII:
        return v.replace(b"\x00", b"").decode("ascii", errors='ignore').strip()

    return v


def _walk_fields_encode(schema: Schema, fields: List[Union[Group, Field]], obj: dict, fmt: list, vals: list, cursor: Cursor):
    for f in fields:
        if isinstance(f, Group):
            xs = obj[f.name]

            fmt1 = []
            vals1 = []
            block_length = None
            for x in xs:
                _walk_fields_encode(schema, f.fields, x, fmt1, vals1, Cursor(0))
                if block_length is None:
                    block_length = struct.calcsize("<" + ''.join(fmt1))

            dimension = {"numInGroup": len(obj[f.name]), "blockLength": block_length or 0}
            dimension_fmt = _pack_format(schema, schema.types['groupSizeEncoding'])

            fmt.append(dimension_fmt)
            for t in schema.types['groupSizeEncoding'].types:
                vals.append(dimension[t.name])

            fmt.extend(fmt1)
            vals.extend(vals1)
            # cursor.val += struct.calcsize("<" + dimension_fmt) + block_length

        elif isinstance(f.type, Type):
            t = f.type.primitiveType
            if t == PrimitiveType.CHAR and f.type.length > 1:
                fmt.append(str(f.type.length) + "s")
                vals.append(obj[f.name].encode())
                cursor.val += f.type.length
            else:
                fmt.append(FORMAT[t])
                vals.append(obj[f.name])
                cursor.val += FORMAT_SIZES[t]

        elif isinstance(f.type, Enum):
            fmt.append(FORMAT[PrimitiveType(f.type.encodingType.value)])
            vals.append(f.type.find_value_by_name(obj[f.name]))
            cursor.val += FORMAT_SIZES[PrimitiveType(f.type.encodingType.value)]

        elif isinstance(f.type, PrimitiveType):
            fmt.append(FORMAT[f.type])
            vals.append(obj[f.name])
            cursor.val += FORMAT_SIZES[f.type]

        else:
            assert 0


def _walk_fields_wrap_composite(
    schema: Schema, rv: Dict[str, Union[Pointer, WrappedGroup]],
    composite: Composite, cursor: Cursor
):
    for t in composite.types:
        if isinstance(t, Composite):
            rv1 = {}
            offset = cursor.val
            _walk_fields_wrap_composite(schema, rv1, t, cursor)
            rv[t.name] = WrappedComposite(t.name, rv1, None, offset)

        else:
            t1 = t.primitiveType
            if t1 == PrimitiveType.CHAR and t.length > 1:
                rv[t.name] = Pointer(cursor.val, str(t.length) + "s", t.length)
                cursor.val += t.length
            else:
                rv[t.name] = Pointer(cursor.val, FORMAT[t1], FORMAT_SIZES[t1])
                cursor.val += FORMAT_SIZES[t1]


def _walk_fields_wrap(
    schema: Schema, rv: Dict[str, Union[Pointer, WrappedGroup]],
    fields: List[Union[Group, Field]], cursor: Cursor
):
    for f in fields:
        if isinstance(f, Group):
            num_in_group_offset = None
            block_length_offset = None

            cursor1 = Cursor(0)
            for t in f.dimensionType.types:
                if t.name == 'numInGroup':
                    num_in_group_offset = cursor1.val

                elif t.name == 'blockLength':
                    block_length_offset = cursor1.val

                cursor1.val += FORMAT_SIZES[t.primitiveType]

            assert num_in_group_offset is not None
            assert block_length_offset is not None

            dimensionTypeTypes = {t.name: t for t in f.dimensionType.types}

            rv1 = {}
            _walk_fields_wrap(schema, rv1, f.fields, cursor1)
            rv[f.name] = WrappedGroup(
                None, f.name, -1, rv1, None,
                Pointer(
                    num_in_group_offset,
                    FORMAT[dimensionTypeTypes['numInGroup'].primitiveType],
                    FORMAT_SIZES[dimensionTypeTypes['numInGroup'].primitiveType],
                ),
                Pointer(
                    block_length_offset,
                    FORMAT[dimensionTypeTypes['blockLength'].primitiveType],
                    FORMAT_SIZES[dimensionTypeTypes['blockLength'].primitiveType],
                ),
            )

        elif isinstance(f.type, Type):
            t = f.type.primitiveType
            if t == PrimitiveType.CHAR and f.type.length > 1:
                rv[f.name] = Pointer(cursor.val, str(f.type.length) + "s", f.type.length)
                cursor.val += f.type.length
            else:
                rv[f.name] = Pointer(cursor.val, FORMAT[t], FORMAT_SIZES[t])
                cursor.val += FORMAT_SIZES[t]

        elif isinstance(f.type, Enum):
            rv[f.name] = Pointer(
                cursor.val,
                FORMAT[PrimitiveType(f.type.encodingType.value)],
                FORMAT_SIZES[PrimitiveType(f.type.encodingType.value)],
            )
            rv[f.name].enum = f.type
            cursor.val += FORMAT_SIZES[PrimitiveType(f.type.encodingType.value)]

        elif isinstance(f.type, PrimitiveType):
            rv[f.name] = Pointer(cursor.val, FORMAT[f.type], FORMAT_SIZES[f.type])
            cursor.val += FORMAT_SIZES[f.type]

        else:
            assert 0


def _walk_fields_decode(schema: Schema, rv: dict, fields: List[Union[Group, Field]], vals: List, cursor: Cursor):
    for f in fields:
        if isinstance(f, Group):
            num_in_group = vals[cursor.val + 1]
            cursor.val += 2

            rv[f.name] = []
            for _ in range(num_in_group):
                rv1 = {}
                _walk_fields_decode(schema, rv1, f.fields, vals, cursor)
                rv[f.name].append(rv1)

        elif isinstance(f.type, Type):
            rv[f.name] = _prettify_type(schema, f.type, vals[cursor.val])
            cursor.val += 1

        elif isinstance(f.type, Enum):
            v = vals[cursor.val]
            cursor.val += 1

            if isinstance(v, bytes):
                if v == b'\x00':
                    rv[f.name] = v
                else:
                    rv[f.name] = f.type.find_name_by_value(v.decode("ascii", errors='ignore'))
            else:
                rv[f.name] = f.type.find_name_by_value(str(v))

        elif isinstance(f.type, PrimitiveType):
            v = vals[cursor.val]
            cursor.val += 1
            rv[f.name] = v

        else:
            assert 0


def _parse_schema(f: TextIO) -> Schema:
    doc = lxml.etree.parse(f)
    stack = []

    for action, elem in lxml.etree.iterwalk(doc, ("start", "end")):
        assert action in ("start", "end")

        tag = elem.tag
        if "}" in tag:
            tag = tag[tag.index("}") + 1:]

        if tag == "messageSchema":
            if action == "start":
                attrs = dict(elem.items())
                x = Schema(attrs['id'], attrs['version'])
                stack.append(x)
            elif action == "end":
                pass

        elif tag == "types":
            pass

        elif tag == "composite":
            if action == "start":
                name = next(v for k, v in elem.items() if k == 'name')
                x = Composite(name=name)
                for k, v in elem.items():
                    if k == 'name':
                        pass
                    elif k == 'description':
                        x.description = v

                stack.append(x)

            elif action == 'end':
                x = stack.pop()
                assert isinstance(stack[-1], Schema)
                stack[-1].types[x.name] = x

        elif tag == "type":
            if action == "start":
                attrs = dict(elem.items())
                x = Type(name=attrs['name'], primitiveType=PrimitiveType(attrs['primitiveType']))

                if x.primitiveType == PrimitiveType.CHAR:
                    if 'length' in attrs:
                        x.length = int(attrs['length'])
                    if 'characterEncoding' in attrs:
                        x.characterEncoding = CharacterEncoding(attrs['characterEncoding'])

                stack.append(x)

            elif action == "end":
                x = stack.pop()
                assert isinstance(stack[-1], (Composite, Schema))
                if isinstance(stack[-1], Composite):
                    stack[-1].types.append(x)
                elif isinstance(stack[-1], Schema):
                    stack[-1].types[x.name] = x

        elif tag == "enum":
            if action == "start":
                attrs = dict(elem.items())

                if attrs['encodingType'] in ENUM_ENCODING_TYPES:
                    encoding_type = EnumEncodingType(attrs['encodingType'])
                else:
                    encoding_type = stack[0].types[attrs['encodingType']]

                stack.append(Enum(name=attrs['name'], encodingType=encoding_type))

            elif action == "end":
                x = stack.pop()
                assert isinstance(stack[-1], Schema)
                stack[-1].types[x.name] = x

        elif tag == "validValue":
            if action == "start":
                attrs = dict(elem.items())
                stack.append(EnumValue(name=attrs['name'], value=elem.text.strip()))

            elif action == "end":
                x = stack.pop()
                assert isinstance(stack[-1], Enum)
                stack[-1].valid_values.append(x)

        elif tag == "set":
            if action == "start":
                attrs = dict(elem.items())

                if attrs['encodingType'] in SET_ENCODING_TYPES:
                    encoding_type = SetEncodingType(attrs['encodingType'])
                else:
                    encoding_type = stack[0].types[attrs['encodingType']]

                stack.append(Set(name=attrs['name'], encodingType=encoding_type))

            elif action == "end":
                x = stack.pop()
                assert isinstance(stack[-1], Schema)
                stack[-1].types[x.name] = x

        elif tag == "choice":
            if action == "start":
                attrs = dict(elem.items())
                stack.append(SetChoice(name=attrs['name'], value=elem.text.strip()))

            elif action == "end":
                x = stack.pop()
                assert isinstance(stack[-1], Set)
                stack[-1].choices.append(x)

        elif tag == "field":
            if action == "start":
                assert len(elem) == 0

                attrs = dict(elem.items())
                if attrs['type'] in PRIMITIVE_TYPES:
                    field_type = PrimitiveType(attrs['type'])
                else:
                    field_type = stack[0].types[attrs['type']]

                assert isinstance(stack[-1], (Group, Message))
                stack[-1].fields.append(Field(name=attrs['name'], id=attrs['id'], type=field_type))

        elif tag == "message":
            if action == "start":
                attrs = dict(elem.items())
                stack.append(Message(name=attrs['name'], id=int(attrs['id']), description=attrs.get('description')))

            elif action == "end":
                x = stack.pop()
                assert isinstance(stack[-1], Schema)
                stack[-1].messages[x.id] = x

        elif tag == "group":
            if action == "start":
                attrs = dict(elem.items())
                stack.append(Group(
                    name=attrs['name'],
                    id=attrs['id'],
                    dimensionType=stack[0].types[attrs.get('dimensionType', 'groupSizeEncoding')]))

            elif action == "end":
                x = stack.pop()
                assert isinstance(stack[-1], (Group, Message))
                stack[-1].fields.append(x)

        else:
            assert 0, f"Unknown tag '{tag}'"

    return stack[0]
