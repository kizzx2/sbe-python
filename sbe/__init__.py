import enum
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TextIO, Union

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


FORMAT = dict(zip(PrimitiveType.__members__.values(), "cBHIQbhiqfd"))
FORMAT_SIZES = {k: struct.calcsize(v) for k, v in FORMAT.items()}
PRIMITIVE_TYPES = set(x.value for x in PrimitiveType.__members__.values())


class SetEncodingType(enum.Enum):
    UINT8 = 'uint8'
    UINT16 = 'uint16'
    UINT32 = 'uint32'
    UINT64 = 'uint64'


class EnumEncodingType(enum.Enum):
    UINT8 = 'uint8'
    CHAR = 'char'


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

    def __repr__(self):
        return self.value.__repr__()


@dataclass
class DecodedMessage:
    message_name: str
    header: dict
    value: dict


@dataclass
class Type:
    name: str
    primitiveType: PrimitiveType
    presence = Presence.REQUIRED
    semanticType: Optional[str] = None
    description: Optional[str] = None
    length = 1
    characterEncoding: Optional[CharacterEncoding] = None

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

    def __repr__(self):
        return (self.name, self.value).__repr__()


@dataclass
class Enum:
    name: str
    encodingType: Union[EnumEncodingType, Type]
    presence = Presence.REQUIRED
    semanticType: Optional[str] = None
    description: Optional[str] = None
    valid_values: List[EnumValue] = field(default_factory=lambda: [])

    def find_value_by_name(self, name: str) -> str:
        return int(next(x for x in self.valid_values if x.name == name).value)

    def find_name_by_value(self, val: str) -> str:
        return next(x for x in self.valid_values if x.value == val).name

    def __repr__(self):
        return f"<Enum '{self.name}'>"


@dataclass
class Composite:
    name: str
    types: List[Union['Composite', Type]] = field(default_factory=lambda: [])
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

    def __repr__(self):
        return f"<Set '{self.name}'>"


@dataclass
class Field:
    name: str
    id: str
    type: Union[PrimitiveType, str]
    description: Optional[str] = None
    sinceVersion: int = 0

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
    fields: List[Union['Group', Field]] = field(default_factory=lambda: [])


@dataclass
class Message:
    name: str
    id: int
    description: Optional[str] = None
    fields: List[Union[Group, Field]] = field(default_factory=lambda: [], repr=False)


@dataclass
class Cursor:
    val: int = 0


@dataclass
class Schema:
    id: int
    version: int
    types: Dict[str, Union[Composite, Type]] = field(default_factory=lambda: {})
    messages: Dict[str, Message] = field(default_factory=lambda: {})

    @classmethod
    def parse(cls, f: TextIO):
        return _parse_schema(f)

    def encode(self, message: Message, obj: dict, header: Optional[dict] = None):
        if header is None:
            header = {}

        fmts = []
        vals = []
        cursor = Cursor()

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

    def decode(self, buffer: Union[bytes, memoryview]):
        buffer = memoryview(buffer)

        header = _unpack_composite(self, self.types['messageHeader'], buffer)
        body_offset = header.size

        m = self.messages[header.value['templateId']]

        cursor = Cursor()
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
        _walk_fields(self, rv, m.fields, vals, Cursor())
        return DecodedMessage(m.name, header.value, rv)


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


def _pack_format(schema: Schema, composite: Composite):
    fmt = []
    for t in composite.types:
        if t.length > 1 and t.primitiveType == PrimitiveType.CHAR:
            fmt.append(str(t.length) + 's')
        else:
            fmt.append(FORMAT[t.primitiveType])

    return ''.join(fmt)


def _pack_composite(schema: Schema, composite: Composite, obj: dict):
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


def _prettify_type(schema: Schema, t: Type, v):
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
                _walk_fields_encode(schema, f.fields, x, fmt1, vals1, Cursor())
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


def _walk_fields(schema: Schema, rv: dict, fields: List[Union[Group, Field]], vals: List, cursor: Cursor):
    for f in fields:
        if isinstance(f, Group):
            num_in_group = vals[cursor.val + 1]
            cursor.val += 2

            rv[f.name] = []
            for _ in range(num_in_group):
                rv1 = {}
                _walk_fields(schema, rv1, f.fields, vals, cursor)
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
                stack.append(Enum(name=attrs['name'], encodingType=EnumEncodingType(attrs['encodingType'])))

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
