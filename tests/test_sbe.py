import sbe

def test_parse1():
    with open('tests/dat/example-schema.xml', 'r') as f:
        sbe.Schema.parse(f)

def test_parse2():
    with open('tests/dat/b3-entrypoint-messages-8.0.0.xml', 'r') as f:
        sbe.Schema.parse(f)

def test_nullValue():
    with open('tests/dat/example-schema.xml', 'r') as f:
        s = sbe.Schema.parse(f)
        nullable = s.messages[2]

        encodedInt = s.encode(nullable, {'nullable': 5})
        decodedInt = s.decode(encodedInt)
        assert decodedInt.value['nullable'] == 5

        encodedNull = s.encode(nullable, {'nullable': None})
        decodedNull = s.decode(encodedNull)
        assert decodedNull.value['nullable'] is None
