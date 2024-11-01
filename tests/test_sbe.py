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

def test_blockLength():
    with open('tests/dat/example-schema.xml', 'r') as f:
        s = sbe.Schema.parse(f)
        msg  = s.messages[3]

        encoded = s.encode(msg, {'year': 1990, 'AGroup': [{'numbers': 123},
                                                          {'numbers': 456}]})
        # BlockHeader = 8b
        # Body = 2b year + 2b padding
        # Repeating group = 2 * (4b numbers + 2b padding)
        expLen = 8 + 4 + 2*6
        assert len(encoded) == expLen, "Encoded SBE not padded properly"

        decoded  = s.decode(encoded)
        assert decoded.value['year'] == 1990
        assert decoded.value['AGroup'][1]['numbers'] == 456
