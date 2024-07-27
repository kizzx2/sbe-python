import sbe

def test_parse1():
    with open('tests/dat/example-schema.xml', 'r') as f:
        sbe.Schema.parse(f)

def test_parse2():
    with open('tests/dat/b3-entrypoint-messages-8.0.0.xml', 'r') as f:
        sbe.Schema.parse(f)
