import sbe
import os

def test_parse1():
    with open('tests/dat/example-schema.xml', 'r') as f:
        schema = sbe.Schema.parse(f)
