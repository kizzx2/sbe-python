# sbe-python

Easy to use, pure Python FIX (SBE)(https://www.fixtrading.org/standards/sbe/) encoder and decoder.

## Install

```bash
pip isntall sbe
```

## Usage

### Decoding

```python
import sbe

with open('your-schema.xml', 'r') as f:
  schema = sbe.Schema.parse(f)

wtih open('your-data.sbe', 'rb') as f:
  buf = f.read()

# Get a Python dict in one-line
schema.decode(buf)

# If you need an offset, apply them Pythonicaly
schema.decode(buf[19:])
```

### Encoding

```python
import sbe

with open('./your-schema.xml', 'r') as f:
  schema = sbe.Schema.parse(f)

# message_id from the schema you want to encode
message_id = 3

# Encode from Python dict in one-line
schema.encode(scheam.messages[3], obj)
```
