# sbe-python

Easy to use, fast, pure Python FIX [SBE](https://www.fixtrading.org/standards/sbe/) encoder and decoder.

## Install

```bash
pip install sbe
```

## Usage

### Simple Decoding

Decode SBE to Python dictionaries in one line. Good for exploratory data analysis.

```python
import sbe

with open('your-schema.xml', 'r') as f:
  schema = sbe.Schema.parse(f)

wtih open('your-data.sbe', 'rb') as f:
  buf = f.read()

# Get a Python dict in one-line
x = schema.decode(buf)

x.name  # The template message name

x.value
# {'userId': 11,
# 'timestamp': 1598784004840,
# 'orderSize': 0,
# 'price': 5678.0,
# ...

# If you need an initial offset, apply it Pythonically
schema.decode(buf[19:])

# decode_header to filter out messages based on header to avoid decoding
# message bodies that are not needed
schema.decode_header(buf)['templateId']
```

### High Performance Decoding (Wrapping)

This gives you decent performance while still retaining high code readability.

```python
import sbe

with open('your-schema.xml', 'r') as f:
  schema = sbe.Schema.parse(f)

wtih open('your-data.sbe', 'rb') as f:
  buf = f.read()

# Wrap the buffer without decoding it, fields are converted to Python variables
# on demand
x = schema.wrap(buf)

x.header['templateId']
x.body['price']
x.body['someGroup'][2]['price']
```

### Direct Access with Pointers

`get_raw_pointer` gives you the required information to unpack a variable from `memoryview` / `bytes`. This gets you very close to the fastest achievable performance in Python:

```python
import sbe

with open('your-schema.xml', 'r') as f:
  schema = sbe.Schema.parse(f)

header_pointer = schema.header_wrapper.get_raw_pointer('templateId')

# Let's say we are only interested in messages of templateId == 3
price_pointer = schema.message_wrappers[3].get_raw_pointer('price')

wtih open('your-data.sbe', 'rb') as f:
  buf = f.read()

# pass `memoryview` to `unpack` to avoid copying
buf = memoryview(buf)[initial_offset:]
template_id = header_pointer.unpack(buf)  # calls buf[offset:offset+size].cast("I")[0] directly

if template_id == 3:
  print(price_pointer.unpack(buf))
```

### Encoding

```python
import sbe

with open('./your-schema.xml', 'r') as f:
  schema = sbe.Schema.parse(f)

# message_id from the schema you want to encode
message_id = 3

obj = {
  'userId': 11,
  'price': 5678.0,
  # ...
}

# Encode from Python dict in one-line
schema.encode(schema.messages[3], obj)

# You can supply your header values as a dict
schema.encode(schema.messages[3], obj, headers)
```
