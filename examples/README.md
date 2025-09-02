# ZMQ Stream Examples

This directory contains example scripts for working with the ZMQ stream from `dr_beam.py`'s `AvgStreamingOp`.

## Files

### 1. `minimal_stream_catcher.py`
A minimal working example that demonstrates the basic functionality needed to receive and process streaming data from the OVRO data recorder.

**Features:**
- Basic ZMQ SUB socket connection
- Message parsing (topic, header, data)
- Data conversion from bytes to numpy arrays
- Simple statistics printing
- Optional frame saving for inspection

**Usage:**
```bash
python minimal_stream_catcher.py
```

### 2. `advanced_stream_catcher.py`
A more sophisticated example that includes ring buffer management, data processing, and plotting capabilities.

**Features:**
- Ring buffer for data storage
- Data averaging and downsampling (3072 → 768 channels)
- Real-time statistics tracking
- Optional matplotlib plotting
- Configurable buffer length and plot intervals

**Usage:**
```bash
# Basic usage
python advanced_stream_catcher.py

# With options
python advanced_stream_catcher.py --buffer-length 200 --plot-interval 60 --duration 300
```

**Command line options:**
- `--addr`: Stream address (default: 127.0.0.1)
- `--port`: Stream port (default: 9798)
- `--buffer-length`: Ring buffer length (default: 100)
- `--duration`: Duration to run in seconds (default: unlimited)
- `--plot-interval`: Plot creation interval in seconds (default: 30)

### 3. `test_stream_generator.py`
A test script that generates dummy ZMQ data for testing the stream catchers without requiring the full OVRO pipeline.

**Features:**
- Generates realistic dummy data matching dr_beam.py format
- Configurable streaming interval
- Proper header format with timestamps
- Thread-safe operation

**Usage:**
```bash
# Generate test data for 60 seconds
python test_stream_generator.py --duration 60

# With custom settings
python test_stream_generator.py --interval 0.5 --duration 120
```

## Quick Start

### Option 1: Test with dummy data
1. Start the test generator:
   ```bash
   python test_stream_generator.py
   ```

2. In another terminal, start a stream catcher:
   ```bash
   python minimal_stream_catcher.py
   ```

### Option 2: Connect to real dr_beam.py
1. Make sure `dr_beam.py` is running with `AvgStreamingOp` enabled
2. Start a stream catcher:
   ```bash
   python minimal_stream_catcher.py
   ```

## Data Format

The ZMQ stream sends multipart messages with the following structure:

```
[topic, header_json, data_bytes]
```

Where:
- `topic`: Always `b"data"` for data messages
- `header_json`: JSON string containing metadata
- `data_bytes`: Binary float32 data

### Header Format
```json
{
    "time_tag": 1234567890123456,
    "nbeam": 1,
    "nchan": 3072,
    "npol": 4,
    "timestamp": 1234567890.123,
    "last_block_time": "2025-01-17 12:34:56.789",
    "data_shape": [1, 3072, 4],
    "data_type": "<f4"
}
```

### Data Shape
- **Input**: (nbeam, nchan, npol) = (1, 3072, 4)
- **Processing**: Extract pol=0 → (3072,)
- **Downsampling**: Average by factor of 4 → (768,)

## Dependencies

### Required
- `zmq` (pyzmq)
- `numpy`
- `json` (built-in)

### Optional
- `matplotlib` (for plotting in advanced_stream_catcher.py)

Install with:
```bash
pip install pyzmq numpy matplotlib
```

