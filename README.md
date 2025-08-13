# Streaming System Documentation

This document describes the streaming system that consists of an `AvgStreamingOp` in the OVRO data recorder and a `StreamReceiver` in the SunSpecStreamSys repository.

## Overview

The streaming system provides real-time data streaming from the OVRO data recorder to external applications via ZMQ (ZeroMQ). The system:

1. **AvgStreamingOp**: Averages data over time and streams it every 0.5 seconds
2. **StreamReceiver**: Receives the streamed data and maintains a ring buffer

## Components

### 1. AvgStreamingOp (in ovro_data_recorder/scripts/dr_beam.py)

This operation class:
- Reads power spectra data from the input ring
- Averages data over time (axis=0)
- Streams averaged data via ZMQ every 0.5 seconds
- Sends data to `localhost:9798` by default

**Key Features:**
- Configurable streaming interval (default: 0.5s)
- Configurable destination address and port
- Automatic data averaging and accumulation
- ZMQ PUB socket for broadcasting data

**Usage:**
The `AvgStreamingOp` is automatically added to the pipeline when running `dr_beam.py`. It will start streaming data as soon as the pipeline is running.

### 2. StreamReceiver (in SunSpecStreamSys/stream_receiver.py)

This receiver class:
- Connects to the ZMQ stream from `AvgStreamingOp`
- Processes incoming data frames
- Maintains a ring buffer of 1200 × 1536 data points
- Can be started/stopped at any time

**Key Features:**
- Automatic connection to streaming source
- Ring buffer with configurable length (default: 1200)
- Data processing: extracts pol=0 and averages down to N_freq/2
- Thread-safe operation with proper shutdown handling
- Automatic garbage collection to reduce memory overhead
- Memory usage monitoring and optimization

## Data Flow

```
OVRO Data Recorder → AvgStreamingOp → ZMQ Stream → StreamReceiver → Ring Buffer
```

### Data Processing Pipeline:

1. **Input**: (ntime_gulp, nbeam, nchan, npol) = (250, 1, 3072, 4)
2. **Time Averaging**: Average over time axis → (1, 3072, 4)
3. **Streaming**: Send via ZMQ every 0.5s
4. **Reception**: Receive in StreamReceiver
5. **Processing**: Extract pol=0 → (3072,)
6. **Frequency Averaging**: Average down by factor of 2 → (1536,)
7. **Ring Buffer**: Store in 1200 × 1536 circular buffer

## Installation and Setup

### Prerequisites

```bash
# In SunSpecStreamSys directory
pip install -r requirements.txt
```

### Dependencies

- `numpy` >= 1.19.0
- `pyzmq` >= 22.0.0

## Usage

### Starting the Stream

1. **Start the OVRO data recorder** (this will automatically start `AvgStreamingOp`):
   ```bash
   cd ovro_data_recorder/scripts
   python dr_beam.py -o -b 4 --remote-addr 127.0.0.1 --remote-port 5555 -l /path/to/logfile.log
   ```

2. **Start the StreamReceiver** (can be done anytime):
   ```bash
   cd SunSpecStreamSys
   python stream_receiver.py --addr 127.0.0.1 --port 9798 --buffer-length 1200
   ```

### Command Line Options for StreamReceiver

- `--addr`: Stream address (default: 127.0.0.1)
- `--port`: Stream port (default: 9798)
- `--buffer-length`: Ring buffer length (default: 1200)
- `--gc-interval`: Garbage collection interval in frames (default: 100)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Programmatic Usage

```python
from stream_receiver import StreamReceiver

# Create receiver
receiver = StreamReceiver(
    stream_addr='127.0.0.1',
    stream_port=9798,
    buffer_length=1200
)

# Start receiving
receiver.start()

# Get latest data
latest_data = receiver.get_latest_data(n_frames=10)  # Get last 10 frames

# Get status
status = receiver.get_buffer_status()

# Stop when done
receiver.stop()
```

## Testing

### Test Ring Buffer Functionality

```bash
cd SunSpecStreamSys
python test_receiver.py
```

This will test the ring buffer functionality without requiring an active stream.

### Test with Active Stream

1. Start the OVRO data recorder first
2. Run the test script and choose to test actual streaming
3. Monitor the output for data reception

## Configuration

### AvgStreamingOp Configuration

The streaming operation can be configured by modifying the `dr_beam.py` script:

- **Streaming interval**: Change `stream_interval` parameter
- **Destination**: Modify `stream_addr` and `stream_port` parameters
- **Data processing**: Adjust averaging and accumulation logic

### StreamReceiver Configuration

The receiver can be configured through:

- Constructor parameters
- Command line arguments
- Runtime configuration methods

## Monitoring and Debugging

### Logging

Both components provide comprehensive logging:
- `AvgStreamingOp`: Logs streaming operations and data statistics
- `StreamReceiver`: Logs connection status, data processing, and buffer updates

### Status Monitoring

The `StreamReceiver` provides status information:
- Buffer index and shape
- Running state
- Latest data statistics

### Performance Metrics

- Data processing time
- Buffer update frequency
- Memory usage (ring buffer size)
- Garbage collection statistics
- Memory optimization status

## Troubleshooting

### Common Issues

1. **Connection refused**: Ensure `AvgStreamingOp` is running and streaming
2. **No data received**: Check ZMQ socket configuration and firewall settings
3. **Buffer overflow**: Adjust buffer length or processing frequency
4. **High CPU usage**: Check streaming interval and data processing efficiency

### Debug Mode

Enable debug logging for detailed information:
```bash
python stream_receiver.py --log-level DEBUG
```

### Memory Optimization

The `StreamReceiver` includes several memory optimization features:

- **Automatic Garbage Collection**: Runs every N frames (configurable via `--gc-interval`)
- **Memory Monitoring**: Track RSS memory usage and object counts
- **Manual Optimization**: Call `optimize_memory()` to clear old data and force GC
- **Memory Statistics**: Get detailed memory usage information

```python
# Get memory information
memory_info = receiver.get_memory_info()
print(f"Memory usage: {memory_info['rss_mb']:.1f}MB")

# Force garbage collection
collected = receiver.force_garbage_collection()

# Optimize memory usage
optimization_result = receiver.optimize_memory()
```

## Performance Considerations

- **Streaming interval**: 0.5s provides good balance between real-time updates and system load
- **Buffer size**: 1200 frames provide ~10 minutes of data history
- **Memory usage**: Ring buffer uses ~7.4 MB (1200 × 1536 × 4 bytes)
- **Network**: ZMQ provides efficient, low-latency data transmission

## Future Enhancements

- Configurable data processing pipelines
- Multiple output formats
- Real-time visualization
- Data persistence and archiving
- Network compression and optimization
