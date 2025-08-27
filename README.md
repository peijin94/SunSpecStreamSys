# Streaming System Documentation

This document describes the streaming system that consists of an `AvgStreamingOp` in the OVRO data recorder and a `StreamReceiver` in the SunSpecStreamSys repository.

## Overview

The streaming system provides real-time data streaming from the OVRO data recorder to external applications via ZMQ (ZeroMQ). The system:

1. **AvgStreamingOp**: Averages data over time and streams it every 0.25 seconds
2. **StreamReceiver**: Receives the streamed data and maintains a ring buffer

## Components

### 1. AvgStreamingOp (in ovro_data_recorder/scripts/dr_beam.py)

This operation class:
- Reads power spectra data from the input ring
- Averages data over time (axis=0)
- Streams averaged data via ZMQ every 0.5 seconds
- Sends data to `localhost:9798` by default

**Key Features:**
- Configurable streaming interval (default: 0.25s)
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
- **NEW**: Automatically generates and saves plots every 10 seconds

**Key Features:**
- Automatic connection to streaming source
- Ring buffer with configurable length (default: 1200)
- Data processing: extracts pol=0 and averages down to N_freq/2
- Thread-safe operation with proper shutdown handling
- Automatic garbage collection to reduce memory overhead
- Memory usage monitoring and optimization
- **NEW**: Automatic plotting with configurable interval
- **NEW**: Two-panel plots: latest spectrum + waterfall visualization
- **NEW**: Plots saved to configurable directory with timestamps

## Data Flow

```
OVRO Data Recorder → AvgStreamingOp → ZMQ Stream → StreamReceiver → Ring Buffer
```

### Data Processing Pipeline:

1. **Input**: (ntime_gulp, nbeam, nchan, npol) = (250, 1, 3072, 4)
2. **Time Averaging**: Average over time axis → (1, 3072, 4)
3. **Streaming**: Send via ZMQ every 0.25s
4. **Reception**: Receive in StreamReceiver
5. **Processing**: Extract pol=0 → (3072,)
6. **Frequency Averaging**: Average down by factor of 4 → (768,)
7. **Ring Buffer**: Store in 1200 × 768 circular buffer

## Plotting System

The StreamReceiver now includes an automatic plotting system that generates visualizations every 10 seconds (configurable):

### Plot Types

1. **Latest Spectrum Plot** (top panel):
   - Shows the most recent power spectrum
   - **Log scale** for better dynamic range visualization
   - X-axis: Frequency channels (0-1535)
   - Y-axis: Power values (logarithmic scale)
   - Includes timestamp and lag information
   - Grid overlay for easy reading

2. **Waterfall Plot** (bottom panel):
   - **Time vs Frequency visualization** of recent data
   - **X-axis: Time** (most recent at right)
   - **Y-axis: Frequency channels**
   - **Log-scale color mapping** for better dynamic range
   - Color-coded power values using viridis colormap
   - Includes colorbar and time/lag information

### Plot Features

- **Log Scales**: Both power spectrum and waterfall use logarithmic scaling for better dynamic range
- **Real-time Delay Tracking**: Automatically tracks delay between data timestamp and current time using header time_tag
- **Multiple Delay Methods**: Configurable delay calculation (auto, buffer-based, or manual)
- **Lag Information**: Titles show actual data lag in real-time (not estimated)
- **Orientation**: Waterfall plot shows time progression horizontally (left to right)
- **Dynamic Range**: Automatic y-axis limits based on data percentiles for optimal viewing
- **Real-time Updates**: Delay information updates with each new data frame
- **Robust Error Handling**: Automatic fallback to buffer-based calculation if timestamp parsing fails
- **LWA Timestamp Support**: Direct parsing of fixed-format timestamp strings (e.g., "1755552894.787341")
- **UTC Time Consistency**: All timestamps and delay calculations use UTC time via AstroPy Time module for consistency across timezones

### Delay Calculation Methods

The system supports three methods for calculating the delay between data timestamp and current time:

1. **`auto` (default)**: 
   - Uses `header["timestamp"]` field for accurate delay calculation
   - Automatically detects LWA time tag format and converts to UTC
   - Falls back to buffer-based calculation if timestamp parsing fails
   - Provides real-time delay updates with each data frame

2. **`buffer`**: 
   - Uses relative buffer position for delay estimation
   - Calculates: `(buffer_length - buffer_index) * streaming_interval`
   - Useful when timestamp information is unavailable or unreliable

3. **`manual`**: 
   - Uses manually set delay value via `receiver.set_delay(seconds)`
   - Useful for testing or when you want to override automatic calculation

#### LWA System Support

The system now properly handles **LWA timestamps**:
- **Fixed format**: `header["timestamp"]` is always a string like `"1755552894.787341"`
- **Direct parsing**: Converts timestamp string directly to float using `float(timestamp_str)`
- **UTC timing**: All delay calculations use UTC time via AstroPy Time module
- **Example timestamp**: `"1755552894.787341"` represents seconds since 1970-01-01 UTC
- **Simple conversion**: No complex tick calculations needed - direct epoch time comparison

**Usage:**
```python
# Set delay calculation method
receiver.set_delay_calculation_method('buffer')  # Always use buffer-based
receiver.set_delay_calculation_method('manual')  # Use manual setting
receiver.set_delay_calculation_method('auto')    # Default: try time_tag, fallback to buffer

# Manually set delay
receiver.set_delay(2.5)  # Set delay to 2.5 seconds
```

### Plot Configuration

- **Interval**: Configurable plotting frequency (default: 10 seconds, 0 = disable plotting)
- **Directory**: Plots saved to `/fast/peijinz/streaming/figs/` by default
- **Format**: High-resolution PNG files (150 DPI)
- **Naming**: `spectrum_YYYYMMDD_HHMMSS.png` format
- **Size**: 12×10 inch figures with tight layout

### Plotting Thread

- Runs in a separate daemon thread to avoid blocking data reception
- Automatically creates plot directory if it doesn't exist
- Handles errors gracefully and continues operation
- Integrates with the main shutdown process

## Web Interface

The StreamReceiver now includes an optional web interface for real-time spectrum visualization:

### Features

- **Live Spectrum Display**: Real-time waterfall visualization of incoming data
- **Interactive Controls**: Adjustable update rate, waterfall height, and color mapping
- **Multiple Color Palettes**: Turbo, Viridis, and Grayscale options
- **Responsive Design**: Works on desktop and mobile devices
- **Fullscreen Mode**: Immersive viewing experience
- **Performance Monitoring**: Real-time FPS counter and frame statistics

### Enabling the Web Interface

To enable the web interface, add the `--start-webshow` flag when starting the receiver:

```bash
python3 stream_receiver.py --start-webshow --addr 127.0.0.1 --port 9798
```

The web interface will be available at `http://localhost:9527`.

### Web Interface Controls

- **Pause/Resume**: Stop or start data updates
- **Update Rate**: Control how frequently data is fetched (1-60 Hz)
- **Waterfall Height**: Adjust the number of visible time lines (128-1200)
- **Floor/Ceiling**: Set dB range for color mapping
- **Linear Mapping**: Toggle between dB and linear color scaling
- **Color Palette**: Choose between Turbo, Viridis, and Grayscale
- **Refresh Data**: Manually fetch latest data
- **Fullscreen**: Toggle fullscreen mode

### Web Interface Architecture

- **Backend**: Flask web server running in a separate thread
- **Frontend**: HTML5 Canvas with JavaScript for real-time rendering
- **Data API**: RESTful endpoints for spectrum data and status
- **Real-time Updates**: Automatic data fetching and display updates
- **Ring Buffer Visualization**: Efficient waterfall display using offscreen canvas

### Web Interface Dependencies

The web interface requires additional Python packages:
- `flask>=2.0.0` - Web framework
- `flask-cors>=3.0.0` - Cross-origin resource sharing support

Install with:
```bash
pip install -r requirements.txt
```

## Installation and Setup

### Prerequisites

```bash
# In SunSpecStreamSys directory
pip install -r requirements.txt
```

### Dependencies

- `numpy` >= 1.19.0
- `pyzmq` >= 22.0.0
- `matplotlib` >= 3.3.0
- `psutil` >= 5.8.0
- `astropy` >= 4.0.0

### Headless Server Configuration

The plotting system is automatically configured for headless servers (SSH sessions without display):

- **Matplotlib Backend**: Uses 'Agg' backend (non-interactive)
- **Environment Variables**: Automatically set `MPLBACKEND=Agg` and `DISPLAY=""`
- **No Display Required**: Plots are saved directly to files without GUI
- **SSH Compatible**: Works perfectly in remote server environments

This configuration ensures that plotting works seamlessly on headless servers where no X11 display is available.

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
- `--plot-interval`: Plotting interval in seconds (0 = disable plotting, default: 10)
- `--plot-dir`: Directory to save plots (default: /fast/peijinz/streaming/figs/)
- `--start-webshow`: Enable web interface at localhost:9898 (default: False)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Programmatic Usage

```python
from stream_receiver import StreamReceiver

# Create receiver with plotting enabled
receiver = StreamReceiver(
    stream_addr='127.0.0.1',
    stream_port=9798,
    buffer_length=1200,
    plot_interval=10,  # Plot every 10 seconds
    plot_dir='/fast/peijinz/streaming/figs/'  # Save plots here
)

# Create receiver with plotting disabled
receiver_no_plots = StreamReceiver(
    stream_addr='127.0.0.1',
    stream_port=9798,
    buffer_length=1200,
    plot_interval=0,  # Disable plotting
    plot_dir='/fast/peijinz/streaming/figs/'  # This will be ignored when plotting is disabled
)

# Start receiving
receiver.start()

# Get latest data
latest_data = receiver.get_latest_data(n_frames=10)  # Get last 10 frames

# Get status (now includes plotting information)
status = receiver.get_buffer_status()
print(f"Plotting: {status['plot_running']}, Last plot: {status['last_plot_time']}")
print(f"Current delay: {status['current_delay']:.3f}s")

# Get delay information directly
current_delay = receiver.get_current_delay()
print(f"Data lag: {current_delay:.3f} seconds")

# Stop when done
receiver.stop()
```

# Create receiver with web interface enabled
receiver_web = StreamReceiver(
    stream_addr='127.0.0.1',
    stream_port=9798,
    buffer_length=1200,
    plot_interval=0,  # Disable plotting when using web interface
    start_webshow=True  # Enable web interface
)

# Start receiving and web server
receiver_web.start()

# Web interface will be available at http://localhost:9527
print("Web interface started at http://localhost:9527")

## Testing

### Test Ring Buffer Functionality

```bash
cd SunSpecStreamSys
python tests/test_receiver.py
```

This will test the ring buffer functionality without requiring an active stream.

### Test Plotting Functionality

```bash
cd SunSpecStreamSys
python tests/test_plotting.py
```

This will test the plotting system without requiring an active stream.

### Test Headless Plotting

```bash
cd SunSpecStreamSys
python tests/test_headless_plotting.py
```

This will verify that plotting works correctly on headless servers without display issues.

### Test with Active Stream

1. Start the OVRO data recorder first
2. Run the test script and choose to test actual streaming
3. Monitor the output for data reception

### Test Web Interface

```bash
cd SunSpecStreamSys
python tests/test_web_interface.py
```

This will test the web interface functionality without requiring an active stream.

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
- **Current delay between data timestamp and real-time**

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

### Display and Matplotlib Issues

5. **"Could not connect to display" error**: 
   - This is normal on headless servers
   - The system automatically uses the 'Agg' backend
   - Plots are saved to files without displaying
   - No action needed

6. **"QApplication was not created in the main() thread" warning**:
   - This warning is harmless on headless servers
   - Matplotlib automatically handles threading issues
   - Plots will still be generated and saved correctly

7. **"UserWarning: Starting a Matplotlib GUI outside of the main thread"**:
   - This warning is expected and can be ignored
   - The plotting system is designed to work in background threads
   - No impact on functionality

### Delay Calculation Issues

8. **Large negative delays (e.g., -342329810474.9s)**:
   - This indicates invalid `time_tag` values in the header data
   - The system automatically falls back to buffer-based calculation
   - Check header data format and `time_tag` values
   - Use `receiver.set_delay_calculation_method('buffer')` for reliable timing
   - Run `python tests/debug_time_tag.py` to diagnose time format issues

9. **Large positive delays (>1 hour)**:
   - This is normal for LWA systems using timestamps from earlier observations
   - The delay represents the time difference between data timestamp and current time
   - System automatically handles LWA timestamp conversion using `lwa_time_tag_to_datetime()`
   - Use `receiver.set_delay_calculation_method('buffer')` if you prefer relative timing

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

- **Streaming interval**: 0.25s provides good balance between real-time updates and system load
- **Buffer size**: 1200 frames provide ~5 minutes of data history
- **Memory usage**: Ring buffer uses ~3.7 MB (1200 × 768 × 4 bytes)
- **Network**: ZMQ provides efficient, low-latency data transmission

## Future Enhancements

- Configurable data processing pipelines
- Multiple output formats
- Real-time visualization
- Data persistence and archiving
- Network compression and optimization
