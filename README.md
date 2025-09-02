# Streaming System Documentation

This document describes the streaming system that consists of an `AvgStreamingOp` in the OVRO data recorder and a `StreamReceiver` in the SunSpecStreamSys repository.


## Components

### 1. AvgStreamingOp (in ovro_data_recorder/scripts/dr_beam.py)

This operation class:
- Reads power spectra data from the input ring
- Averages data over time (axis=0)
- Streams averaged data via ZMQ every 0.25 seconds
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
- Maintains a ring buffer of 1200 × 768 data points
- Can be started/stopped at any time
- **NEW**: Automatically generates and saves plots every 10 seconds

**Key Features:**
- Automatic connection to streaming source
- Ring buffer with configurable length (default: 1200)
- Data processing: extracts pol=0 and averages down to N_freq/4
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
3. **Streaming**: Send via ZMQ every 0.25s with updated header format
4. **Reception**: Receive in StreamReceiver
5. **Processing**: Extract pol=0 → (3072,)
6. **Frequency Averaging**: Average down by factor of 4 → (768,)
7. **Ring Buffer**: Store in 1200 × 768 circular buffer

### Header Format

The streaming header now includes:
- `timestamp`: Current time when message is created (float, seconds since epoch)
- `last_block_time`: Last block processing time (string, LWA time format)
- `time_tag`: Legacy time tag from LWA system
- `nbeam`, `nchan`, `npol`: Data dimensions
- `data_shape`: Shape of the averaged data
- `data_type`: Data type specification


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

### To run the capture service

Do:

```bash
python3 stream_receiver.py --start-webshow --addr [address] --port 9798
```

The `[address]` should be `127.0.0.1` if this is on local machine (calim02).

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
- `--start-webshow`: Enable web interface at localhost:9527 (default: False)
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
Web interface will be available at http://localhost:9527

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

