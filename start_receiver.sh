#!/bin/bash

# Startup script for StreamReceiver
# This script starts the StreamReceiver with common configuration options

# Set environment variables for headless matplotlib operation
export MPLBACKEND=Agg
export DISPLAY=""

echo "Starting StreamReceiver..."
echo "Address: 127.0.0.1"
echo "Port: 9798"
echo "Buffer Length: 1200"
echo "GC Interval: 100 frames"
echo "Plot Interval: 10 seconds"
echo "Plot Directory: /fast/peijinz/streaming/figs/"
echo "Matplotlib Backend: $MPLBACKEND (headless mode)"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import zmq, numpy, matplotlib, flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip3 install -r requirements.txt
fi

# Create plots directory
echo "Creating plots directory..."
mkdir -p /fast/peijinz/streaming/figs/

# Start the receiver
echo "Launching StreamReceiver..."
echo ""
echo "Options:"
echo "  --plot-interval 10     # Plot every 10 seconds (0 = disable plotting)"
echo "  --start-webshow        # Enable web interface at http://localhost:9527"
echo ""
echo "To test the system before running:"
echo "  python tests/test_headless_plotting.py  # Test plotting without display"
echo "  python tests/test_plotting.py           # Test plotting functionality"
echo "  python tests/test_receiver.py           # Test basic functionality"
echo ""
echo "Starting with default settings..."
python3 stream_receiver.py --addr 127.0.0.1 --port 9798 --buffer-length 1200 --gc-interval 100 --plot-interval 30 --plot-dir /fast/peijinz/streaming/figs/ --log-level INFO

echo ""
echo "To enable web interface, add --start-webshow flag:"
echo "  python3 stream_receiver.py --start-webshow [other options]"
echo "Then open http://localhost:9527 in your browser"
