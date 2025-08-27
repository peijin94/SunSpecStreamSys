#!/usr/bin/env python3

"""
Test script to generate dummy streaming data
This helps test if the StreamReceiver is working correctly without needing the full OVRO pipeline.
"""

import zmq
import json
import numpy as np
import time
import threading

class DummyStreamer:
    """Generate dummy streaming data for testing"""
    
    def __init__(self, stream_addr='127.0.0.1', stream_port=9798, stream_interval=0.25):
        self.stream_addr = stream_addr
        self.stream_port = stream_port
        self.stream_interval = stream_interval
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://{self.stream_addr}:{self.stream_port}")
        
        # Control flags
        self.running = False
        self.shutdown_event = threading.Event()
        
        print(f"DummyStreamer initialized: {stream_addr}:{stream_port}, interval: {stream_interval}s")
    
    def generate_dummy_data(self, frame_count):
        """Generate dummy data with realistic shape"""
        # Generate data with shape (1, 3072, 4) - (nbeam, nchan, npol)
        # Use different patterns to make it easy to verify
        data = np.zeros((1, 3072, 4), dtype=np.float32)
        
        # Create some interesting patterns
        for pol in range(4):
            # Add some frequency-dependent structure
            freq_pattern = np.sin(2 * np.pi * np.arange(3072) / 100) * 0.1
            # Add some time-dependent variation
            time_variation = np.sin(2 * np.pi * frame_count / 10) * 0.05
            # Add some noise
            noise = np.random.normal(0, 0.01, 3072)
            
            data[0, :, pol] = 1.0 + freq_pattern + time_variation + noise + pol * 0.2
        
        return data
    
    def stream_loop(self):
        """Main streaming loop"""
        print("Starting dummy streaming...")
        frame_count = 0
        
        while not self.shutdown_event.is_set():
            try:
                # Generate dummy data
                dummy_data = self.generate_dummy_data(frame_count)
                
                # Prepare header
                stream_header = {
                    'time_tag': int(time.time() * 1e6),  # Microsecond timestamp
                    'nbeam': 1,
                    'nchan': 3072,
                    'npol': 4,
                    'timestamp': time.time(),
                    'data_shape': dummy_data.shape,
                    'frame_count': frame_count
                }
                
                # Send data via ZMQ
                header_msg = json.dumps(stream_header).encode()
                data_msg = dummy_data.tobytes()
                self.socket.send_multipart([b"data", header_msg, data_msg])
                
                print(f"Streamed frame {frame_count}: shape={dummy_data.shape}, "
                      f"mean={dummy_data.mean():.6f}, std={dummy_data.std():.6f}")
                
                frame_count += 1
                
                # Wait for next interval
                time.sleep(self.stream_interval)
                
            except Exception as e:
                print(f"Error in streaming loop: {str(e)}")
                time.sleep(0.1)
        
        print("Dummy streaming stopped")
    
    def start(self):
        """Start the dummy streamer"""
        if self.running:
            print("Streamer is already running")
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self.stream_loop, daemon=True)
        self.stream_thread.start()
        
        print("DummyStreamer started")
    
    def stop(self):
        """Stop the dummy streamer"""
        if not self.running:
            print("Streamer is not running")
            return
        
        print("Stopping DummyStreamer...")
        self.running = False
        self.shutdown_event.set()
        
        # Wait for thread to finish
        if hasattr(self, 'stream_thread'):
            self.stream_thread.join(timeout=5.0)
        
        # Close ZMQ resources
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.term()
        
        print("DummyStreamer stopped")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dummy Streamer for Testing")
    parser.add_argument('--addr', type=str, default='127.0.0.1',
                       help='Stream address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=9798,
                       help='Stream port (default: 9798)')
    parser.add_argument('--interval', type=float, default=0.25,
                       help='Streaming interval in seconds (default: 0.25)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration to stream in seconds (default: 60)')
    
    args = parser.parse_args()
    
    # Create and start streamer
    streamer = DummyStreamer(
        stream_addr=args.addr,
        stream_port=args.port,
        stream_interval=args.interval
    )
    
    try:
        streamer.start()
        
        # Stream for specified duration
        print(f"Streaming for {args.duration} seconds... Press Ctrl+C to stop early")
        time.sleep(args.duration)
        
    except KeyboardInterrupt:
        print("\nStopping early...")
    finally:
        streamer.stop()

if __name__ == "__main__":
    main()
