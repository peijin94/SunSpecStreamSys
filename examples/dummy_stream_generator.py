#!/usr/bin/env python3

"""
Test stream generator to create dummy ZMQ data for testing stream catchers
This script generates realistic dummy data that matches the format from dr_beam.py
"""

import zmq
import json
import numpy as np
import time
import threading
import argparse

class TestStreamGenerator:
    """Generate test ZMQ stream data"""
    
    def __init__(self, stream_addr='127.0.0.1', stream_port=9798, stream_interval=0.25):
        self.stream_addr = stream_addr
        self.stream_port = stream_port
        self.stream_interval = stream_interval
        
        # Data parameters (matching dr_beam.py)
        self.nbeam = 1
        self.nchan = 3072
        self.npol = 4
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://{self.stream_addr}:{self.stream_port}")
        
        # Control flags
        self.running = False
        self.shutdown_event = threading.Event()
        
        print(f"TestStreamGenerator initialized: {stream_addr}:{stream_port}")
        print(f"Stream interval: {stream_interval}s")
        print(f"Data shape: ({self.nbeam}, {self.nchan}, {self.npol})")
    
    def generate_test_data(self, frame_count):
        """Generate realistic test data"""
        # Create data with shape (nbeam, nchan, npol)
        data = np.zeros((self.nbeam, self.nchan, self.npol), dtype=np.float32)
        
        # Add some interesting patterns
        for pol in range(self.npol):
            # Add frequency-dependent structure
            freq_pattern = np.sin(2 * np.pi * np.arange(self.nchan) / 100) * 0.1
            
            # Add time-dependent variation
            time_variation = np.sin(2 * np.pi * frame_count / 10) * 0.05
            
            # Add some noise
            noise = np.random.normal(0, 0.01, self.nchan)
            
            # Add polarization-dependent offset
            pol_offset = pol * 0.2
            
            data[0, :, pol] = 1.0 + freq_pattern + time_variation + noise + pol_offset
        
        return data
    
    def stream_loop(self):
        """Main streaming loop"""
        print("Starting test streaming...")
        frame_count = 0
        
        while not self.shutdown_event.is_set():
            try:
                # Generate test data
                test_data = self.generate_test_data(frame_count)
                
                # Prepare header (matching dr_beam.py format)
                stream_header = {
                    'time_tag': int(time.time() * 1e6),  # Microsecond timestamp
                    'nbeam': self.nbeam,
                    'nchan': self.nchan,
                    'npol': self.npol,
                    'timestamp': time.time(),  # Current time
                    'last_block_time': time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'data_shape': test_data.shape,
                    'data_type': '<f4'
                }
                
                # Send data via ZMQ
                header_msg = json.dumps(stream_header).encode()
                data_msg = test_data.tobytes()
                self.socket.send_multipart([b"data", header_msg, data_msg])
                
                print(f"Streamed frame {frame_count:4d}: shape={test_data.shape}, "
                      f"mean={test_data.mean():.6f}, std={test_data.std():.6f}")
                
                frame_count += 1
                
                # Wait for next interval
                time.sleep(self.stream_interval)
                
            except Exception as e:
                print(f"Error in streaming loop: {e}")
                time.sleep(0.1)
        
        print("Test streaming stopped")
    
    def start(self):
        """Start the test streamer"""
        if self.running:
            print("Streamer is already running")
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self.stream_loop, daemon=True)
        self.stream_thread.start()
        
        print("TestStreamGenerator started")
    
    def stop(self):
        """Stop the test streamer"""
        if not self.running:
            print("Streamer is not running")
            return
        
        print("Stopping TestStreamGenerator...")
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
        
        print("TestStreamGenerator stopped")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test ZMQ Stream Generator")
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
    streamer = TestStreamGenerator(
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


