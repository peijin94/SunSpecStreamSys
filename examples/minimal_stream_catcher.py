#!/usr/bin/env python3

"""
Minimal working example to catch the ZMQ stream from dr_beam.py
This script demonstrates the basic functionality needed to receive and process
streaming data from the AvgStreamingOp in the OVRO data recorder.
"""

import zmq
import json
import numpy as np
import time
import sys

def main():
    """Main function to demonstrate ZMQ stream reception"""
    
    # Configuration
    stream_addr = '127.0.0.1'
    stream_port = 9798
    
    print(f"Connecting to ZMQ stream at {stream_addr}:{stream_port}")
    print("Make sure dr_beam.py is running with AvgStreamingOp enabled")
    print("Press Ctrl+C to stop\n")
    
    # Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{stream_addr}:{stream_port}")
    socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
    
    # Set socket timeout for non-blocking operation
    socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            try:
                # Receive multipart message: [topic, header, data]
                message = socket.recv_multipart()
                
                if len(message) >= 3:
                    topic = message[0]
                    header_data = message[1]
                    data = message[2]
                    
                    # Parse header
                    header = json.loads(header_data.decode())
                    
                    # Process data based on topic
                    if topic == b"data":
                        frame_count += 1
                        
                        # Convert bytes back to numpy array
                        # Expected shape: (nbeam, nchan, npol) = (1, 3072, 4)
                        frame_data = np.frombuffer(data, dtype=np.float32)
                        expected_size = header['nbeam'] * header['nchan'] * header['npol']
                        
                        if frame_data.size == expected_size:
                            # Reshape to (nbeam, nchan, npol)
                            frame_data = frame_data.reshape(header['nbeam'], header['nchan'], header['npol'])
                            
                            # Extract basic statistics
                            pol0_data = frame_data[0, :, 0]  # Extract pol=0
                            
                            # Print frame information
                            current_time = time.time()
                            elapsed = current_time - start_time
                            
                            print(f"Frame {frame_count:4d} | "
                                  f"Time: {elapsed:6.1f}s | "
                                  f"Shape: {frame_data.shape} | "
                                  f"Pol0: min={pol0_data.min():8.3f}, "
                                  f"max={pol0_data.max():8.3f}, "
                                  f"mean={pol0_data.mean():8.3f} | "
                                  f"Timestamp: {header.get('timestamp', 'N/A')}")
                            
                            # Optional: Save first few frames for inspection
                            if frame_count <= 3:
                                filename = f"frame_{frame_count:03d}.npy"
                                np.save(filename, frame_data)
                                print(f"  -> Saved to {filename}")
                                
                        else:
                            print(f"Warning: Unexpected data size {frame_data.size}, expected {expected_size}")
                    else:
                        print(f"Unknown topic: {topic}")
                        
            except zmq.Again:
                # Timeout - no message received
                print(".", end="", flush=True)
                time.sleep(0.1)
                continue
                
    except KeyboardInterrupt:
        print(f"\n\nStopped after receiving {frame_count} frames")
        print(f"Total time: {time.time() - start_time:.1f} seconds")
        if frame_count > 0:
            print(f"Average rate: {frame_count / (time.time() - start_time):.2f} frames/second")
    
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        socket.close()
        context.term()
        print("ZMQ connection closed")

def print_header_info(header):
    """Print detailed header information"""
    print("\n=== Header Information ===")
    for key, value in header.items():
        print(f"  {key}: {value}")
    print("========================\n")

def analyze_data_structure(frame_data, header):
    """Analyze the structure of received data"""
    print(f"\n=== Data Analysis ===")
    print(f"Shape: {frame_data.shape}")
    print(f"Data type: {frame_data.dtype}")
    print(f"Memory size: {frame_data.nbytes} bytes")
    
    # Analyze each polarization
    for pol in range(header['npol']):
        pol_data = frame_data[0, :, pol]
        print(f"Pol {pol}: min={pol_data.min():.6f}, max={pol_data.max():.6f}, mean={pol_data.mean():.6f}")
    
    print("=====================\n")

if __name__ == "__main__":
    main()
