#!/usr/bin/env python3

"""
Test script for StreamReceiver
This script demonstrates how to use the StreamReceiver class.
"""

import time
import numpy as np
from stream_receiver import StreamReceiver

def test_receiver():
    """Test the StreamReceiver functionality"""
    
    # Create receiver instance
    receiver = StreamReceiver(
        stream_addr='127.0.0.1',
        stream_port=9798,
        buffer_length=1200
    )
    
    print("StreamReceiver created. Starting...")
    
    try:
        # Start the receiver
        receiver.start()
        
        # Monitor for some time
        print("Monitoring stream for 30 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 30:
            time.sleep(2)
            
            # Get status
            status = receiver.get_buffer_status()
            print(f"Status: Buffer index: {status['buffer_index']}, "
                  f"Running: {status['is_running']}, "
                  f"Buffer shape: {status['buffer_shape']}")
            
            # If we have some data, show a sample
            if status['buffer_index'] > 0:
                latest_data = receiver.get_latest_data(n_frames=1)
                if latest_data.size > 0:
                    print(f"Latest data sample: min={latest_data.min():.6f}, "
                          f"max={latest_data.max():.6f}, "
                          f"mean={latest_data.mean():.6f}")
        
        print("Test completed.")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        # Stop the receiver
        receiver.stop()
        print("Receiver stopped.")

def test_ring_buffer():
    """Test the ring buffer functionality"""
    
    print("\nTesting ring buffer functionality...")
    
    # Create receiver with small buffer for testing
    receiver = StreamReceiver(
        stream_addr='127.0.0.1',
        stream_port=9798,
        buffer_length=5  # Small buffer for testing
    )
    
    # Simulate some data updates
    test_data = np.random.random((1536,)).astype(np.float32)
    
    print("Simulating data updates...")
    for i in range(10):
        # Manually update buffer (simulating received data)
        receiver.ring_buffer[receiver.buffer_index, :] = test_data + i * 0.1
        receiver.buffer_index = (receiver.buffer_index + 1) % receiver.buffer_length
        
        print(f"Update {i}: buffer_index = {receiver.buffer_index}")
        
        # Get latest data
        latest = receiver.get_latest_data(n_frames=3)
        print(f"  Latest 3 frames shape: {latest.shape}")
    
    print("Ring buffer test completed.")

if __name__ == "__main__":
    print("StreamReceiver Test Script")
    print("=" * 40)
    
    # Test ring buffer functionality
    test_ring_buffer()
    
    # Test actual receiver (only if stream is available)
    print("\n" + "=" * 40)
    print("Note: To test actual streaming, make sure AvgStreamingOp is running")
    print("and streaming data to localhost:9798")
    
    response = input("\nDo you want to test actual streaming? (y/n): ")
    if response.lower() == 'y':
        test_receiver()
    else:
        print("Skipping streaming test.")
    
    print("\nTest script completed.")
