#!/usr/bin/env python3

"""
Quick test script to verify the streaming system works
"""

import time
import threading
from test_streaming import DummyStreamer
from stream_receiver import StreamReceiver

def test_streaming_system():
    """Test the complete streaming system"""
    
    print("Testing Streaming System")
    print("=" * 40)
    
    # Create dummy streamer
    print("1. Creating dummy streamer...")
    streamer = DummyStreamer(stream_interval=0.5)
    
    # Create receiver
    print("2. Creating receiver...")
    receiver = StreamReceiver(buffer_length=10)  # Small buffer for testing
    
    try:
        # Start streamer
        print("3. Starting dummy streamer...")
        streamer.start()
        
        # Wait a moment for streamer to initialize
        time.sleep(1)
        
        # Start receiver
        print("4. Starting receiver...")
        receiver.start()
        
        # Monitor for 10 seconds
        print("5. Monitoring for 10 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 10:
            status = receiver.get_buffer_status()
            print(f"Status: {status}")
            
            if status['buffer_index'] > 0:
                latest_data = receiver.get_latest_data(n_frames=1)
                if latest_data.size > 0:
                    print(f"  Data: shape={latest_data.shape}, mean={latest_data.mean():.6f}")
            
            time.sleep(1)
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
    finally:
        # Cleanup
        print("Cleaning up...")
        receiver.stop()
        streamer.stop()
        print("Cleanup completed.")

if __name__ == "__main__":
    test_streaming_system()
