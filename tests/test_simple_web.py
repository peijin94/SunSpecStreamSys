#!/usr/bin/env python3

"""
Simple test script to verify web interface with updated settings
"""

import time
import numpy as np
from stream_receiver import StreamReceiver

def test_simple_web():
    """Test the web interface with simplified settings"""
    
    print("Testing Simple Web Interface")
    print("=" * 40)
    
    # Create receiver with web interface enabled
    receiver = StreamReceiver(
        stream_addr='127.0.0.1',
        stream_port=9798,
        buffer_length=10,
        plot_interval=0,
        start_webshow=True
    )
    
    print(f"Web interface enabled: {receiver.start_webshow}")
    
    try:
        # Start the receiver
        print("Starting receiver...")
        receiver.start()
        
        # Wait for web server to start
        print("Waiting for web server to start...")
        time.sleep(3)
        
        # Check web server status
        status = receiver.get_webserver_status()
        print(f"Web server status: {status}")
        
        # Add some test data with the specified range (10^4 to 10^9)
        print("\nAdding test data in range 10^4 to 10^9...")
        for i in range(5):
            # Generate data in the specified range
            base_value = 1e4 + (i * 1e8)  # Start at 10^4, increment by 10^8
            test_data = np.random.uniform(base_value, base_value * 1.1, (768,)).astype(np.float32)
            
            receiver.ring_buffer[receiver.buffer_index, :] = test_data
            receiver.buffer_index = (receiver.buffer_index + 1) % receiver.buffer_length
            print(f"  Added frame {i+1}, buffer_index: {receiver.buffer_index}")
            print(f"  Data range: {test_data.min():.2e} to {test_data.max():.2e}")
        
        print(f"\n✅ Web interface test completed!")
        print(f"Web server should be running at: http://localhost:9527")
        print(f"Data range: 10^4 to 10^9")
        print(f"Update rate: Every 0.5 seconds")
        print(f"Press Ctrl+C to stop...")
        
        # Keep running to test web interface
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Stopping receiver...")
        receiver.stop()
        print("Test completed.")

if __name__ == "__main__":
    test_simple_web()
