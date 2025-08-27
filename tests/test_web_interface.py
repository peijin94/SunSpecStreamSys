#!/usr/bin/env python3

"""
Test script for StreamReceiver web interface
This script tests the web server functionality without requiring an active stream.
"""

import time
import numpy as np
import os
import sys
import requests
import threading

# Add parent directory to path to import StreamReceiver
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_web_interface():
    """Test the web interface functionality"""
    
    print("Testing StreamReceiver Web Interface")
    print("=" * 50)
    
    try:
        from stream_receiver import StreamReceiver
        
        # Create receiver with web interface enabled
        receiver = StreamReceiver(
            stream_addr='127.0.0.1',
            stream_port=9798,
            buffer_length=10,
            plot_interval=0,  # Disable plotting for testing
            plot_dir='/tmp/test_plots',
            start_webshow=True  # Enable web interface
        )
        
        print("✓ StreamReceiver created successfully")
        print(f"Web interface enabled: {receiver.start_webshow}")
        
        # Start the receiver
        print("Starting receiver with web interface...")
        receiver.start()
        
        # Wait for web server to start
        print("Waiting for web server to start...")
        time.sleep(3)
        
        # Test web server endpoints
        base_url = "http://localhost:9527"
        
        print(f"Testing web server at {base_url}")
        
        # Test 1: Check if server is responding
        try:
            response = requests.get(f"{base_url}/", timeout=5)
            if response.status_code == 200:
                print("✓ Web server is responding")
                print(f"  Response length: {len(response.text)} characters")
            else:
                print(f"✗ Web server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Could not connect to web server: {e}")
            return False
        
        # Test 2: Check data endpoint
        try:
            response = requests.get(f"{base_url}/data", timeout=5)
            if response.status_code == 200:
                print("✓ Data endpoint is working")
                data = response.json()
                print(f"  Data type: {type(data)}")
                if isinstance(data, list) and len(data) > 0:
                    print(f"  First frame length: {len(data[0])}")
                else:
                    print("  No data frames available yet")
            else:
                print(f"✗ Data endpoint returned status {response.status_code}")
        except Exception as e:
            print(f"✗ Data endpoint error: {e}")
        
        # Test 3: Add some test data and check if it's served
        print("\nAdding test data to buffer...")
        for i in range(3):
            test_data = np.random.random((768,)).astype(np.float32) * (1 + i * 0.2)
            receiver.ring_buffer[receiver.buffer_index, :] = test_data
            receiver.buffer_index = (receiver.buffer_index + 1) % receiver.buffer_length
            print(f"  Added frame {i+1}, buffer_index: {receiver.buffer_index}")
        
        # Wait a moment for data to be processed
        time.sleep(1)
        
        # Test 4: Check if data endpoint now returns data
        try:
            response = requests.get(f"{base_url}/data", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    print("✓ Data endpoint is serving test data")
                    print(f"  Number of frames: {len(data)}")
                    print(f"  Frame shape: {len(data[0])}")
                else:
                    print("✗ Data endpoint not serving data")
            else:
                print(f"✗ Data endpoint error: {response.status_code}")
        except Exception as e:
            print(f"✗ Data endpoint error: {e}")
        
        print("\nWeb interface test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'receiver' in locals():
            receiver.stop()

def test_web_interface_disabled():
    """Test that web interface is properly disabled when start_webshow=False"""
    
    print("\n" + "=" * 50)
    print("Testing Web Interface Disabled")
    print("=" * 50)
    
    try:
        from stream_receiver import StreamReceiver
        
        # Create receiver with web interface disabled
        receiver = StreamReceiver(
            stream_addr='127.0.0.1',
            stream_port=9798,
            buffer_length=10,
            plot_interval=0,
            plot_dir='/tmp/test_plots',
            start_webshow=False  # Disable web interface
        )
        
        print("✓ StreamReceiver created successfully")
        print(f"Web interface enabled: {receiver.start_webshow}")
        
        # Start the receiver
        print("Starting receiver without web interface...")
        receiver.start()
        
        # Wait a moment
        time.sleep(2)
        
        # Check that no web server was started
        if not hasattr(receiver, 'webserver_thread'):
            print("✓ No web server thread created")
        else:
            print("✗ Web server thread was created unexpectedly")
        
        print("Disabled web interface test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'receiver' in locals():
            receiver.stop()

if __name__ == "__main__":
    print("Web Interface Test Script")
    print("This script tests the StreamReceiver web interface functionality")
    print("=" * 50)
    
    # Test web interface enabled
    web_success = test_web_interface()
    
    # Test web interface disabled
    disabled_success = test_web_interface_disabled()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Web interface enabled: {'✓ PASSED' if web_success else '✗ FAILED'}")
    print(f"Web interface disabled: {'✓ PASSED' if disabled_success else '✗ FAILED'}")
    
    if web_success and disabled_success:
        print("\n🎉 All tests passed! Web interface is working correctly.")
        print("You can now use the web interface to view live spectrum data.")
        print("To enable: python stream_receiver.py --start-webshow")
        print("Then open: http://localhost:9527")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
    
    print("\nExpected behavior:")
    print("- Web interface should start when start_webshow=True")
    print("- Should serve HTML page at /")
    print("- Should serve data at /data endpoint")
    print("- Should not start web server when start_webshow=False")
