#!/usr/bin/env python3

"""
Test script for StreamReceiver plotting functionality
This script tests the plotting system without requiring an active stream.
"""

import time
import numpy as np
import os
from stream_receiver import StreamReceiver

def test_plotting_functionality():
    """Test the plotting functionality of StreamReceiver"""
    
    print("Testing StreamReceiver Plotting Functionality")
    print("=" * 50)
    
    # Test plot directory
    test_plot_dir = '/fast/peijinz/streaming/figs/'
    print(f"Test plot directory: {test_plot_dir}")
    
    # Create receiver with short plotting interval for testing
    receiver = StreamReceiver(
        stream_addr='127.0.0.1',
        stream_port=9798,
        buffer_length=10,  # Small buffer for testing
        plot_interval=2,   # Plot every 2 seconds for testing
        plot_dir=test_plot_dir
    )
    
    print("Receiver created with plotting enabled")
    print(f"Plot interval: {receiver.plot_interval}s")
    print(f"Plot directory: {receiver.plot_dir}")
    
    try:
        # Start the receiver
        print("\nStarting receiver...")
        receiver.start()
        
        # Simulate some data to test plotting
        print("Simulating data for plotting test...")
        for i in range(5):
            # Generate some test data
            test_data = np.random.random((768,)).astype(np.float32) * (1 + i * 0.1)
            
            # Manually update buffer (simulating received data)
            receiver.ring_buffer[receiver.buffer_index, :] = test_data
            receiver.buffer_index = (receiver.buffer_index + 1) % receiver.buffer_length
            
            print(f"Added test data frame {i+1}, buffer_index: {receiver.buffer_index}")
            
            # Wait for plotting to occur
            time.sleep(3)
        
        # Check if plots were created
        print("\nChecking for created plots...")
        plot_files = [f for f in os.listdir(test_plot_dir) if f.startswith('spectrum_') and f.endswith('.png')]
        
        if plot_files:
            print(f"Found {len(plot_files)} plot files:")
            for plot_file in sorted(plot_files):
                file_path = os.path.join(test_plot_dir, plot_file)
                file_size = os.path.getsize(file_path)
                print(f"  {plot_file} ({file_size} bytes)")
        else:
            print("No plot files found. This might indicate an issue with the plotting system.")
        
        # Get status information
        status = receiver.get_buffer_status()
        print(f"\nFinal status:")
        print(f"  Buffer index: {status['buffer_index']}")
        print(f"  Plot running: {status['plot_running']}")
        print(f"  Last plot time: {status['last_plot_time']}")
        
        print("\nPlotting test completed!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop the receiver
        print("\nStopping receiver...")
        receiver.stop()
        print("Test completed.")

def test_plot_creation():
    """Test the plot creation method directly"""
    
    print("\n" + "=" * 50)
    print("Testing plot creation method directly...")
    
    # Create receiver
    receiver = StreamReceiver(
        stream_addr='127.0.0.1',
        stream_port=9798,
        buffer_length=5,
        plot_interval=1,
        plot_dir='/fast/peijinz/streaming/figs/'
    )
    
    # Add some test data
    for i in range(3):
        test_data = np.random.random((768,)).astype(np.float32) * (1 + i * 0.2)
        receiver.ring_buffer[receiver.buffer_index, :] = test_data
        receiver.buffer_index = (receiver.buffer_index + 1) % receiver.buffer_length
    
    print(f"Added {receiver.buffer_index} test frames")
    
    # Test plot creation
    print("Testing plot creation...")
    receiver.create_plot()
    
    # Check if plot was created
    plot_files = [f for f in os.listdir('/fast/peijinz/streaming/figs/') if f.startswith('spectrum_') and f.endswith('.png')]
    if plot_files:
        latest_plot = sorted(plot_files)[-1]
        print(f"Latest plot created: {latest_plot}")
    else:
        print("No plots found")
    
    print("Direct plot creation test completed!")

def test_disabled_plotting():
    """Test that plotting is properly disabled when plot_interval=0"""
    
    print("\n" + "=" * 50)
    print("Testing disabled plotting functionality...")
    
    # Create receiver with plotting disabled
    receiver = StreamReceiver(
        stream_addr='127.0.0.1',
        stream_port=9798,
        buffer_length=5,
        plot_interval=0,  # Disable plotting
        plot_dir='/fast/peijinz/streaming/figs/'
    )
    
    print("Receiver created with plotting disabled (plot_interval=0)")
    
    try:
        # Start the receiver
        print("Starting receiver...")
        receiver.start()
        
        # Add some test data
        for i in range(3):
            test_data = np.random.random((768,)).astype(np.float32) * (1 + i * 0.2)
            receiver.ring_buffer[receiver.buffer_index, :] = test_data
            receiver.buffer_index = (receiver.buffer_index + 1) % receiver.buffer_length
        
        print(f"Added {receiver.buffer_index} test frames")
        
        # Wait a bit to see if any plotting occurs
        print("Waiting 5 seconds to verify no plotting occurs...")
        time.sleep(5)
        
        # Check status
        status = receiver.get_buffer_status()
        print(f"Status: plot_running={status['plot_running']}, last_plot_time={status['last_plot_time']}")
        
        # Verify plotting is disabled
        if not status['plot_running']:
            print("✓ Plotting correctly disabled")
        else:
            print("✗ Plotting incorrectly enabled")
        
        # Test that create_plot does nothing when disabled
        print("Testing create_plot when disabled...")
        receiver.create_plot()
        
        print("Disabled plotting test completed!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        receiver.stop()

if __name__ == "__main__":
    print("StreamReceiver Plotting Test Script")
    print("This script tests the new plotting functionality")
    print("=" * 50)
    
    # Test plotting functionality
    test_plotting_functionality()
    
    # Test direct plot creation
    test_plot_creation()
    
    # Test disabled plotting
    test_disabled_plotting()
    
    print("\nAll tests completed!")
    print("Check the plots directory for generated images:")
    print("/fast/peijinz/streaming/figs/")
