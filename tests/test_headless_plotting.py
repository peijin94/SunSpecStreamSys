#!/usr/bin/env python3

"""
Test script for headless plotting functionality
This script verifies that plotting works without a display on headless servers.
"""

import os
import sys
import time
import numpy as np

# Set environment variables for headless operation
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''

def test_headless_plotting():
    """Test that plotting works in headless mode"""
    
    print("Testing Headless Plotting Functionality")
    print("=" * 50)
    
    try:
        # Import matplotlib after setting environment variables
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        print(f"✓ Matplotlib backend: {matplotlib.get_backend()}")
        print(f"✓ Environment MPLBACKEND: {os.environ.get('MPLBACKEND', 'Not set')}")
        print(f"✓ Environment DISPLAY: '{os.environ.get('DISPLAY', 'Not set')}'")
        
        # Test creating a simple plot
        print("\nCreating test plot...")
        
        # Generate some test data
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, 100)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, y, 'b-', linewidth=2)
        ax.set_title('Test Plot - Headless Mode')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        test_plot_dir = '/fast/peijinz/streaming/figs/'
        os.makedirs(test_plot_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(test_plot_dir, f"headless_test_{timestamp}.png")
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Test plot saved: {filename}")
        
        # Check file was created
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"✓ File exists with size: {file_size} bytes")
        else:
            print("✗ File was not created")
            return False
        
        print("\n✓ Headless plotting test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_stream_receiver_import():
    """Test that StreamReceiver can be imported without display errors"""
    
    print("\n" + "=" * 50)
    print("Testing StreamReceiver import in headless mode...")
    
    try:
        # Import StreamReceiver
        from stream_receiver import StreamReceiver
        
        print("✓ StreamReceiver imported successfully")
        
        # Test creating instance
        receiver = StreamReceiver(
            stream_addr='127.0.0.1',
            stream_port=9798,
            buffer_length=10,
            plot_interval=5,  # Short interval for testing
            plot_dir='/fast/peijinz/streaming/figs/'
        )
        
        print("✓ StreamReceiver instance created successfully")
        print(f"  Plot interval: {receiver.plot_interval}")
        print(f"  Plot directory: {receiver.plot_dir}")
        
        # Clean up
        del receiver
        print("✓ StreamReceiver instance cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Headless Plotting Test Script")
    print("This script verifies plotting works without a display")
    print("=" * 50)
    
    # Test basic headless plotting
    plot_success = test_headless_plotting()
    
    # Test StreamReceiver import
    import_success = test_stream_receiver_import()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Headless plotting: {'✓ PASSED' if plot_success else '✗ FAILED'}")
    print(f"StreamReceiver import: {'✓ PASSED' if import_success else '✗ FAILED'}")
    
    if plot_success and import_success:
        print("\n🎉 All tests passed! Headless plotting is working correctly.")
        print("You can now run the StreamReceiver without display issues.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
    
    print(f"\nCheck the plots directory for test images:")
    print("/fast/peijinz/streaming/figs/")
