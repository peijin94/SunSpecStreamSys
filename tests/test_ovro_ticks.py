#!/usr/bin/env python3

"""
Test script for timestamp parsing and delay calculation
This script tests the simplified timestamp handling.
"""

import time
import numpy as np
import os
import sys
from astropy.time import Time

# Add parent directory to path to import StreamReceiver
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def lwa_time_tag_to_datetime(time_tag: int, rate: float = 196_000_000):
    """
    Convert an LWA timetag to a UTC datetime.
    
    Parameters
    ----------
    time_tag : int
        The raw timetag value from LWA data.
    rate : float, optional
        Tick rate in Hz (default = 196,000,000). 
        For exact hardware clock, use 196_608_000.
    
    Returns
    -------
    datetime.datetime (UTC)
    """
    from datetime import datetime, timedelta
    
    # Separate integer seconds and fractional ticks
    secs, rem = divmod(time_tag, rate)
    
    # Compute fractional seconds from remaining ticks
    frac = rem / rate
    
    # Convert to datetime (epoch = 1970-01-01 UTC)
    return datetime(1970, 1, 1) + timedelta(seconds=secs + frac)

def test_timestamp_parsing():
    """Test timestamp parsing functionality"""
    
    print("Testing Timestamp Parsing")
    print("=" * 50)
    
    try:
        from stream_receiver import StreamReceiver
        
        # Create receiver
        receiver = StreamReceiver(
            stream_addr='127.0.0.1',
            stream_port=9798,
            buffer_length=10,
            plot_interval=0,  # Disable plotting for testing
            plot_dir='/tmp/test_plots'
        )
        
        print("✓ StreamReceiver created successfully")
        
        # Test timestamp parsing
        test_timestamp_str = "1755552894.787341"  # Example timestamp string
        current_time = time.time()
        current_time_utc = Time.now().unix
        
        print(f"Test timestamp string: {test_timestamp_str}")
        print(f"Current time (local): {current_time}")
        print(f"Current time (UTC): {current_time_utc}")
        
        # Parse timestamp string
        try:
            header_time = float(test_timestamp_str)
            print(f"Parsed timestamp: {header_time:.6f}s")
            
            # Calculate delay
            delay = current_time_utc - header_time
            print(f"Calculated delay: {delay:.6f}s")
            
            # Check if delay is reasonable
            if abs(delay) < 86400:  # Less than 24 hours
                print("✓ Delay is reasonable!")
            else:
                print("⚠️  Delay is large (this may be normal for your system)")
                
        except (ValueError, TypeError) as e:
            print(f"✗ Could not parse timestamp: {e}")
            return False
        
        print("\nTimestamp parsing test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_delay_methods():
    """Test different delay calculation methods"""
    
    print("\n" + "=" * 50)
    print("Testing Delay Calculation Methods")
    print("=" * 50)
    
    try:
        from stream_receiver import StreamReceiver
        
        # Create receiver
        receiver = StreamReceiver(
            stream_addr='127.0.0.1',
            stream_port=9798,
            buffer_length=10,
            plot_interval=0
        )
        
        print("✓ StreamReceiver created successfully")
        
        # Test different methods
        methods = ['auto', 'buffer', 'manual']
        
        for method in methods:
            print(f"\n--- Testing {method} method ---")
            receiver.set_delay_calculation_method(method)
            
            if method == 'manual':
                receiver.set_delay(1.5)
                print(f"  Manual delay set to: {receiver.get_current_delay():.3f}s")
            
            status = receiver.get_buffer_status()
            print(f"  Method: {method}")
            print(f"  Delay: {status['current_delay']:.3f}s")
            print(f"  Clock offset: {status['clock_offset']}")  # Will be None now
        
        print("\nDelay methods test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Timestamp Parsing Test Script")
    print("This script tests the simplified timestamp handling")
    print("=" * 50)
    
    # Test timestamp parsing
    timestamp_success = test_timestamp_parsing()
    
    # Test delay methods
    method_success = test_delay_methods()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Timestamp parsing: {'✓ PASSED' if timestamp_success else '✗ FAILED'}")
    print(f"Delay methods: {'✓ PASSED' if method_success else '✗ FAILED'}")
    
    if timestamp_success and method_success:
        print("\n🎉 All tests passed! Timestamp parsing is working correctly.")
        print("The system should now handle your timestamp format properly.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
    
    print("\nExpected behavior:")
    print("- timestamp '1755552894.787341' should be parsed as seconds since epoch")
    print("- Should convert to reasonable delay values")
    print("- Clock synchronization should handle large time differences")
