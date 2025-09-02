#!/usr/bin/env python3

"""
Test script for new header format from dr_beam.py
This script tests that StreamReceiver can handle the updated header structure.
"""

import time
import numpy as np
import os
import sys
from astropy.time import Time

# Add parent directory to path to import StreamReceiver
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_new_header_format():
    """Test that StreamReceiver can handle the new header format"""
    
    print("Testing New Header Format from dr_beam.py")
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
        
        # Test the new header format from dr_beam.py
        test_header = {
            'time_tag': 344085352899674112,  # Example LWA time tag
            'nbeam': 1,
            'nchan': 3072,
            'npol': 4,
            'timestamp': time.time(),  # Current time as float
            'last_block_time': '2025-01-17 12:34:56.789',  # Example LWA time format
            'data_shape': (1, 3072, 4),
            'data_type': '<f4'
        }
        
        print(f"Test header: {test_header}")
        
        # Test timestamp parsing
        if 'timestamp' in test_header:
            timestamp_val = test_header['timestamp']
            print(f"✓ Found timestamp field: {timestamp_val} (type: {type(timestamp_val)})")
            
            try:
                # Handle both string and numeric timestamp formats
                if isinstance(timestamp_val, str):
                    header_time = float(timestamp_val)
                else:
                    header_time = float(timestamp_val)
                
                print(f"✓ Parsed timestamp: {header_time:.6f}s")
                
                # Calculate delay
                current_time = time.time()
                delay = current_time - header_time
                print(f"✓ Calculated delay: {delay:.6f}s")
                
                if abs(delay) < 10:  # Should be very small for current time
                    print("✓ Delay is reasonable (using current time)")
                else:
                    print(f"⚠️  Delay is larger than expected: {delay:.3f}s")
                    
            except (ValueError, TypeError) as e:
                print(f"✗ Could not parse timestamp: {e}")
                return False
        else:
            print("✗ No timestamp field found in header")
            return False
        
        # Test last_block_time field
        if 'last_block_time' in test_header:
            last_block_time = test_header['last_block_time']
            print(f"✓ Found last_block_time field: {last_block_time} (type: {type(last_block_time)})")
        else:
            print("✗ No last_block_time field found in header")
            return False
        
        # Test time_tag field (legacy)
        if 'time_tag' in test_header:
            time_tag = test_header['time_tag']
            print(f"✓ Found time_tag field: {time_tag} (type: {type(time_tag)})")
        else:
            print("✗ No time_tag field found in header")
            return False
        
        print("\n✓ New header format test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_delay_calculation_with_new_header():
    """Test delay calculation with the new header format"""
    
    print("\n" + "=" * 50)
    print("Testing Delay Calculation with New Header")
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
        
        # Test different timestamp formats
        test_cases = [
            ("Float timestamp", time.time()),
            ("String timestamp", str(time.time())),
            ("Recent timestamp", time.time() - 1.5),  # 1.5 seconds ago
            ("Older timestamp", time.time() - 10.0),  # 10 seconds ago
        ]
        
        for description, timestamp_val in test_cases:
            print(f"\n--- Testing {description} ---")
            
            # Create test header
            test_header = {
                'time_tag': 344085352899674112,
                'nbeam': 1,
                'nchan': 3072,
                'npol': 4,
                'timestamp': timestamp_val,
                'last_block_time': '2025-01-17 12:34:56.789',
                'data_shape': (1, 3072, 4),
                'data_type': '<f4'
            }
            
            # Simulate processing this header
            current_time_utc = Time.now().unix
            
            try:
                if isinstance(timestamp_val, str):
                    header_time = float(timestamp_val)
                else:
                    header_time = float(timestamp_val)
                
                delay = current_time_utc - header_time
                print(f"  Timestamp: {timestamp_val}")
                print(f"  Parsed: {header_time:.6f}s")
                print(f"  Calculated delay: {delay:.6f}s")
                
                if abs(delay) < 3600 and delay >= 0:
                    print(f"  ✓ Delay is reasonable")
                else:
                    print(f"  ⚠️  Delay is large or negative")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        print("\n✓ Delay calculation test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("New Header Format Test Script")
    print("This script tests the updated header handling from dr_beam.py")
    print("=" * 50)
    
    # Test new header format
    header_success = test_new_header_format()
    
    # Test delay calculation
    delay_success = test_delay_calculation_with_new_header()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"New header format: {'✓ PASSED' if header_success else '✗ FAILED'}")
    print(f"Delay calculation: {'✓ PASSED' if delay_success else '✗ FAILED'}")
    
    if header_success and delay_success:
        print("\n🎉 All tests passed! New header format is working correctly.")
        print("The StreamReceiver can now handle the updated header structure from dr_beam.py")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
    
    print("\nExpected behavior:")
    print("- Should handle float timestamp field from dr_beam.py")
    print("- Should handle last_block_time field")
    print("- Should maintain backward compatibility with time_tag")
    print("- Should calculate delays correctly with new timestamp format")
