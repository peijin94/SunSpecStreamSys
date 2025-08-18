#!/usr/bin/env python3

"""
Debug script for time_tag parsing issues
This script helps diagnose problems with delay calculations.
"""

import os
import sys
import time
import numpy as np
from astropy.time import Time

def test_time_tag_formats():
    """Test different time_tag formats to understand the issue"""
    
    print("Testing Time Tag Formats")
    print("=" * 50)
    
    current_time = time.time()
    current_time_utc = Time.now().unix
    print(f"Current time (epoch): {current_time}")
    print(f"Current time (UTC epoch): {current_time_utc}")
    print(f"Current time (readable): {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current time (UTC): {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
    
    # Test different time formats
    test_cases = [
        ("Current epoch seconds", current_time),
        ("Current epoch microseconds", int(current_time * 1e6)),
        ("Current epoch nanoseconds", int(current_time * 1e9)),
        ("Small relative time", 0.5),
        ("Large negative number", -342329810474.9),
        ("OVRO 196 MHz ticks", 344085352899674112),  # Your actual time_tag
        ("String format", time.strftime('%Y-%m-%d %H:%M:%S')),
        ("Compact format", time.strftime('%Y%m%d_%H%M%S')),
    ]
    
    for description, time_tag in test_cases:
        print(f"\n--- {description} ---")
        print(f"Time tag: {time_tag}")
        print(f"Type: {type(time_tag)}")
        
        try:
            if isinstance(time_tag, (int, float)):
                # Try different interpretations
                if time_tag > 1e15:
                    interpreted = time_tag / 1e9
                    print(f"  Interpreted as nanoseconds: {interpreted}")
                    delay = current_time - interpreted
                    print(f"  Calculated delay: {delay:.3f}s")
                elif time_tag > 1e12:
                    interpreted = time_tag / 1e6
                    print(f"  Interpreted as microseconds: {interpreted}")
                    delay = current_time - interpreted
                    print(f"  Calculated delay: {delay:.3f}s")
                elif time_tag > 1e9:
                    interpreted = time_tag
                    print(f"  Interpreted as seconds: {interpreted}")
                    delay = current_time - interpreted
                    print(f"  Calculated delay: {delay:.3f}s")
                else:
                    print(f"  Small number: {time_tag}")
                    delay = current_time - time_tag
                    print(f"  Calculated delay: {delay:.3f}s")
                
                # Sanity check
                if abs(delay) > 3600:
                    print(f"  ⚠️  WARNING: Unreasonable delay: {delay:.3f}s")
                elif delay < 0:
                    print(f"  ⚠️  WARNING: Negative delay: {delay:.3f}s")
                else:
                    print(f"  ✓ Reasonable delay: {delay:.3f}s")
                
                # Special handling for OVRO tick format
                if time_tag > 1e15:
                    ovro_ticks_sec = time_tag / 196000000.0
                    ovro_delay = current_time - ovro_ticks_sec
                    print(f"  🎯 OVRO 196 MHz ticks: {ovro_ticks_sec:.6f}s")
                    print(f"  🎯 OVRO delay: {ovro_delay:.6f}s")
                    
                    if abs(ovro_delay) < 3600 and ovro_delay >= 0:
                        print(f"    ✓ OVRO ticks give reasonable delay!")
                    else:
                        print(f"    ⚠️  OVRO ticks still give unreasonable delay")
                    
            elif isinstance(time_tag, str):
                try:
                    parsed_time = time.mktime(time.strptime(time_tag, "%Y-%m-%d %H:%M:%S"))
                    delay = current_time - parsed_time
                    print(f"  Parsed as datetime: {parsed_time}")
                    print(f"  Calculated delay: {delay:.3f}s")
                except ValueError:
                    try:
                        parsed_time = time.mktime(time.strptime(time_tag, "%Y%m%d_%H%M%S"))
                        delay = current_time - parsed_time
                        print(f"  Parsed as compact datetime: {parsed_time}")
                        print(f"  Calculated delay: {delay:.3f}s")
                    except ValueError:
                        print(f"  ❌ Could not parse string format")
                        
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_delay_calculation():
    """Test the delay calculation logic"""
    
    print("\n" + "=" * 50)
    print("Testing Delay Calculation Logic")
    print("=" * 50)
    
    # Simulate the problematic time_tag
    problematic_time_tag = -342329810474.9
    current_time = time.time()
    
    print(f"Problematic time_tag: {problematic_time_tag}")
    print(f"Current time: {current_time}")
    
    # Try different interpretations
    interpretations = [
        ("as seconds", problematic_time_tag),
        ("as microseconds", problematic_time_tag / 1e6),
        ("as nanoseconds", problematic_time_tag / 1e9),
    ]
    
    for desc, interpreted in interpretations:
        delay = current_time - interpreted
        print(f"  {desc}: delay = {delay:.3f}s")
        
        if abs(delay) > 3600:
            print(f"    ⚠️  Unreasonable delay")
        elif delay < 0:
            print(f"    ⚠️  Negative delay")
        else:
            print(f"    ✓ Reasonable delay")

if __name__ == "__main__":
    print("Time Tag Debug Script")
    print("This script helps diagnose time_tag parsing issues")
    print("=" * 50)
    
    test_time_tag_formats()
    test_delay_calculation()
    
    print("\n" + "=" * 50)
    print("DEBUG SUMMARY")
    print("=" * 50)
    print("If you see large negative delays, the time_tag format is likely incorrect.")
    print("Check the header data to see what format the time_tag is actually in.")
    print("Common issues:")
    print("  - Wrong time unit (nanoseconds vs microseconds vs seconds)")
    print("  - Wrong epoch (different reference time)")
    print("  - Relative time vs absolute time")
    print("  - String format vs numeric format")
