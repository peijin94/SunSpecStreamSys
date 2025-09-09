#!/usr/bin/env python3

"""
Test script for StreamReceiver web interface
This script tests the web server functionality assuming it's already running.
"""

import requests
import argparse
import sys

def test_web_interface(base_url):
    """Test the web interface functionality"""
    
    print("Testing StreamReceiver Web Interface")
    print("=" * 50)
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
                print(f"  Number of frames: {len(data)}")
                if isinstance(data[0], list):
                    print(f"  First frame length: {len(data[0])}")
                else:
                    print(f"  First frame type: {type(data[0])}")
            else:
                print("  No data frames available yet")
        else:
            print(f"✗ Data endpoint returned status {response.status_code}")
    except Exception as e:
        print(f"✗ Data endpoint error: {e}")
    
    print("\nWeb interface test completed!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test StreamReceiver web interface")
    parser.add_argument('--url', type=str, default='http://localhost:9527',
                       help='Base URL of the web server')
    
    args = parser.parse_args()
    
    print("Web Interface Test Script")
    print("This script tests the StreamReceiver web interface functionality")
    print("=" * 50)
    
    # Test web interface
    web_success = test_web_interface(args.url)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Web interface test: {'✓ PASSED' if web_success else '✗ FAILED'}")
    
    if web_success:
        print("\n🎉 Web interface test passed! The web server is working correctly.")
        print("You can access the web interface at the provided URL.")
    else:
        print("\n❌ Web interface test failed. Check the error messages above.")
        print("Make sure the web server is running and accessible.")
    
    print("\nExpected behavior:")
    print("- Web server should be running and accessible")
    print("- Should serve HTML page at /")
    print("- Should serve data at /data endpoint")
