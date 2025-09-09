#!/usr/bin/env python3

"""
Example script showing how to run stress tests from a remote machine
This demonstrates the complete workflow for testing the StreamReceiver web interface.
"""

import requests
import time
import subprocess
import sys
import os
from datetime import datetime

def test_connectivity(url):
    """Test basic connectivity to the web server"""
    print(f"Testing connectivity to {url}...")
    
    try:
        response = requests.get(f"{url}/", timeout=10)
        if response.status_code == 200:
            print("✅ Web server is accessible")
            return True
        else:
            print(f"❌ Web server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to web server: {e}")
        return False

def run_stress_test(test_type, url, output_dir):
    """Run a specific stress test"""
    print(f"\n{'='*60}")
    print(f"Running {test_type} stress test")
    print(f"{'='*60}")
    
    # Map test types to commands
    commands = {
        'quick': ['python3', 'simple_stress_test.py', '--url', url, '--requests', '50', '--threads', '10'],
        'data': ['python3', 'data_endpoint_stress.py', '--url', url, '--requests', '100', '--threads', '20'],
        'comprehensive': ['python3', 'stress_test_web.py', '--url', url, '--threads', '20', '--duration', '30', '--test-type', 'mixed']
    }
    
    if test_type not in commands:
        print(f"❌ Unknown test type: {test_type}")
        return False
    
    cmd = commands[test_type]
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Test completed successfully")
            print("Output:")
            print(result.stdout)
            return True
        else:
            print(f"❌ Test failed with return code {result.returncode}")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def main():
    """Main function demonstrating remote stress testing"""
    
    # Configuration
    TARGET_URL = "http://192.168.1.100:9527"  # Change this to your target server
    OUTPUT_DIR = f"stress_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("StreamReceiver Web Interface Remote Stress Testing Example")
    print("=" * 60)
    print(f"Target URL: {TARGET_URL}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print()
    
    # Step 1: Test connectivity
    print("Step 1: Testing connectivity...")
    if not test_connectivity(TARGET_URL):
        print("❌ Cannot connect to target server. Please check:")
        print("  1. The web server is running on the target machine")
        print("  2. The URL is correct")
        print("  3. Network connectivity between machines")
        print("  4. Firewall settings")
        sys.exit(1)
    
    # Step 2: Create output directory
    print("\nStep 2: Creating output directory...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ Created directory: {OUTPUT_DIR}")
    
    # Step 3: Run quick stress test
    print("\nStep 3: Running quick stress test...")
    if not run_stress_test('quick', TARGET_URL, OUTPUT_DIR):
        print("❌ Quick stress test failed")
        sys.exit(1)
    
    # Step 4: Run data endpoint test
    print("\nStep 4: Running data endpoint stress test...")
    if not run_stress_test('data', TARGET_URL, OUTPUT_DIR):
        print("❌ Data endpoint stress test failed")
        sys.exit(1)
    
    # Step 5: Run comprehensive test
    print("\nStep 5: Running comprehensive stress test...")
    if not run_stress_test('comprehensive', TARGET_URL, OUTPUT_DIR):
        print("❌ Comprehensive stress test failed")
        sys.exit(1)
    
    # Step 6: Summary
    print("\n" + "="*60)
    print("STRESS TESTING COMPLETED")
    print("="*60)
    print("✅ All tests completed successfully!")
    print(f"📁 Results saved in: {OUTPUT_DIR}")
    print()
    print("Next steps:")
    print("1. Review the test results")
    print("2. Check server performance during tests")
    print("3. Analyze any performance bottlenecks")
    print("4. Optimize server configuration if needed")
    print("5. Run tests regularly to monitor performance")

if __name__ == "__main__":
    main()
