#!/usr/bin/env python3

"""
Specialized stress test for the /data endpoint
This script focuses specifically on testing the data endpoint which serves spectrum data.
"""

import requests
import time
import threading
import json
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
import statistics

class DataEndpointStressTester:
    """Stress tester specifically for the /data endpoint"""
    
    def __init__(self, base_url, num_requests=200, num_threads=20, request_interval=0.1):
        self.base_url = base_url.rstrip('/')
        self.num_requests = num_requests
        self.num_threads = num_threads
        self.request_interval = request_interval
        
        self.results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'data_sizes': [],
            'errors': {},
            'data_quality_issues': 0
        }
        self.lock = threading.Lock()
    
    def test_data_endpoint(self):
        """Test the /data endpoint and analyze the response"""
        url = f"{self.base_url}/data"
        start_time = time.time()
        
        try:
            response = requests.get(url, timeout=10)
            response_time = time.time() - start_time
            
            with self.lock:
                self.results['total_requests'] += 1
                self.results['response_times'].append(response_time)
                
                if response.status_code == 200:
                    self.results['successful_requests'] += 1
                    
                    # Analyze the data
                    try:
                        data = response.json()
                        self.results['data_sizes'].append(len(response.content))
                        
                        # Check data quality
                        if self.analyze_data_quality(data):
                            self.results['data_quality_issues'] += 1
                        
                        # Only print on errors or every 10th request
                        if self.results['total_requests'] % 10 == 0:
                            print(f"✓ Data: {response.status_code} ({response_time:.3f}s) - "
                                  f"Size: {len(response.content)} bytes - "
                                  f"Frames: {len(data) if isinstance(data, list) else 'N/A'}")
                    except json.JSONDecodeError:
                        print(f"✗ Data: Invalid JSON ({response_time:.3f}s)")
                        self.results['data_quality_issues'] += 1
                else:
                    self.results['failed_requests'] += 1
                    error_key = f"HTTP_{response.status_code}"
                    self.results['errors'][error_key] = self.results['errors'].get(error_key, 0) + 1
                    print(f"✗ Data: {response.status_code} ({response_time:.3f}s)")
            
            return response.status_code == 200
            
        except requests.exceptions.Timeout:
            response_time = time.time() - start_time
            with self.lock:
                self.results['total_requests'] += 1
                self.results['failed_requests'] += 1
                self.results['response_times'].append(response_time)
                self.results['errors']['TIMEOUT'] = self.results['errors'].get('TIMEOUT', 0) + 1
            print(f"✗ Data: TIMEOUT ({response_time:.3f}s)")
            return False
            
        except Exception as e:
            response_time = time.time() - start_time
            with self.lock:
                self.results['total_requests'] += 1
                self.results['failed_requests'] += 1
                self.results['response_times'].append(response_time)
                error_key = f"EXCEPTION_{type(e).__name__}"
                self.results['errors'][error_key] = self.results['errors'].get(error_key, 0) + 1
            print(f"✗ Data: {type(e).__name__} ({response_time:.3f}s)")
            return False
    
    def analyze_data_quality(self, data):
        """Analyze the quality of the received data"""
        issues = 0
        
        if not isinstance(data, list):
            print("  ⚠️  Data is not a list")
            issues += 1
            return issues > 0
        
        if len(data) == 0:
            print("  ⚠️  Data is empty")
            issues += 1
            return issues > 0
        
        # Check first frame - handle both list and direct array formats
        first_frame = data
        if isinstance(first_frame, list):
            # Data is in list format
            if len(first_frame) != 768:
                print(f"  ⚠️  First frame has wrong length: {len(first_frame)} (expected 768)")
                issues += 1
            else:
                # Check for reasonable data values
                try:
                    values = [float(x) for x in first_frame]
                    if all(v == 0 for v in values):
                        print("  ⚠️  All values are zero")
                        issues += 1
                    elif any(v < 0 for v in values):
                        print("  ⚠️  Some values are negative")
                        issues += 1
                    elif any(not isinstance(v, (int, float)) or v != v for v in values):  # Check for NaN
                        print("  ⚠️  Some values are NaN or invalid")
                        issues += 1
                except (ValueError, TypeError):
                    print("  ⚠️  Values are not numeric")
                    issues += 1
        elif hasattr(first_frame, '__len__'):
            # Data is in array format (numpy array or similar)
            if len(first_frame) != 768:
                print(f"  ⚠️  First frame has wrong length: {len(first_frame)} (expected 768)")
                issues += 1
            else:
                # Check for reasonable data values
                try:
                    values = [float(x) for x in first_frame]
                    if all(v == 0 for v in values):
                        print("  ⚠️  All values are zero")
                        issues += 1
                    elif any(v < 0 for v in values):
                        print("  ⚠️  Some values are negative")
                        issues += 1
                    elif any(not isinstance(v, (int, float)) or v != v for v in values):  # Check for NaN
                        print("  ⚠️  Some values are NaN or invalid")
                        issues += 1
                except (ValueError, TypeError):
                    print("  ⚠️  Values are not numeric")
                    issues += 1
        else:
            print(f"  ⚠️  First frame is not a list or array (type: {type(first_frame)})")
            issues += 1
        
        return issues > 0
    
    def worker(self, thread_id):
        """Worker thread that makes data requests"""
        requests_per_thread = self.num_requests // self.num_threads
        
        for i in range(requests_per_thread):
            self.test_data_endpoint()
            time.sleep(self.request_interval)
    
    def run_stress_test(self):
        """Run the stress test"""
        print(f"Starting data endpoint stress test:")
        print(f"  URL: {self.base_url}")
        print(f"  Total requests: {self.num_requests}")
        print(f"  Threads: {self.num_threads}")
        print(f"  Request interval: {self.request_interval}s")
        print("-" * 50)
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.worker, i) for i in range(self.num_threads)]
            
            # Wait for all threads to complete
            for future in futures:
                future.result()
        
        end_time = time.time()
        self.print_results(end_time - start_time)
    
    def print_results(self, duration):
        """Print detailed test results"""
        print("\n" + "=" * 60)
        print("DATA ENDPOINT STRESS TEST RESULTS")
        print("=" * 60)
        
        total = self.results['total_requests']
        success = self.results['successful_requests']
        failed = self.results['failed_requests']
        
        print(f"Test Duration: {duration:.2f} seconds")
        print(f"Total Requests: {total}")
        print(f"Successful: {success} ({(success/total*100):.1f}%)")
        print(f"Failed: {failed} ({(failed/total*100):.1f}%)")
        print(f"Requests/sec: {total/duration:.2f}")
        
        if self.results['response_times']:
            times = self.results['response_times']
            print(f"\nResponse Times:")
            print(f"  Average: {statistics.mean(times):.3f}s")
            print(f"  Median: {statistics.median(times):.3f}s")
            print(f"  Min: {min(times):.3f}s")
            print(f"  Max: {max(times):.3f}s")
            print(f"  95th percentile: {self.percentile(times, 95):.3f}s")
        
        if self.results['data_sizes']:
            sizes = self.results['data_sizes']
            print(f"\nData Sizes:")
            print(f"  Average: {statistics.mean(sizes):.0f} bytes")
            print(f"  Min: {min(sizes)} bytes")
            print(f"  Max: {max(sizes)} bytes")
        
        print(f"\nData Quality Issues: {self.results['data_quality_issues']}")
        
        if self.results['errors']:
            print(f"\nErrors:")
            for error, count in self.results['errors'].items():
                print(f"  {error}: {count}")
        
        # Performance assessment
        print(f"\nPerformance Assessment:")
        success_rate = success / total if total > 0 else 0
        if success_rate > 0.95:
            print("  ✅ Excellent: >95% success rate")
        elif success_rate > 0.90:
            print("  ✅ Good: >90% success rate")
        elif success_rate > 0.80:
            print("  ⚠️  Fair: >80% success rate")
        else:
            print("  ❌ Poor: <80% success rate")
        
        if self.results['response_times']:
            avg_response = statistics.mean(self.results['response_times'])
            if avg_response < 0.5:
                print("  ✅ Excellent: <0.5s average response time")
            elif avg_response < 1.0:
                print("  ✅ Good: <1.0s average response time")
            elif avg_response < 2.0:
                print("  ⚠️  Fair: <2.0s average response time")
            else:
                print("  ❌ Poor: >2.0s average response time")
    
    def percentile(self, data, percentile):
        """Calculate percentile of data"""
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

def test_data_endpoint(url):
    """Test if the data endpoint is accessible and working"""
    try:
        response = requests.get(f"{url}/data", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Data endpoint accessible - {len(data)} frames available")
            return True
        else:
            print(f"❌ Data endpoint returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot access data endpoint: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Stress test the /data endpoint")
    parser.add_argument('--url', type=str, default='http://localhost:9527',
                       help='Base URL of the web server')
    parser.add_argument('--requests', type=int, default=200,
                       help='Total number of requests to make')
    parser.add_argument('--threads', type=int, default=20,
                       help='Number of concurrent threads')
    parser.add_argument('--interval', type=float, default=0.1,
                       help='Interval between requests in seconds')
    
    args = parser.parse_args()
    
    print("Data Endpoint Stress Tester")
    print("=" * 40)
    
    # Test data endpoint first
    if not test_data_endpoint(args.url):
        print("Cannot access data endpoint. Please check:")
        print("1. The StreamReceiver is running with --start-webshow")
        print("2. The URL is correct (default: http://localhost:9527)")
        print("3. The /data endpoint is working")
        print("4. There is data in the buffer")
        sys.exit(1)
    
    # Run stress test
    tester = DataEndpointStressTester(
        args.url, 
        args.requests, 
        args.threads, 
        args.interval
    )
    tester.run_stress_test()

if __name__ == "__main__":
    main()
