#!/usr/bin/env python3

"""
Stress test script for the StreamReceiver web interface
This script can be run from another machine to test the web server under load.
"""

import requests
import time
import threading
import random
import statistics
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import sys

class WebStressTester:
    """Stress tester for the StreamReceiver web interface"""
    
    def __init__(self, base_url, num_threads=20, duration=60, verbose=False):
        self.base_url = base_url.rstrip('/')
        self.num_threads = num_threads
        self.duration = duration
        self.verbose = verbose
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'errors': {},
            'start_time': None,
            'end_time': None
        }
        
        # Thread-safe lock for statistics
        self.stats_lock = threading.Lock()
        
        # Control flags
        self.running = False
        self.shutdown_event = threading.Event()
        
        print(f"WebStressTester initialized:")
        print(f"  Target URL: {self.base_url}")
        print(f"  Threads: {self.num_threads}")
        print(f"  Duration: {self.duration}s")
        print(f"  Verbose: {self.verbose}")
    
    def make_request(self, endpoint, timeout=5):
        """Make a single HTTP request and record statistics"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            response = requests.get(url, timeout=timeout)
            response_time = time.time() - start_time
            
            with self.stats_lock:
                self.stats['total_requests'] += 1
                self.stats['response_times'].append(response_time)
                
                if response.status_code == 200:
                    self.stats['successful_requests'] += 1
                    if self.verbose:
                        print(f"✓ {endpoint}: {response.status_code} ({response_time:.3f}s)")
                else:
                    self.stats['failed_requests'] += 1
                    error_key = f"HTTP_{response.status_code}"
                    self.stats['errors'][error_key] = self.stats['errors'].get(error_key, 0) + 1
                    if self.verbose:
                        print(f"✗ {endpoint}: {response.status_code} ({response_time:.3f}s)")
            
            return {
                'success': response.status_code == 200,
                'status_code': response.status_code,
                'response_time': response_time,
                'content_length': len(response.content) if response.content else 0
            }
            
        except requests.exceptions.Timeout:
            response_time = time.time() - start_time
            with self.stats_lock:
                self.stats['total_requests'] += 1
                self.stats['failed_requests'] += 1
                self.stats['response_times'].append(response_time)
                self.stats['errors']['TIMEOUT'] = self.stats['errors'].get('TIMEOUT', 0) + 1
            
            if self.verbose:
                print(f"✗ {endpoint}: TIMEOUT ({response_time:.3f}s)")
            
            return {
                'success': False,
                'status_code': 'TIMEOUT',
                'response_time': response_time,
                'content_length': 0
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            with self.stats_lock:
                self.stats['total_requests'] += 1
                self.stats['failed_requests'] += 1
                self.stats['response_times'].append(response_time)
                error_key = f"EXCEPTION_{type(e).__name__}"
                self.stats['errors'][error_key] = self.stats['errors'].get(error_key, 0) + 1
            
            if self.verbose:
                print(f"✗ {endpoint}: {type(e).__name__} ({response_time:.3f}s)")
            
            return {
                'success': False,
                'status_code': type(e).__name__,
                'response_time': response_time,
                'content_length': 0
            }
    
    def worker_thread(self, thread_id):
        """Worker thread that makes requests continuously"""
        endpoints = ['/', '/data']
        request_count = 0
        
        while not self.shutdown_event.is_set():
            # Randomly select an endpoint
            endpoint = random.choice(endpoints)
            
            # Make the request
            result = self.make_request(endpoint)
            request_count += 1
            
            # Add some random delay between requests (0.1 to 1.0 seconds)
            delay = random.uniform(0.1, 1.0)
            time.sleep(delay)
        
        if self.verbose:
            print(f"Thread {thread_id} completed {request_count} requests")
    
    def burst_worker(self, thread_id, burst_size=10):
        """Worker that makes bursts of requests"""
        endpoints = ['/', '/data']
        
        while not self.shutdown_event.is_set():
            # Make a burst of requests
            for _ in range(burst_size):
                if self.shutdown_event.is_set():
                    break
                
                endpoint = random.choice(endpoints)
                self.make_request(endpoint)
                
                # Small delay between requests in burst
                time.sleep(0.05)
            
            # Longer delay between bursts
            time.sleep(random.uniform(1.0, 3.0))
    
    def data_analysis_worker(self, thread_id):
        """Worker that focuses on data analysis (frequent /data requests)"""
        while not self.shutdown_event.is_set():
            # Make data request
            result = self.make_request('/data')
            
            # If we got data, analyze it
            if result['success'] and result['content_length'] > 0:
                try:
                    # Make a quick request to get the actual data
                    response = requests.get(f"{self.base_url}/data", timeout=2)
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list) and len(data) > 0:
                            # Simulate some analysis
                            time.sleep(0.1)
                except:
                    pass
            
            # Wait before next request
            time.sleep(0.5)
    
    def start_stress_test(self, test_type='mixed'):
        """Start the stress test"""
        print(f"\nStarting stress test: {test_type}")
        print(f"Duration: {self.duration} seconds")
        print(f"Threads: {self.num_threads}")
        print("-" * 50)
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Create thread pool
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            
            # Start worker threads based on test type
            if test_type == 'mixed':
                # Mixed workload: different types of workers
                for i in range(self.num_threads):
                    if i < self.num_threads // 3:
                        future = executor.submit(self.worker_thread, i)
                    elif i < 2 * self.num_threads // 3:
                        future = executor.submit(self.burst_worker, i)
                    else:
                        future = executor.submit(self.data_analysis_worker, i)
                    futures.append(future)
            
            elif test_type == 'sustained':
                # Sustained load: all threads making regular requests
                for i in range(self.num_threads):
                    future = executor.submit(self.worker_thread, i)
                    futures.append(future)
            
            elif test_type == 'burst':
                # Burst load: all threads making bursts
                for i in range(self.num_threads):
                    future = executor.submit(self.burst_worker, i)
                    futures.append(future)
            
            elif test_type == 'data_heavy':
                # Data-heavy: focus on /data endpoint
                for i in range(self.num_threads):
                    future = executor.submit(self.data_analysis_worker, i)
                    futures.append(future)
            
            # Monitor progress
            start_time = time.time()
            while time.time() - start_time < self.duration and not self.shutdown_event.is_set():
                time.sleep(5)  # Report every 5 seconds
                self.print_progress()
            
            # Stop the test
            self.shutdown_event.set()
            self.stats['end_time'] = time.time()
            
            # Wait for all threads to complete
            for future in as_completed(futures, timeout=10):
                try:
                    future.result()
                except Exception as e:
                    if self.verbose:
                        print(f"Thread error: {e}")
        
        self.running = False
        self.print_final_results()
    
    def print_progress(self):
        """Print current progress"""
        with self.stats_lock:
            elapsed = time.time() - self.stats['start_time']
            total = self.stats['total_requests']
            successful = self.stats['successful_requests']
            failed = self.stats['failed_requests']
            
            if total > 0:
                success_rate = (successful / total) * 100
                avg_response_time = statistics.mean(self.stats['response_times']) if self.stats['response_times'] else 0
                requests_per_sec = total / elapsed if elapsed > 0 else 0
                
                print(f"[{elapsed:.1f}s] Requests: {total} | "
                      f"Success: {success_rate:.1f}% | "
                      f"Avg Response: {avg_response_time:.3f}s | "
                      f"Rate: {requests_per_sec:.1f} req/s")
    
    def print_final_results(self):
        """Print final test results"""
        print("\n" + "=" * 60)
        print("STRESS TEST RESULTS")
        print("=" * 60)
        
        with self.stats_lock:
            total_time = self.stats['end_time'] - self.stats['start_time']
            total_requests = self.stats['total_requests']
            successful = self.stats['successful_requests']
            failed = self.stats['failed_requests']
            
            print(f"Test Duration: {total_time:.2f} seconds")
            print(f"Total Requests: {total_requests}")
            print(f"Successful: {successful} ({(successful/total_requests*100):.1f}%)")
            print(f"Failed: {failed} ({(failed/total_requests*100):.1f}%)")
            print(f"Requests/sec: {total_requests/total_time:.2f}")
            
            if self.stats['response_times']:
                response_times = self.stats['response_times']
                print(f"\nResponse Times:")
                print(f"  Average: {statistics.mean(response_times):.3f}s")
                print(f"  Median: {statistics.median(response_times):.3f}s")
                print(f"  Min: {min(response_times):.3f}s")
                print(f"  Max: {max(response_times):.3f}s")
                print(f"  95th percentile: {self.percentile(response_times, 95):.3f}s")
                print(f"  99th percentile: {self.percentile(response_times, 99):.3f}s")
            
            if self.stats['errors']:
                print(f"\nErrors:")
                for error_type, count in self.stats['errors'].items():
                    print(f"  {error_type}: {count}")
            
            # Performance assessment
            print(f"\nPerformance Assessment:")
            if successful / total_requests > 0.95:
                print("  ✓ Excellent: >95% success rate")
            elif successful / total_requests > 0.90:
                print("  ✓ Good: >90% success rate")
            elif successful / total_requests > 0.80:
                print("  ⚠️  Fair: >80% success rate")
            else:
                print("  ✗ Poor: <80% success rate")
            
            if self.stats['response_times']:
                avg_response = statistics.mean(self.stats['response_times'])
                if avg_response < 0.5:
                    print("  ✓ Excellent: <0.5s average response time")
                elif avg_response < 1.0:
                    print("  ✓ Good: <1.0s average response time")
                elif avg_response < 2.0:
                    print("  ⚠️  Fair: <2.0s average response time")
                else:
                    print("  ✗ Poor: >2.0s average response time")
    
    def percentile(self, data, percentile):
        """Calculate percentile of data"""
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

def test_connectivity(base_url):
    """Test basic connectivity to the web server"""
    print(f"Testing connectivity to {base_url}...")
    
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("✓ Web server is accessible")
            return True
        else:
            print(f"✗ Web server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to web server: {e}")
        print("Make sure the StreamReceiver web server is running with --start-webshow")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Stress test the StreamReceiver web interface")
    parser.add_argument('--url', type=str, default='http://localhost:9527',
                       help='Base URL of the web server (default: http://localhost:9527)')
    parser.add_argument('--threads', type=int, default=20,
                       help='Number of concurrent threads (default: 20)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in seconds (default: 60)')
    parser.add_argument('--test-type', type=str, default='mixed',
                       choices=['mixed', 'sustained', 'burst', 'data_heavy'],
                       help='Type of stress test (default: mixed)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("StreamReceiver Web Interface Stress Tester")
    print("=" * 50)
    
    # Test connectivity first
    if not test_connectivity(args.url):
        print("Cannot connect to web server. Please check:")
        print("1. The StreamReceiver is running with --start-webshow")
        print("2. The URL is correct (default: http://localhost:9527)")
        print("3. Network connectivity")
        print("4. Firewall settings")
        sys.exit(1)
    
    # Create and run stress tester
    tester = WebStressTester(
        base_url=args.url,
        num_threads=args.threads,
        duration=args.duration,
        verbose=args.verbose
    )
    
    try:
        tester.start_stress_test(test_type=args.test_type)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        tester.shutdown_event.set()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
