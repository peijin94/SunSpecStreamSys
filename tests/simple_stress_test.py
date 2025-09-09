#!/usr/bin/env python3

"""
Simple stress test script for the StreamReceiver web interface
This is a lightweight version for quick testing from another machine.
"""

import requests
import time
import threading
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

class SimpleStressTester:
    """Simple stress tester for the web interface"""
    
    def __init__(self, base_url, num_requests=100, num_threads=10):
        self.base_url = base_url.rstrip('/')
        self.num_requests = num_requests
        self.num_threads = num_threads
        
        self.results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'response_times': [],
            'errors': {}
        }
        self.lock = threading.Lock()
    
    def make_request(self, endpoint):
        """Make a single request and record results"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            response = requests.get(url, timeout=5)
            response_time = time.time() - start_time
            
            with self.lock:
                self.results['total'] += 1
                self.results['response_times'].append(response_time)
                
                if response.status_code == 200:
                    self.results['success'] += 1
                else:
                    self.results['failed'] += 1
                    error_key = f"HTTP_{response.status_code}"
                    self.results['errors'][error_key] = self.results['errors'].get(error_key, 0) + 1
                    print(f"✗ {endpoint}: {response.status_code} ({response_time:.3f}s)")
            
            return response.status_code == 200
            
        except Exception as e:
            response_time = time.time() - start_time
            with self.lock:
                self.results['total'] += 1
                self.results['failed'] += 1
                self.results['response_times'].append(response_time)
                error_key = f"EXCEPTION_{type(e).__name__}"
                self.results['errors'][error_key] = self.results['errors'].get(error_key, 0) + 1
            
            print(f"✗ {endpoint}: {type(e).__name__} ({response_time:.3f}s)")
            return False
    
    def worker(self, thread_id):
        """Worker thread that makes requests"""
        requests_per_thread = self.num_requests // self.num_threads
        endpoints = ['/', '/data']
        
        for i in range(requests_per_thread):
            endpoint = endpoints[i % len(endpoints)]
            self.make_request(endpoint)
            time.sleep(0.1)  # Small delay between requests
    
    def run_test(self):
        """Run the stress test"""
        print(f"Starting stress test:")
        print(f"  URL: {self.base_url}")
        print(f"  Total requests: {self.num_requests}")
        print(f"  Threads: {self.num_threads}")
        print("-" * 40)
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.worker, i) for i in range(self.num_threads)]
            
            # Wait for all threads to complete
            for future in futures:
                future.result()
        
        end_time = time.time()
        self.print_results(end_time - start_time)
    
    def print_results(self, duration):
        """Print test results"""
        print("\n" + "=" * 40)
        print("STRESS TEST RESULTS")
        print("=" * 40)
        
        total = self.results['total']
        success = self.results['success']
        failed = self.results['failed']
        
        print(f"Duration: {duration:.2f} seconds")
        print(f"Total requests: {total}")
        print(f"Successful: {success} ({(success/total*100):.1f}%)")
        print(f"Failed: {failed} ({(failed/total*100):.1f}%)")
        print(f"Requests/sec: {total/duration:.2f}")
        
        if self.results['response_times']:
            times = self.results['response_times']
            print(f"Average response time: {sum(times)/len(times):.3f}s")
            print(f"Min response time: {min(times):.3f}s")
            print(f"Max response time: {max(times):.3f}s")
        
        if self.results['errors']:
            print("\nErrors:")
            for error, count in self.results['errors'].items():
                print(f"  {error}: {count}")

def test_connection(url):
    """Test if the web server is accessible"""
    try:
        response = requests.get(f"{url}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description="Simple stress test for StreamReceiver web interface")
    parser.add_argument('--url', type=str, default='http://localhost:9527',
                       help='Base URL of the web server')
    parser.add_argument('--requests', type=int, default=100,
                       help='Total number of requests to make')
    parser.add_argument('--threads', type=int, default=10,
                       help='Number of concurrent threads')
    
    args = parser.parse_args()
    
    print("Simple Web Interface Stress Tester")
    print("=" * 40)
    
    # Test connection
    if not test_connection(args.url):
        print(f"❌ Cannot connect to {args.url}")
        print("Make sure the StreamReceiver web server is running with --start-webshow")
        print("Default URL: http://localhost:9527")
        sys.exit(1)
    
    print(f"✅ Connected to {args.url}")
    
    # Run stress test
    tester = SimpleStressTester(args.url, args.requests, args.threads)
    tester.run_test()

if __name__ == "__main__":
    main()
