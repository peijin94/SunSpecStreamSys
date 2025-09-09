#!/usr/bin/env python3

"""
Comprehensive stress test runner for the StreamReceiver web interface
This script runs multiple stress tests and provides a comprehensive report.
"""

import subprocess
import time
import argparse
import sys
import os
from datetime import datetime

class StressTestRunner:
    """Runner for multiple stress tests"""
    
    def __init__(self, base_url, output_dir="./stress_test_results"):
        self.base_url = base_url
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Test scripts
        self.tests = {
            'connectivity': {
                'script': 'test_web_interface.py',
                'description': 'Basic connectivity test',
                'timeout': 30
            },
            'simple_stress': {
                'script': 'simple_stress_test.py',
                'description': 'Simple stress test (100 requests, 10 threads)',
                'timeout': 60,
                'args': ['--requests', '100', '--threads', '10']
            },
            'data_endpoint': {
                'script': 'data_endpoint_stress.py',
                'description': 'Data endpoint stress test (200 requests, 20 threads)',
                'timeout': 120,
                'args': ['--requests', '5000', '--threads', '100']
            },
            'comprehensive': {
                'script': 'stress_test_web.py',
                'description': 'Comprehensive stress test (mixed workload, 60s)',
                'timeout': 90,
                'args': ['--threads', '20', '--duration', '60', '--test-type', 'mixed']
            }
        }
    
    def run_test(self, test_name, test_config):
        """Run a single stress test"""
        print(f"\n{'='*60}")
        print(f"Running: {test_config['description']}")
        print(f"{'='*60}")
        
        # Build command
        script_path = os.path.join(os.path.dirname(__file__), test_config['script'])
        cmd = [sys.executable, script_path, '--url', self.base_url]
        
        if 'args' in test_config:
            cmd.extend(test_config['args'])
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Timeout: {test_config['timeout']}s")
        
        start_time = time.time()
        
        try:
            # Run the test
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=test_config['timeout']
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Store results
            self.results[test_name] = {
                'success': result.returncode == 0,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
            # Print results
            if result.returncode == 0:
                print(f"✅ Test completed successfully in {duration:.2f}s")
            else:
                print(f"❌ Test failed with return code {result.returncode}")
                print(f"Duration: {duration:.2f}s")
            
            # Save output to file
            output_file = os.path.join(self.output_dir, f"{test_name}_output.txt")
            with open(output_file, 'w') as f:
                f.write(f"Test: {test_config['description']}\n")
                f.write(f"URL: {self.base_url}\n")
                f.write(f"Duration: {duration:.2f}s\n")
                f.write(f"Return Code: {result.returncode}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write("\n" + "="*60 + "\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n" + "="*60 + "\n")
                f.write("STDERR:\n")
                f.write(result.stderr)
            
            print(f"Output saved to: {output_file}")
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Test timed out after {test_config['timeout']}s")
            self.results[test_name] = {
                'success': False,
                'duration': test_config['timeout'],
                'stdout': '',
                'stderr': 'Test timed out',
                'returncode': -1
            }
        except Exception as e:
            print(f"❌ Error running test: {e}")
            self.results[test_name] = {
                'success': False,
                'duration': 0,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            }
    
    def run_all_tests(self, test_list=None):
        """Run all or selected stress tests"""
        if test_list is None:
            test_list = list(self.tests.keys())
        
        print(f"Starting stress test suite for {self.base_url}")
        print(f"Output directory: {self.output_dir}")
        print(f"Tests to run: {', '.join(test_list)}")
        
        overall_start = time.time()
        
        for test_name in test_list:
            if test_name in self.tests:
                self.run_test(test_name, self.tests[test_name])
                
                # Brief pause between tests
                print("Waiting 5 seconds before next test...")
                time.sleep(5)
            else:
                print(f"⚠️  Unknown test: {test_name}")
        
        overall_end = time.time()
        self.print_summary(overall_end - overall_start)
    
    def print_summary(self, total_duration):
        """Print summary of all tests"""
        print(f"\n{'='*80}")
        print("STRESS TEST SUITE SUMMARY")
        print(f"{'='*80}")
        
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Tests Run: {len(self.results)}")
        
        successful = sum(1 for r in self.results.values() if r['success'])
        failed = len(self.results) - successful
        
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(successful/len(self.results)*100):.1f}%")
        
        print(f"\nDetailed Results:")
        print(f"{'Test Name':<20} {'Status':<10} {'Duration':<10} {'Notes'}")
        print(f"{'-'*60}")
        
        for test_name, result in self.results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = f"{result['duration']:.1f}s"
            
            notes = []
            if result['returncode'] == -1:
                notes.append("timeout")
            elif result['returncode'] != 0:
                notes.append(f"exit code {result['returncode']}")
            
            note_str = ", ".join(notes) if notes else "OK"
            
            print(f"{test_name:<20} {status:<10} {duration:<10} {note_str}")
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate a summary report file"""
        report_file = os.path.join(self.output_dir, "summary_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("StreamReceiver Web Interface Stress Test Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Target URL: {self.base_url}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            successful = sum(1 for r in self.results.values() if r['success'])
            failed = len(self.results) - successful
            
            f.write(f"Summary:\n")
            f.write(f"  Total Tests: {len(self.results)}\n")
            f.write(f"  Successful: {successful}\n")
            f.write(f"  Failed: {failed}\n")
            f.write(f"  Success Rate: {(successful/len(self.results)*100):.1f}%\n\n")
            
            f.write("Test Results:\n")
            f.write("-" * 30 + "\n")
            
            for test_name, result in self.results.items():
                f.write(f"\n{test_name}:\n")
                f.write(f"  Status: {'PASS' if result['success'] else 'FAIL'}\n")
                f.write(f"  Duration: {result['duration']:.2f}s\n")
                f.write(f"  Return Code: {result['returncode']}\n")
                
                if result['stderr']:
                    f.write(f"  Error: {result['stderr']}\n")
        
        print(f"\nSummary report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive stress tests on StreamReceiver web interface")
    parser.add_argument('--url', type=str, default='http://localhost:9527',
                       help='Base URL of the web server')
    parser.add_argument('--output-dir', type=str, default='./stress_test_results',
                       help='Directory to save test results')
    parser.add_argument('--tests', type=str, nargs='+',
                       choices=['connectivity', 'simple_stress', 'data_endpoint', 'comprehensive'],
                       help='Specific tests to run (default: all)')
    parser.add_argument('--quick', action='store_true',
                       help='Run only quick tests (connectivity + simple_stress)')
    
    args = parser.parse_args()
    
    # Determine which tests to run
    if args.quick:
        test_list = ['connectivity', 'simple_stress']
    elif args.tests:
        test_list = args.tests
    else:
        test_list = None  # Run all tests
    
    # Create and run stress test suite
    runner = StressTestRunner(args.url, args.output_dir)
    runner.run_all_tests(test_list)

if __name__ == "__main__":
    main()
