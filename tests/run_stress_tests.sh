#!/bin/bash

# Stress test runner script for StreamReceiver web interface
# This script makes it easy to run stress tests from another machine

# Default values
URL="http://localhost:9527"
OUTPUT_DIR="./stress_test_results"
TEST_TYPE="all"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "StreamReceiver Web Interface Stress Test Runner"
    echo "=============================================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -u, --url URL          Target URL (default: http://localhost:9527)"
    echo "  -o, --output DIR       Output directory (default: ./stress_test_results)"
    echo "  -t, --test TYPE        Test type: all, quick, connectivity, simple, data, comprehensive"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all tests on localhost:9527"
    echo "  $0 -u http://192.168.1.100:9527      # Test remote server"
    echo "  $0 -t quick                           # Run only quick tests"
    echo "  $0 -t data -o /tmp/results           # Run only data endpoint test"
    echo ""
    echo "Test Types:"
    echo "  all            - Run all stress tests (default)"
    echo "  quick          - Run only connectivity and simple stress tests"
    echo "  connectivity   - Basic connectivity test only"
    echo "  simple         - Simple stress test (100 requests, 5 threads)"
    echo "  data           - Data endpoint stress test (200 requests, 10 threads)"
    echo "  comprehensive  - Comprehensive stress test (mixed workload, 60s)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--url)
            URL="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -t|--test)
            TEST_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate URL format
if [[ ! $URL =~ ^https?:// ]]; then
    print_error "Invalid URL format: $URL"
    print_error "URL must start with http:// or https://"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "python3 is not installed or not in PATH"
    exit 1
fi

# Check if required Python packages are available
print_status "Checking Python dependencies..."
python3 -c "import requests" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "Required Python package 'requests' is not installed"
    print_error "Install with: pip install requests"
    exit 1
fi

# Create output directory
print_status "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Print test configuration
echo ""
print_status "Stress Test Configuration:"
echo "  Target URL: $URL"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Test Type: $TEST_TYPE"
echo ""

# Test connectivity first
print_status "Testing basic connectivity..."
python3 -c "
import requests
import sys
try:
    response = requests.get('$URL/', timeout=10)
    if response.status_code == 200:
        print('✅ Web server is accessible')
        sys.exit(0)
    else:
        print('❌ Web server returned status', response.status_code)
        sys.exit(1)
except Exception as e:
    print('❌ Cannot connect to web server:', e)
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    print_error "Cannot connect to web server at $URL"
    print_error "Please check:"
    print_error "  1. The StreamReceiver is running with --start-webshow"
    print_error "  2. The URL is correct (default: http://localhost:9527)"
    print_error "  3. Network connectivity"
    print_error "  4. Firewall settings"
    exit 1
fi

# Run the appropriate stress test
print_status "Starting stress tests..."

case $TEST_TYPE in
    "all")
        print_status "Running comprehensive stress test suite..."
        python3 run_stress_tests.py --url "$URL" --output-dir "$OUTPUT_DIR"
        ;;
    "quick")
        print_status "Running quick stress tests..."
        python3 run_stress_tests.py --url "$URL" --output-dir "$OUTPUT_DIR" --quick
        ;;
    "connectivity")
        print_status "Running connectivity test..."
        python3 test_web_interface.py --url "$URL"
        ;;
    "simple")
        print_status "Running simple stress test..."
        python3 simple_stress_test.py --url "$URL" --requests 100 --threads 5
        ;;
    "data")
        print_status "Running data endpoint stress test..."
        python3 data_endpoint_stress.py --url "$URL" --requests 200 --threads 10
        ;;
    "comprehensive")
        print_status "Running comprehensive stress test..."
        python3 stress_test_web.py --url "$URL" --threads 15 --duration 60 --test-type mixed
        ;;
    *)
        print_error "Unknown test type: $TEST_TYPE"
        show_usage
        exit 1
        ;;
esac

# Check if tests completed successfully
if [ $? -eq 0 ]; then
    print_success "Stress tests completed successfully!"
    print_status "Results saved in: $OUTPUT_DIR"
    
    # List result files
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        print_status "Generated files:"
        ls -la "$OUTPUT_DIR"
    fi
else
    print_error "Stress tests failed or were interrupted"
    exit 1
fi

echo ""
print_status "Stress testing completed!"
