#!/bin/bash

# Deploy stress testing tools to a remote machine
# This script copies the stress testing tools to another machine for testing

# Default values
REMOTE_HOST=""
REMOTE_USER=""
REMOTE_DIR="~/stress_tests"
LOCAL_DIR="$(dirname "$0")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

show_usage() {
    echo "Deploy Stress Testing Tools"
    echo "=========================="
    echo ""
    echo "Usage: $0 -h HOST [-u USER] [-d DIR]"
    echo ""
    echo "Options:"
    echo "  -h, --host HOST        Remote hostname or IP address (required)"
    echo "  -u, --user USER        Remote username (default: current user)"
    echo "  -d, --dir DIR          Remote directory (default: ~/stress_tests)"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -h 192.168.1.100                    # Deploy to 192.168.1.100"
    echo "  $0 -h test-server -u tester            # Deploy as user 'tester'"
    echo "  $0 -h 192.168.1.100 -d /tmp/stress     # Deploy to /tmp/stress"
    echo ""
    echo "Prerequisites:"
    echo "  - SSH access to the remote host"
    echo "  - Python 3 installed on remote host"
    echo "  - 'requests' Python package on remote host"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        -u|--user)
            REMOTE_USER="$2"
            shift 2
            ;;
        -d|--dir)
            REMOTE_DIR="$2"
            shift 2
            ;;
        --help)
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

# Validate required parameters
if [ -z "$REMOTE_HOST" ]; then
    print_error "Remote host is required"
    show_usage
    exit 1
fi

# Set default user if not provided
if [ -z "$REMOTE_USER" ]; then
    REMOTE_USER=$(whoami)
fi

print_status "Deploying stress testing tools..."
print_status "Remote host: $REMOTE_HOST"
print_status "Remote user: $REMOTE_USER"
print_status "Remote directory: $REMOTE_DIR"
print_status "Local directory: $LOCAL_DIR"

# Test SSH connectivity
print_status "Testing SSH connectivity..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE_USER@$REMOTE_HOST" "echo 'SSH connection successful'" 2>/dev/null; then
    print_error "Cannot connect to $REMOTE_USER@$REMOTE_HOST"
    print_error "Please check:"
    print_error "  1. SSH access is configured"
    print_error "  2. The hostname/IP is correct"
    print_error "  3. SSH keys are set up (or use password authentication)"
    exit 1
fi

print_success "SSH connection successful"

# Create remote directory
print_status "Creating remote directory..."
if ! ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'" 2>/dev/null; then
    print_error "Cannot create remote directory: $REMOTE_DIR"
    exit 1
fi

print_success "Remote directory created"

# Copy stress testing files
print_status "Copying stress testing files..."

# List of files to copy
FILES=(
    "stress_test_web.py"
    "simple_stress_test.py"
    "data_endpoint_stress.py"
    "run_stress_tests.py"
    "run_stress_tests.sh"
    "example_remote_test.py"
    "STRESS_TEST_README.md"
)

# Copy each file
for file in "${FILES[@]}"; do
    if [ -f "$LOCAL_DIR/$file" ]; then
        print_status "Copying $file..."
        if scp "$LOCAL_DIR/$file" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" 2>/dev/null; then
            print_success "Copied $file"
        else
            print_error "Failed to copy $file"
            exit 1
        fi
    else
        print_warning "File not found: $file"
    fi
done

# Make shell script executable
print_status "Making shell script executable..."
if ssh "$REMOTE_USER@$REMOTE_HOST" "chmod +x '$REMOTE_DIR/run_stress_tests.sh'" 2>/dev/null; then
    print_success "Shell script made executable"
else
    print_warning "Could not make shell script executable"
fi

# Check Python dependencies
print_status "Checking Python dependencies on remote host..."
if ssh "$REMOTE_USER@$REMOTE_HOST" "python3 -c 'import requests' 2>/dev/null"; then
    print_success "Python 'requests' package is available"
else
    print_warning "Python 'requests' package not found"
    print_status "Installing Python dependencies..."
    if ssh "$REMOTE_USER@$REMOTE_HOST" "pip3 install requests" 2>/dev/null; then
        print_success "Python dependencies installed"
    else
        print_error "Failed to install Python dependencies"
        print_error "Please install manually: pip3 install requests"
    fi
fi

# Test the deployment
print_status "Testing deployment..."
if ssh "$REMOTE_USER@$REMOTE_HOST" "cd '$REMOTE_DIR' && python3 simple_stress_test.py --help" 2>/dev/null; then
    print_success "Deployment test successful"
else
    print_warning "Deployment test failed - tools may not work correctly"
fi

# Print usage instructions
echo ""
print_success "Deployment completed!"
echo ""
print_status "Usage instructions:"
echo "1. SSH to the remote host:"
echo "   ssh $REMOTE_USER@$REMOTE_HOST"
echo ""
echo "2. Navigate to the stress test directory:"
echo "   cd $REMOTE_DIR"
echo ""
echo "3. Run stress tests:"
echo "   # Quick test"
echo "   ./run_stress_tests.sh -u http://TARGET_IP:9527 -t quick"
echo ""
echo "   # All tests"
echo "   ./run_stress_tests.sh -u http://TARGET_IP:9527"
echo ""
echo "   # Specific test"
echo "   python3 simple_stress_test.py --url http://TARGET_IP:9527"
echo ""
echo "4. View results:"
echo "   ls -la stress_test_results/"
echo ""
print_status "Replace TARGET_IP with the actual IP address of your StreamReceiver server"


