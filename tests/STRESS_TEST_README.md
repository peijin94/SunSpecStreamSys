# Stress Testing Tools for StreamReceiver Web Interface

This directory contains comprehensive stress testing tools for the StreamReceiver web interface. These tools can be run from another machine to test the web server under various load conditions.

## Quick Start

### Prerequisites

**IMPORTANT**: The StreamReceiver web server must already be running before running stress tests.

1. **Start the StreamReceiver with web interface**:
   ```bash
   python3 stream_receiver.py --start-webshow --addr 127.0.0.1 --port 9798
   ```

2. **Verify the web server is accessible**:
   ```bash
   curl http://localhost:9527/
   ```

### From Another Machine

1. **Copy the stress test files** to your testing machine:
   ```bash
   scp -r SunSpecStreamSys/tests/ user@test-machine:~/stress_tests/
   ```

2. **Install dependencies** on the testing machine:
   ```bash
   pip install requests
   ```

3. **Run a quick test**:
   ```bash
   cd stress_tests
   ./run_stress_tests.sh -u http://TARGET_IP:9527 -t quick
   ```

## Available Stress Tests

### 1. Basic Connectivity Test (`test_web_interface.py`)
- **Purpose**: Verify the web server is accessible and basic endpoints work
- **Duration**: ~30 seconds
- **Load**: Light
- **Use case**: Initial verification

```bash
python3 test_web_interface.py --url http://TARGET_IP:9527
```

### 2. Simple Stress Test (`simple_stress_test.py`)
- **Purpose**: Basic load testing with moderate concurrent requests
- **Default**: 100 requests, 10 threads
- **Duration**: ~30-60 seconds
- **Load**: Moderate
- **Use case**: Quick performance check

```bash
python3 simple_stress_test.py --url http://TARGET_IP:9527 --requests 100 --threads 10
```

### 3. Data Endpoint Stress Test (`data_endpoint_stress.py`)
- **Purpose**: Focused testing of the `/data` endpoint (most resource-intensive)
- **Default**: 200 requests, 20 threads
- **Duration**: ~60-120 seconds
- **Load**: High
- **Use case**: Testing data serving performance

```bash
python3 data_endpoint_stress.py --url http://TARGET_IP:9527 --requests 200 --threads 20
```

### 4. Comprehensive Stress Test (`stress_test_web.py`)
- **Purpose**: Full-scale stress testing with multiple workload patterns
- **Default**: 20 threads, 60 seconds, mixed workload
- **Duration**: 60+ seconds
- **Load**: Very High
- **Use case**: Production readiness testing

```bash
python3 stress_test_web.py --url http://TARGET_IP:9527 --threads 20 --duration 60 --test-type mixed
```

### 5. Test Suite Runner (`run_stress_tests.py`)
- **Purpose**: Run multiple tests and generate comprehensive reports
- **Features**: Automated test execution, result aggregation, detailed reporting
- **Use case**: Complete testing workflow

```bash
python3 run_stress_tests.py --url http://TARGET_IP:9527 --output-dir ./results
```

## Test Types and Workloads

### Mixed Workload (`--test-type mixed`)
- 1/3 threads: Regular requests to all endpoints
- 1/3 threads: Burst requests (10 requests per burst)
- 1/3 threads: Data analysis (frequent `/data` requests)

### Sustained Workload (`--test-type sustained`)
- All threads make regular requests with small delays
- Simulates steady user load

### Burst Workload (`--test-type burst`)
- All threads make bursts of requests
- Simulates peak usage periods

### Data-Heavy Workload (`--test-type data_heavy`)
- All threads focus on `/data` endpoint
- Simulates data analysis applications

## Using the Shell Script

The `run_stress_tests.sh` script provides an easy interface for running tests:

```bash
# Run all tests
./run_stress_tests.sh -u http://TARGET_IP:9527

# Run only quick tests
./run_stress_tests.sh -u http://TARGET_IP:9527 -t quick

# Run specific test
./run_stress_tests.sh -u http://TARGET_IP:9527 -t data

# Custom output directory
./run_stress_tests.sh -u http://TARGET_IP:9527 -o /tmp/stress_results
```

## Test Configuration

### Environment Variables
- `STRESS_TEST_URL`: Default target URL
- `STRESS_TEST_THREADS`: Default number of threads
- `STRESS_TEST_DURATION`: Default test duration

### Command Line Options

#### Common Options (all scripts)
- `--url`: Target web server URL
- `--verbose`: Enable verbose output
- `--help`: Show help message

#### Stress Test Specific Options
- `--threads`: Number of concurrent threads
- `--duration`: Test duration in seconds
- `--requests`: Total number of requests to make
- `--interval`: Delay between requests
- `--test-type`: Type of workload pattern

## Interpreting Results

### Success Metrics
- **Success Rate**: >95% = Excellent, >90% = Good, >80% = Fair, <80% = Poor
- **Response Time**: <0.5s = Excellent, <1.0s = Good, <2.0s = Fair, >2.0s = Poor
- **Throughput**: Requests per second (higher is better)

### Common Issues and Solutions

#### High Failure Rate
- **Cause**: Server overload, network issues, or configuration problems
- **Solutions**: 
  - Reduce thread count
  - Increase request intervals
  - Check server resources (CPU, memory)
  - Verify network connectivity

#### High Response Times
- **Cause**: Server performance issues, large data payloads, or network latency
- **Solutions**:
  - Check server CPU and memory usage
  - Optimize data processing
  - Check network latency
  - Consider server-side caching

#### Data Quality Issues
- **Cause**: Empty responses, malformed data, or processing errors
- **Solutions**:
  - Check if data source is active
  - Verify data processing pipeline
  - Check for memory issues
  - Review error logs

## Monitoring During Tests

### Server-Side Monitoring
```bash
# Monitor CPU and memory
top -p $(pgrep -f stream_receiver)

# Monitor network connections
netstat -an | grep :9527

# Monitor disk I/O
iostat -x 1

# Monitor system load
htop
```

### Client-Side Monitoring
```bash
# Monitor network usage
iftop -i eth0

# Monitor system resources
htop

# Monitor Python process
ps aux | grep python
```

## Test Scenarios

### 1. Development Testing
```bash
# Quick verification
./run_stress_tests.sh -t quick

# Basic performance check
./run_stress_tests.sh -t simple
```

### 2. Pre-Production Testing
```bash
# Comprehensive testing
./run_stress_tests.sh -t all

# Focus on data endpoint
./run_stress_tests.sh -t data
```

### 3. Production Load Testing
```bash
# High-load testing
python3 stress_test_web.py --threads 50 --duration 300 --test-type mixed

# Sustained load testing
python3 stress_test_web.py --threads 20 --duration 600 --test-type sustained
```

### 4. Network Testing
```bash
# Test from different network locations
python3 simple_stress_test.py --url http://TARGET_IP:9527 --requests 500 --threads 20

# Test with different request intervals
python3 data_endpoint_stress.py --url http://TARGET_IP:9527 --interval 0.05
```

## Troubleshooting

### Common Error Messages

#### "Cannot connect to web server"
- Check if the web server is running
- Verify the URL and port
- Check firewall settings
- Test network connectivity

#### "Timeout" errors
- Server is overloaded
- Network latency is high
- Increase timeout values
- Reduce concurrent load

#### "HTTP 500" errors
- Server-side errors
- Check server logs
- Verify data source is active
- Check server resources

#### "Data quality issues"
- Empty data responses
- Malformed JSON
- Check data processing pipeline
- Verify buffer has data

### Debug Mode
Run tests with verbose output to see detailed information:
```bash
python3 stress_test_web.py --url http://TARGET_IP:9527 --verbose
```

## Best Practices

1. **Start Small**: Begin with quick tests and gradually increase load
2. **Monitor Resources**: Watch both client and server resources during tests
3. **Test Incrementally**: Test different components separately before full integration
4. **Document Results**: Save test results for comparison and analysis
5. **Test Regularly**: Run stress tests as part of your development workflow
6. **Simulate Real Usage**: Use test patterns that match your actual usage

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Stress Test
on: [push, pull_request]
jobs:
  stress-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: pip install requests
      - name: Run stress tests
        run: |
          cd tests
          ./run_stress_tests.sh -u http://test-server:9527 -t quick
```

### Jenkins Pipeline Example
```groovy
pipeline {
    agent any
    stages {
        stage('Stress Test') {
            steps {
                sh 'cd tests && ./run_stress_tests.sh -u http://test-server:9527 -t all'
            }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'stress_test_results/**', fingerprint: true
        }
    }
}
```

## Support

For issues or questions about the stress testing tools:
1. Check the error messages and logs
2. Review the troubleshooting section
3. Verify your test configuration
4. Check server-side logs and resources
5. Consider reducing load and testing incrementally
