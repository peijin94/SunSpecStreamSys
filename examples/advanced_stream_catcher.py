#!/usr/bin/env python3

"""
Advanced example to catch and process the ZMQ stream from dr_beam.py
This script demonstrates more sophisticated data processing including:
- Ring buffer management
- Data averaging and downsampling
- Real-time statistics
- Plotting capabilities
"""

import zmq
import json
import numpy as np
import time
import sys
import os
from collections import deque

# Optional matplotlib import for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available, plotting disabled")

class AdvancedStreamCatcher:
    """Advanced stream catcher with ring buffer and processing capabilities"""
    
    def __init__(self, stream_addr='127.0.0.1', stream_port=9798, buffer_length=100):
        self.stream_addr = stream_addr
        self.stream_port = stream_port
        self.buffer_length = buffer_length
        
        # Data processing parameters
        self.N_freq = 3072
        self.N_pol = 4
        self.pol_index = 0  # Use pol=0
        
        # Ring buffer: buffer_length x 768 (N_freq/4)
        self.ring_buffer = np.zeros((buffer_length, 768), dtype=np.float32)
        self.buffer_index = 0
        
        # Statistics tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_stats_time = time.time()
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{self.stream_addr}:{self.stream_port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        
        print(f"AdvancedStreamCatcher initialized: {stream_addr}:{stream_port}")
        print(f"Buffer length: {buffer_length}, Plotting: {PLOTTING_AVAILABLE}")
    
    def process_frame(self, data, header):
        """Process one frame of data"""
        try:
            # Convert bytes back to numpy array
            frame_data = np.frombuffer(data, dtype=np.float32)
            expected_size = header['nbeam'] * header['nchan'] * header['npol']
            
            if frame_data.size != expected_size:
                print(f"Warning: Unexpected data size {frame_data.size}, expected {expected_size}")
                return
            
            # Reshape to (nbeam, nchan, npol)
            frame_data = frame_data.reshape(header['nbeam'], header['nchan'], header['npol'])
            
            # Extract pol=0 and average down to N_freq/4 = 768
            pol_data = frame_data[0, :, self.pol_index]  # Shape: (3072,)
            
            # Average down by factor of 4
            pol_data_reshaped = pol_data.reshape(768, 4)
            averaged_data = np.mean(pol_data_reshaped, axis=1)  # Shape: (768,)
            
            # Update ring buffer
            self.ring_buffer[self.buffer_index, :] = averaged_data
            self.buffer_index = (self.buffer_index + 1) % self.buffer_length
            
            self.frame_count += 1
            
            # Print periodic statistics
            current_time = time.time()
            if current_time - self.last_stats_time >= 5.0:  # Every 5 seconds
                self.print_statistics(current_time)
                self.last_stats_time = current_time
            
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    def print_statistics(self, current_time):
        """Print current statistics"""
        elapsed = current_time - self.start_time
        rate = self.frame_count / elapsed if elapsed > 0 else 0
        
        # Get latest data for analysis
        if self.buffer_index > 0:
            latest_data = self.get_latest_data(n_frames=1)
            if latest_data.size > 0:
                data_stats = {
                    'min': latest_data.min(),
                    'max': latest_data.max(),
                    'mean': latest_data.mean(),
                    'std': latest_data.std()
                }
            else:
                data_stats = {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
        else:
            data_stats = {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
        
        print(f"\n=== Statistics (Frame {self.frame_count}) ===")
        print(f"Elapsed time: {elapsed:.1f}s")
        print(f"Frame rate: {rate:.2f} frames/second")
        print(f"Buffer index: {self.buffer_index}/{self.buffer_length}")
        print(f"Latest data: min={data_stats['min']:.6f}, max={data_stats['max']:.6f}, "
              f"mean={data_stats['mean']:.6f}, std={data_stats['std']:.6f}")
        print("=" * 40)
    
    def get_latest_data(self, n_frames=1):
        """Get the latest n_frames from the ring buffer"""
        if n_frames > self.buffer_length:
            n_frames = self.buffer_length
        
        if self.buffer_index == 0:
            return np.array([])
        
        # Calculate indices for the most recent frames
        indices = []
        for i in range(n_frames):
            idx = (self.buffer_index - 1 - i) % self.buffer_length
            indices.append(idx)
        
        # Return data in chronological order (oldest first)
        return self.ring_buffer[indices[::-1]]
    
    def create_plot(self, save_dir='./plots'):
        """Create and save a plot of the current data"""
        if not PLOTTING_AVAILABLE:
            print("Plotting not available (matplotlib not installed)")
            return
        
        if self.buffer_index == 0:
            print("No data to plot")
            return
        
        try:
            # Create plot directory
            os.makedirs(save_dir, exist_ok=True)
            
            # Get recent data
            n_frames = min(self.buffer_index, 20)  # Last 20 frames
            recent_data = self.get_latest_data(n_frames=n_frames)
            
            if recent_data.size == 0:
                return
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Latest spectrum
            latest_spectrum = recent_data[-1, :]
            freq_channels = np.arange(768)
            ax1.plot(freq_channels, latest_spectrum, 'b-', linewidth=0.8)
            ax1.set_title(f'Latest Spectrum - Frame {self.frame_count}')
            ax1.set_xlabel('Frequency Channel')
            ax1.set_ylabel('Power')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Waterfall plot
            if recent_data.shape[0] > 1:
                time_axis = np.arange(n_frames)
                freq_axis = np.arange(768)
                
                im = ax2.pcolormesh(time_axis, freq_axis, recent_data.T, 
                                   shading='auto', cmap='viridis')
                ax2.set_title(f'Waterfall Plot - Last {n_frames} frames')
                ax2.set_xlabel('Time (frames ago)')
                ax2.set_ylabel('Frequency Channel')
                
                # Add colorbar
                plt.colorbar(im, ax=ax2, label='Power')
            
            plt.tight_layout()
            
            # Save plot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"stream_analysis_{timestamp}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Plot saved: {filename}")
            
        except Exception as e:
            print(f"Error creating plot: {e}")
    
    def run(self, duration=None, plot_interval=30):
        """Run the stream catcher"""
        print(f"Starting stream catcher...")
        print(f"Duration: {'unlimited' if duration is None else f'{duration}s'}")
        print(f"Plot interval: {plot_interval}s")
        print("Press Ctrl+C to stop\n")
        
        last_plot_time = time.time()
        start_time = time.time()
        
        try:
            while True:
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    print(f"\nDuration limit reached ({duration}s)")
                    break
                
                try:
                    # Receive message
                    message = self.socket.recv_multipart()
                    
                    if len(message) >= 3:
                        topic = message[0]
                        header_data = message[1]
                        data = message[2]
                        
                        # Parse header
                        header = json.loads(header_data.decode())
                        
                        # Process data
                        if topic == b"data":
                            self.process_frame(data, header)
                            
                            # Create plot if interval has passed
                            current_time = time.time()
                            if (current_time - last_plot_time) >= plot_interval:
                                self.create_plot()
                                last_plot_time = current_time
                        else:
                            print(f"Unknown topic: {topic}")
                            
                except zmq.Again:
                    # Timeout - no message received
                    print(".", end="", flush=True)
                    time.sleep(0.1)
                    continue
                    
        except KeyboardInterrupt:
            print(f"\n\nStopped by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.socket.close()
        self.context.term()
        print("ZMQ connection closed")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced ZMQ Stream Catcher")
    parser.add_argument('--addr', type=str, default='127.0.0.1',
                       help='Stream address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=9798,
                       help='Stream port (default: 9798)')
    parser.add_argument('--buffer-length', type=int, default=100,
                       help='Ring buffer length (default: 100)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration to run in seconds (default: unlimited)')
    parser.add_argument('--plot-interval', type=int, default=30,
                       help='Plot creation interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Create and run stream catcher
    catcher = AdvancedStreamCatcher(
        stream_addr=args.addr,
        stream_port=args.port,
        buffer_length=args.buffer_length
    )
    
    catcher.run(duration=args.duration, plot_interval=args.plot_interval)

if __name__ == "__main__":
    main()


