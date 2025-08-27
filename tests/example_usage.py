#!/usr/bin/env python3

"""
Example usage of StreamReceiver
This script demonstrates practical usage patterns for the StreamReceiver class.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from stream_receiver import StreamReceiver

class DataAnalyzer:
    """Example class showing how to use StreamReceiver for data analysis"""
    
    def __init__(self, stream_addr='127.0.0.1', stream_port=9798):
        self.receiver = StreamReceiver(
            stream_addr=stream_addr,
            stream_port=stream_port,
            buffer_length=1200
        )
        self.running = False
        
    def start_monitoring(self):
        """Start monitoring the data stream"""
        print("Starting data monitoring...")
        self.receiver.start()
        self.running = True
        
        try:
            while self.running:
                # Get current status
                status = self.receiver.get_buffer_status()
                
                if status['buffer_index'] > 0:
                    # Get the latest frame
                    latest_frame = self.receiver.get_latest_data(n_frames=1)
                    
                    if latest_frame.size > 0:
                        # Analyze the data
                        self.analyze_frame(latest_frame[0])
                
                time.sleep(0.1)  # Check every 100ms
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        finally:
            self.stop_monitoring()
    
    def analyze_frame(self, frame_data):
        """Analyze a single frame of data"""
        # Calculate basic statistics
        mean_val = np.mean(frame_data)
        std_val = np.std(frame_data)
        min_val = np.min(frame_data)
        max_val = np.max(frame_data)
        
        # Print statistics (you could log these or send to monitoring system)
        print(f"Frame Stats: mean={mean_val:.6f}, std={std_val:.6f}, "
              f"min={min_val:.6f}, max={max_val:.6f}")
    
    def get_historical_data(self, minutes=5):
        """Get historical data for analysis"""
        # Assuming 0.5s per frame, calculate how many frames to get
        frames_needed = int(minutes * 60 * 2)  # 2 frames per second
        
        if frames_needed > 1200:
            frames_needed = 1200  # Limit to buffer size
        
        return self.receiver.get_latest_data(n_frames=frames_needed)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        self.receiver.stop()
        print("Data monitoring stopped.")

def real_time_plotting_example():
    """Example of real-time plotting with the receiver"""
    
    # Create receiver
    receiver = StreamReceiver(buffer_length=100)  # Smaller buffer for plotting
    
    try:
        receiver.start()
        
        # Setup matplotlib for real-time plotting
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 6))
        
        print("Starting real-time plotting... Press Ctrl+C to stop")
        
        while True:
            # Get latest data
            if receiver.get_buffer_status()['buffer_index'] > 0:
                latest_data = receiver.get_latest_data(n_frames=1)
                
                if latest_data.size > 0:
                    # Clear previous plot
                    ax.clear()
                    
                    # Plot the latest frame
                    ax.plot(latest_data[0])
                    ax.set_title(f"Real-time Spectrum (Frame {receiver.get_buffer_status()['buffer_index']})")
                    ax.set_xlabel("Frequency Channel")
                    ax.set_ylabel("Power")
                    ax.grid(True)
                    
                    # Update display
                    plt.draw()
                    plt.pause(0.01)
            
            time.sleep(0.5)  # Update every 0.5 seconds
            
    except KeyboardInterrupt:
        print("\nPlotting stopped.")
    finally:
        receiver.stop()
        plt.ioff()
        plt.close()

def data_logging_example():
    """Example of logging data to files"""
    
    receiver = StreamReceiver()
    
    try:
        receiver.start()
        
        # Open log file
        with open('spectrum_data.log', 'w') as log_file:
            log_file.write("timestamp,frame_index,mean_power,std_power,min_power,max_power\n")
            
            print("Starting data logging... Press Ctrl+C to stop")
            
            while True:
                if receiver.get_buffer_status()['buffer_index'] > 0:
                    latest_data = receiver.get_latest_data(n_frames=1)
                    
                    if latest_data.size > 0:
                        frame = latest_data[0]
                        
                        # Calculate statistics
                        mean_power = np.mean(frame)
                        std_power = np.std(frame)
                        min_power = np.min(frame)
                        max_power = np.max(frame)
                        
                        # Log to file
                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                        frame_idx = receiver.get_buffer_status()['buffer_index']
                        
                        log_file.write(f"{timestamp},{frame_idx},{mean_power:.6f},"
                                     f"{std_power:.6f},{min_power:.6f},{max_power:.6f}\n")
                        log_file.flush()  # Ensure data is written immediately
                        
                        print(f"Logged frame {frame_idx}: mean={mean_power:.6f}")
                
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        print("\nLogging stopped.")
    finally:
        receiver.stop()
        print("Data logging completed. Check 'spectrum_data.log' for results.")

def main():
    """Main function demonstrating different usage patterns"""
    
    print("StreamReceiver Usage Examples")
    print("=" * 40)
    print("1. Basic monitoring")
    print("2. Real-time plotting")
    print("3. Data logging")
    print("4. Exit")
    
    while True:
        choice = input("\nSelect an example (1-4): ").strip()
        
        if choice == '1':
            print("\nStarting basic monitoring...")
            analyzer = DataAnalyzer()
            analyzer.start_monitoring()
            
        elif choice == '2':
            print("\nStarting real-time plotting...")
            real_time_plotting_example()
            
        elif choice == '3':
            print("\nStarting data logging...")
            data_logging_example()
            
        elif choice == '4':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()

