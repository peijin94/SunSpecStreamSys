#!/usr/bin/env python3

import zmq
import json
import numpy as np
import time
import threading
import logging
import gc
from collections import deque

class StreamReceiver:
    """
    Receiver for streaming data from AvgStreamingOp.
    Connects to ZMQ stream, processes data, and maintains a ring buffer.
    """
    
    def __init__(self, stream_addr='127.0.0.1', stream_port=9798, buffer_length=1200, gc_interval=100):
        self.stream_addr = stream_addr
        self.stream_port = stream_port
        self.buffer_length = buffer_length
        self.gc_interval = gc_interval  # Garbage collection interval
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{self.stream_addr}:{self.stream_port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        
        # Ring buffer: 1200 x 1536 (N_freq/2)
        self.ring_buffer = np.zeros((buffer_length, 1536), dtype=np.float32)
        self.buffer_index = 0
        
        # Data processing parameters
        self.N_freq = 3072
        self.N_pol = 4
        self.pol_index = 0  # Use pol=0
        
        # Control flags
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Setup logging
        self.setup_logging()
        
        self.log.info(f"StreamReceiver initialized: {stream_addr}:{stream_port}, buffer: {buffer_length}x1536")
    
    def setup_logging(self):
        """Setup logging configuration"""
        self.log = logging.getLogger(__name__)
        if not self.log.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                                        datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            self.log.addHandler(handler)
            self.log.setLevel(logging.INFO)
    
    def process_frame(self, data, header):
        """
        Process one frame of data.
        
        Parameters:
        -----------
        data : bytes
            Raw data bytes
        header : dict
            Header information including time_tag
        """
        try:
            # Convert bytes back to numpy array
            # Expected shape: (1, N_freq, N_pol) = (1, 3072, 4)
            frame_data = np.frombuffer(data, dtype=np.float32)
            expected_size = 1 * self.N_freq * self.N_pol
            
            if frame_data.size != expected_size:
                self.log.warning(f"Unexpected data size: {frame_data.size}, expected: {expected_size}")
                return
            
            # Reshape to (1, 3072, 4)
            frame_data = frame_data.reshape(1, self.N_freq, self.N_pol)
            
            # Extract pol=0 and average down to N_freq/2 = 1536
            pol_data = frame_data[0, :, self.pol_index]  # Shape: (3072,)
            
            # Average down by factor of 2
            # Reshape to (1536, 2) and take mean along axis 1
            pol_data_reshaped = pol_data.reshape(1536, 2)
            averaged_data = np.mean(pol_data_reshaped, axis=1)  # Shape: (1536,)
            
            # Update ring buffer
            self.ring_buffer[self.buffer_index, :] = averaged_data
            
            # Update buffer index (circular)
            self.buffer_index = (self.buffer_index + 1) % self.buffer_length
            
            self.log.debug(f"Processed frame: time_tag={header.get('time_tag', 'N/A')}, "
                          f"buffer_index={self.buffer_index}, data_shape={averaged_data.shape}")
            
            # Clean up temporary objects to reduce memory overhead
            del frame_data, pol_data, pol_data_reshaped, averaged_data
            
        except Exception as e:
            self.log.error(f"Error processing frame: {str(e)}")
    
    def receive_loop(self):
        """Main receive loop"""
        self.log.info("Starting receive loop...")
        
        # Garbage collection counters
        gc_counter = 0
        
        while not self.shutdown_event.is_set():
            try:
                # Receive multipart message: [topic, header, data]
                message = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                
                if len(message) >= 3:
                    topic = message[0]
                    header_data = message[1]
                    data = message[2]
                    
                    # Parse header
                    header = json.loads(header_data.decode())
                    
                    # Process data based on topic
                    if topic == b"data":
                        # This is the actual data
                        self.process_frame(data, header)
                        
                        # Increment counter and run garbage collection periodically
                        gc_counter += 1
                        if gc_counter >= self.gc_interval:
                            gc.collect()
                            gc_counter = 0
                            self.log.debug("Garbage collection performed")
                    else:
                        self.log.warning(f"Unknown topic: {topic}")
                
            except zmq.Again:
                # No message available (non-blocking)
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            except Exception as e:
                self.log.error(f"Error in receive loop: {str(e)}")
                time.sleep(0.1)  # Wait a bit before retrying
        
        # Final garbage collection before stopping
        gc.collect()
        self.log.info("Receive loop stopped")
    
    def start(self):
        """Start the receiver"""
        if self.running:
            self.log.warning("Receiver is already running")
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start receive thread
        self.receive_thread = threading.Thread(target=self.receive_loop, daemon=True)
        self.receive_thread.start()
        
        self.log.info("StreamReceiver started")
    
    def stop(self):
        """Stop the receiver"""
        if not self.running:
            self.log.warning("Receiver is not running")
            return
        
        self.log.info("Stopping StreamReceiver...")
        self.running = False
        self.shutdown_event.set()
        
        # Wait for thread to finish
        if hasattr(self, 'receive_thread'):
            self.receive_thread.join(timeout=5.0)
        
        # Close ZMQ resources
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.term()
        
        # Final garbage collection
        gc.collect()
        
        self.log.info("StreamReceiver stopped")
    
    def get_buffer_status(self):
        """Get current buffer status"""
        return {
            'buffer_index': self.buffer_index,
            'buffer_shape': self.ring_buffer.shape,
            'is_running': self.running
        }
    
    def force_garbage_collection(self):
        """Manually trigger garbage collection"""
        collected = gc.collect()
        self.log.info(f"Manual garbage collection: collected {collected} objects")
        return collected
    
    def get_memory_info(self):
        """Get memory usage information"""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'gc_objects': len(gc.get_objects()),
            'gc_garbage': len(gc.garbage)
        }
    
    def optimize_memory(self):
        """Optimize memory usage by clearing old data and running GC"""
        # Clear old data from ring buffer if memory usage is high
        memory_info = self.get_memory_info()
        
        if memory_info['rss_mb'] > 100:  # If using more than 100MB
            # Clear half of the ring buffer
            clear_start = (self.buffer_index + self.buffer_length // 2) % self.buffer_length
            clear_end = self.buffer_index
            
            if clear_start < clear_end:
                self.ring_buffer[clear_start:clear_end, :] = 0
            else:
                self.ring_buffer[clear_start:, :] = 0
                self.ring_buffer[:clear_end, :] = 0
            
            self.log.info(f"Cleared ring buffer to reduce memory usage (RSS: {memory_info['rss_mb']:.1f}MB)")
        
        # Force garbage collection
        collected = self.force_garbage_collection()
        
        return {
            'memory_before': memory_info,
            'memory_after': self.get_memory_info(),
            'objects_collected': collected
        }
    
    def get_latest_data(self, n_frames=1):
        """
        Get the latest n_frames from the ring buffer
        
        Parameters:
        -----------
        n_frames : int
            Number of most recent frames to return
            
        Returns:
        --------
        numpy.ndarray
            Latest data with shape (n_frames, 1536)
        """
        if n_frames > self.buffer_length:
            n_frames = self.buffer_length
        
        # Calculate indices for the most recent frames
        indices = []
        for i in range(n_frames):
            idx = (self.buffer_index - 1 - i) % self.buffer_length
            indices.append(idx)
        
        # Return data in chronological order (oldest first)
        return self.ring_buffer[indices[::-1]]


def main():
    """Main function for standalone testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stream Receiver for AvgStreamingOp")
    parser.add_argument('--addr', type=str, default='127.0.0.1',
                       help='Stream address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=9798,
                       help='Stream port (default: 9798)')
    parser.add_argument('--buffer-length', type=int, default=1200,
                       help='Ring buffer length (default: 1200)')
    parser.add_argument('--gc-interval', type=int, default=100,
                       help='Garbage collection interval in frames (default: 100)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create and start receiver
    receiver = StreamReceiver(
        stream_addr=args.addr,
        stream_port=args.port,
        buffer_length=args.buffer_length,
        gc_interval=args.gc_interval
    )
    
    try:
        receiver.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            # Print status every 10 seconds
            if int(time.time()) % 10 == 0:
                status = receiver.get_buffer_status()
                memory_info = receiver.get_memory_info()
                print(f"Status: {status}")
                print(f"Memory: RSS={memory_info['rss_mb']:.1f}MB, "
                      f"Objects={memory_info['gc_objects']}, "
                      f"Garbage={memory_info['gc_garbage']}")
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
