#!/usr/bin/env python3

import zmq
import json
import numpy as np
import time
import threading
import logging
import gc
import os
from collections import deque
from datetime import datetime, timezone, timedelta
from astropy.time import Time

from util import paint_arr_to_jpg

# Configure matplotlib for headless server (no display required)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers

# Set environment variables for headless operation
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Web server imports
try:
    from flask import Flask, render_template, jsonify
    from flask_cors import CORS
    from waitress import serve
    import threading
    FLASK_AVAILABLE = True
    WAITRESS_AVAILABLE = True
except ImportError as e:
    FLASK_AVAILABLE = False
    WAITRESS_AVAILABLE = False
    print(f"Warning: Web server dependencies not available ({e}). Web interface will be disabled.")

def lwa_time_tag_to_datetime(time_tag: int, rate: float = 196_000_000) -> datetime:
    """
    Convert an LWA timetag to a UTC datetime.
    
    Parameters
    ----------
    time_tag : int
        The raw timetag value from LWA data.
    rate : float, optional
        Tick rate in Hz (default = 196,000,000). 
        For exact hardware clock, use 196_608_000.
    
    Returns
    -------
    datetime.datetime (UTC)
    """
    # Separate integer seconds and fractional ticks
    secs, rem = divmod(time_tag, rate)
    
    # Compute fractional seconds from remaining ticks
    frac = rem / rate
    
    # Convert to datetime (epoch = 1970-01-01 UTC)
    return datetime(1970, 1, 1) + timedelta(seconds=secs + frac)

class StreamReceiver:
    """
    Receiver for streaming data from AvgStreamingOp.
    Connects to ZMQ stream, processes data, and maintains a ring buffer.
    """
    
    def __init__(self, stream_addr='127.0.0.1', stream_port=9798, buffer_length=600, gc_interval=100, 
                 start_webshow=False, streaming_interval=0.5, verbose=False, plot_dir='./figs/', save_dir='./stream_spec_npz/', save_all_to_file=False):
        self.stream_addr = stream_addr
        self.stream_port = stream_port
        self.buffer_length = buffer_length
        self.gc_interval = gc_interval  # Garbage collection interval
        self.start_webshow = start_webshow # Flag to start web server
        self.streaming_interval = streaming_interval  # Streaming interval in seconds per frame
        self.verbose = verbose

        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{self.stream_addr}:{self.stream_port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        
        # Ring buffer: 600 x 768 (N_freq/4)
        self.ring_buffer = np.zeros((self.buffer_length, 768), dtype=np.float32)
        self.ring_buffer_v = np.zeros((self.buffer_length, 768), dtype=np.float32)
        self.buffer_index = 0
        self.ring_arr_mjd = np.zeros((self.buffer_length, 1), dtype=np.float64)
        
        self.save_all_to_file = save_all_to_file
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Data processing parameters
        self.N_freq = 3072
        self.N_pol = 4
        
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Control flags
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Delay tracking
        self.delay = 0.0  # Current delay in seconds
        self.delay_method = 'auto' # Default to 'auto'
        
        # Type 3 detection (radio burst detection)
        self.detection_interval = 5.0  # Detection interval in seconds
        self.last_detection_time = 0
        self.latest_detections = []  # Store latest detection results
        self.detection_lock = threading.Lock()  # Thread safety for detections
        
        # Setup logging
        self.setup_logging()
        
        self.log.info(f"StreamReceiver initialized: {stream_addr}:{stream_port}")
        if self.start_webshow:
            self.log.info("Web server enabled")
    
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
    

    def save_latest_data_to_jpg(self, norm_factor=1e4*24, out_size=(640,640)):
        """
        Save the latest data to a jpg image.
        Parameters
        ----------
        norm_factor : float, optional
            Normalization factor for the data. Default is 1e4*24.
        out_size: tuple, optional
            Output size for the image. Default is (320,320).
        Returns
        -------
        str: Path to the saved image
        """

        latest_data = self.get_latest_data(n_frames=self.buffer_length)[0]
        # latest_data = np.flip(latest_data, axis=1)
        # use scikit-image to resize the image
        
        from skimage.transform import resize
        data_out = resize(latest_data, out_size, anti_aliasing=True)

        paint_arr_to_jpg(data_out/norm_factor, filename=f'{self.plot_dir}/latest_data.jpg', vmax=200, vmin=0.5, scaling='log')
        #self.log.info(f"Latest data saved to {self.plot_dir}/latest_data.jpg")

        return f'{self.plot_dir}/latest_data.jpg'


    def save_latest_data_to_npz(self, norm_factor=1e4*24, save_dir='./figs/',
        filename='latest_data.npz', save_when_sun_up=True): # norm factor to convert to sfu
        """Save the latest data to a npz file"""


        import check_sun_elevation
        if save_when_sun_up:
            if not check_sun_elevation.is_sun_up():
                self.log.info("Sun is down, not saving data")
                return

        latest_data, mjd_data = self.get_latest_data(n_frames=self.buffer_length, pol='I')
        latest_data_v = self.get_latest_data(n_frames=self.buffer_length, pol='V')[0]

        # create 3-dim array with shape (*latest_data.shape, 2)
        data_out = np.zeros((*latest_data.shape, 2), dtype=np.float32)
        data_out[..., 0] = latest_data
        data_out[..., 1] = latest_data_v

        freq_lower = 196*(600/8192)
        freq_upper = 196*((600+3072-1)/8192)
        freq_array = np.linspace(freq_lower, freq_upper, data_out.shape[1])

        np.savez(f'{save_dir}/{filename}', data=data_out/norm_factor, mjd=self.ring_arr_mjd, freq=freq_array)
        self.log.info(f"Latest data saved to {save_dir}/{filename}")
        return f'{save_dir}/{filename}'


    def run_type3_detection(self):
        """Run Type 3 radio burst detection using YOLO v8s model"""
        current_time = time.time()
        
        # Check if enough time has passed since last detection
        if current_time - self.last_detection_time < self.detection_interval:
            return
        
        
        try:
            # Import YOLO here to avoid import errors if not installed
            from ultralytics import YOLO
            
            
            # Load YOLO model
            model_path = 'model/best.v11.pt'
            if not os.path.exists(model_path):
                self.log.error(f"YOLO model not found at {model_path}")
                return
            
            model = YOLO(model_path)
            
            # Save latest data as image for detection
            image_path = self.save_latest_data_to_jpg()
            
            # Check if image file exists
            if not os.path.exists(image_path):
                self.log.error(f"Latest data image not found at {image_path}")
                return
            
            # Run YOLO prediction on the image
            results = model.predict(image_path, conf=0.8, iou=0.6, verbose=False)
            
            # Process detection results
            detections = []
            if results and len(results) > 0:

                self.last_detection_time = current_time
                result = results[0]  # Get first (and only) result
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Get bounding box coordinates (xyxy format)
                        box = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = box
                        
                        # Convert to normalized coordinates (x, y, width, height)
                        img_height, img_width = 640, 640  # Image size from save_latest_data_to_jpg

                        # flip y coordinate
                        y1 = img_height - y1
                        y2 = img_height - y2
                        
                        x = float(x1 / img_width)
                        y = float(y1 / img_height)
                        
                        # flip y coordinate
                        #y = 1 - y
                        
                        width = float((x2 - x1) / img_width)
                        height = float((y2 - y1) / img_height)
                        
                        # Get class and confidence
                        class_id = int(boxes.cls[i].cpu().numpy())
                        confidence = float(boxes.conf[i].cpu().numpy())
                        
                        # Map class ID to class name (assuming 0=type3, 1=type3b)
                        class_name = 'type3' if class_id == 0 else 'type3b'
                        
                        detection = {
                            'id': i,
                            'class_id': class_id,
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x, y, width, height],  # [x, y, width, height] normalized (all Python floats)
                            'timestamp': current_time
                        }
                        detections.append(detection)
            else:
                self.log.info(f"No detections found in {image_path}")
            
            # Update latest detections with thread safety
            with self.detection_lock:
                self.latest_detections = detections
            
            if self.verbose and len(detections) > 0:
                self.log.info(f"Type 3 detection: {len(detections)} bursts detected")
                for det in detections:
                    self.log.info(f"  - {det['class']}: {det['confidence']:.3f} at {det['bbox']}")
            
        except ImportError as e:
            self.log.error(f"YOLO dependencies not available: {e}")
            self.log.info("Please install ultralytics: pip install ultralytics")
        except Exception as e:
            self.log.error(f"Error in Type 3 detection: {str(e)}")
            import traceback
            self.log.error(f"Traceback: {traceback.format_exc()}")


    def run_type3_detection_dummy(self):
        """Run Type 3 radio burst detection (dummy implementation)"""
        current_time = time.time()
        
        # Check if enough time has passed since last detection
        if current_time - self.last_detection_time < self.detection_interval:
            return
        
        self.last_detection_time = current_time

        latest_data_jpg = self.save_latest_data_to_jpg()
        #latest_data_npz = self.save_latest_data_to_npz()

        # For now, create random bounding boxes
        num_detections = np.random.randint(0, 8)  # 0-3 detections
        detections = []
        
        for i in range(num_detections):
            # Generate random bounding box coordinates
            # x, y, width, height (normalized to 0-1)
            x = np.random.uniform(0.1, 0.8)
            y = np.random.uniform(0.1, 0.8)
            width = np.random.uniform(0.02, 0.12)
            height = np.random.uniform(0.25, 0.6)
            
            # Ensure box stays within bounds
            if x + width > 1.0:
                width = 1.0 - x
            if y + height > 1.0:
                height = 1.0 - y
            
            # Randomly assign class (0 = type3, 1 = type3b)
            class_id = np.random.randint(0, 2)
            class_name = 'type3' if class_id == 0 else 'type3b'
            
            detection = {
                'id': i,
                'class_id': class_id,
                'class': class_name,
                'confidence': np.random.uniform(0.6, 0.95),
                'bbox': [x, y, width, height],  # [x, y, width, height]
                'timestamp': current_time}
            detections.append(detection)
        
        # Update latest detections with thread safety
        with self.detection_lock:
            self.latest_detections = detections
        
        if self.verbose:
            self.log.info(f"Type 3 detection: {len(detections)} bursts detected")
 
    
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
            # Debug header information (only for first frame to avoid spam)
            if self.buffer_index < 1 and self.verbose:  # Only debug first frame if verbose
                self.debug_header_info(header)
            # Convert bytes back to numpy array
            # Expected shape: (1, N_freq, N_pol) = (1, 3072, 4)
            frame_data = np.frombuffer(data, dtype=np.float32)
            expected_size = 1 * self.N_freq * self.N_pol
            
            if frame_data.size != expected_size:
                self.log.warning(f"Unexpected data size: {frame_data.size}, expected: {expected_size}")
                return
            
            # Reshape to (1, 3072, 4)  XX,YY, XY_real, XY_imag
            frame_data = frame_data.reshape(1, self.N_freq, self.N_pol)
            
            # Extract pol=0 and average down to N_freq/4 = 768
            pol_data   = (frame_data[0, :, 0] + frame_data[0, :, 1]) / 2  # Shape: (3072,)
            pol_data_v = (frame_data[0, :, 3])  # Shape: (3072,)

            # Average down by factor of 4
            # Reshape to (768, 4) and take mean along axis 1
            pol_data_reshaped = pol_data.reshape(768, 4)
            pol_data_v_reshaped = pol_data_v.reshape(768, 4)

            averaged_data = np.mean(pol_data_reshaped, axis=1)  # Shape: (768,)
            averaged_data_v = np.mean(pol_data_v_reshaped, axis=1)  # Shape: (768,)

            # Update ring buffer
            self.ring_buffer[self.buffer_index, :] = averaged_data
            self.ring_buffer_v[self.buffer_index, :] = averaged_data_v

            # Update buffer index (circular)
            self.buffer_index = (self.buffer_index + 1) % self.buffer_length

            # update ring_arr_mjd
            string_time = header["last_block_time"]
            self.ring_arr_mjd[self.buffer_index, 0] = Time(string_time, format='iso', scale='utc').mjd
            
            # Calculate delay between data timestamp and current time
            if self.delay_method == 'manual':
                # Use manually set delay, don't recalculate
                pass
            elif self.delay_method == 'auto':
                current_time_utc = Time.now().unix
                
                #print("last_block_time", header["last_block_time"], "timestamp", header["timestamp"])
                # Use header["timestamp"] for delay calculation (new format from dr_beam.py)
                if "timestamp" in header:
                    timestamp_val = header["timestamp"]
                    self.log.debug(f"Using header timestamp: {timestamp_val}")
                    
                    try:
                        # Parse timestamp string (format: "1755552894.787341")
                        header_time = float(timestamp_val)
                        self.delay = current_time_utc - header_time
                        self.log.debug(f"Calculated delay: {self.delay:.6f}s")
                        
                        # Sanity check for unreasonable delays
                        if abs(self.delay) > 3600 or self.delay < 0:
                            self.log.warning(f"Unreasonable delay calculated: {self.delay:.3f}s, timestamp: {timestamp_val}, header_time: {header_time}, current_utc: {current_time_utc:.6f}")
                            self.delay = 0.0
                    except (ValueError, TypeError) as e:
                        self.log.warning(f"Could not parse timestamp: {timestamp_val}, error: {e}")
                        self.delay = 0.0
                else:
                    self.log.warning("No 'timestamp' found in header, using fallback delay calculation")
                    self.delay = 0.0
                
                # Fallback to buffer-based calculation if delay is unreasonable
                if abs(self.delay) > 3600 or self.delay < 0:
                    relative_delay = (self.buffer_length - self.buffer_index) * self.streaming_interval
                    self.log.info(f"Using fallback delay calculation: {relative_delay:.3f}s (buffer-based)")
                    self.delay = relative_delay
            
            if self.verbose:
                self.log.debug(f"Processed frame: buffer_index={self.buffer_index}, delay={self.delay:.3f}s")
            
            # Run Type 3 detection every 5 frames
            if (self.buffer_index + 3) % 5 == 0:
                self.run_type3_detection()

            # save all when buffer_index everytime when length-1
            if self.save_all_to_file and self.buffer_index == self.buffer_length - 1:
                
                time_now_iso_str = Time.now().iso[:19].replace(' ', '_').replace(':', '_')
                date_now_str = Time.now().iso[:10].replace(' ', '_').replace(':', '_')
                dir_data = os.path.join(self.save_dir, date_now_str)
                #mkdir
                os.makedirs(dir_data, exist_ok=True)
                self.save_latest_data_to_npz(save_dir=dir_data, filename=f'{time_now_iso_str}.npz')
            
            # Clean up temporary objects to reduce memory overhead
            del frame_data, pol_data, pol_data_reshaped, averaged_data
            
        except Exception as e:
            self.log.error(f"Error processing frame: {str(e)}")
    
    def receive_loop(self):
        """Main receive loop"""
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

                    else:
                        self.log.warning(f"Unknown topic: {topic}")

                    if self.verbose:    
                        self.log.info(f"Received message: {topic} first 10 numbers: {data[:10]}")
                
            except zmq.Again:
                # No message available (non-blocking)
                time.sleep(0.005)  # Small sleep to prevent busy waiting
                continue
            except Exception as e:
                self.log.error(f"Error in receive loop: {str(e)}")
                time.sleep(0.1)  # Wait a bit before retrying
        
        # Final garbage collection before stopping
        gc.collect()
    
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
        

        # Start web server only if enabled
        if self.start_webshow:
            self.start_webserver()
        
        self.log.info("StreamReceiver started")
    
    def stop(self):
        """Stop the receiver"""
        if not self.running:
            self.log.warning("Receiver is not running")
            return
        
        self.running = False
        self.shutdown_event.set()
        
        
        # Stop web server
        if self.start_webshow:
            self.stop_webserver()
        
        # Wait for receive thread to finish
        if hasattr(self, 'receive_thread'):
            self.receive_thread.join(timeout=5.0)
        
        # Close ZMQ resources
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.term()
        
        # Final garbage collection
        gc.collect()
    
    def get_current_delay(self):
        """Get the current delay between data time and current time"""
        return self.delay
    
    def set_delay(self, delay_seconds):
        """Manually set the delay value (useful for debugging)"""
        self.delay = delay_seconds
        self.log.info(f"Delay manually set to: {self.delay:.3f}s")
    
    def set_delay_calculation_method(self, method='auto'):
        """
        Set the delay calculation method
        
        Parameters:
        -----------
        method : str
            'auto' - Try to use time_tag, fallback to buffer-based
            'buffer' - Always use buffer-based calculation
            'manual' - Use manually set delay value
        """
        self.delay_method = method
        self.log.info(f"Delay calculation method set to: {method}")
    
    def debug_header_info(self, header):
        """Debug method to inspect header information"""
        self.log.info("=== Header Debug Information ===")
        if self.verbose:
            for key, value in header.items():
                self.log.info(f"  {key}: {value} (type: {type(value)})")
        
        current_time = time.time()
        
        # Check for timestamp field 
        if "timestamp" in header:
            timestamp_val = header["timestamp"]
            self.log.info(f"  Header timestamp: {timestamp_val} (type: {type(timestamp_val)})")
            
            try:
                # Parse timestamp string (format: "1755552894.787341")
                header_time = float(timestamp_val)
                self.log.info(f"  Parsed timestamp: {header_time:.6f}s")
                
                # Calculate delay
                delay = current_time - header_time
                self.log.info(f"  Calculated delay: {delay:.6f}s")
                
                # Check if delay is reasonable
                if abs(delay) < 86400:  # Less than 24 hours
                    self.log.info("  ✓ Delay is reasonable")
                else:
                    self.log.info("  ⚠️  Delay is large (this may be normal for your system)")
                    
            except (ValueError, TypeError) as e:
                self.log.info(f"  ✗ Could not parse timestamp: {e}")
        else:
            self.log.info("  No 'timestamp' field found in header")
        
        # Check for last_block_time field (new from dr_beam.py)
        if "last_block_time" in header:
            last_block_time = header["last_block_time"]
            self.log.info(f"  Header last_block_time: {last_block_time} (type: {type(last_block_time)})")
        else:
            self.log.info("  No 'last_block_time' field found in header")
        
        self.log.info("=== End Header Debug ===")
    
    def get_buffer_status(self):
        """Get current buffer status"""
        return {
            'buffer_index': self.buffer_index,
            'buffer_shape': self.ring_buffer.shape,
            'is_running': self.running,
            'current_delay': self.delay,
            'clock_offset': None, # Removed clock_offset
            'last_sync_time': None, # Removed last_sync_time
            'synchronized_time': Time.now().unix, # Use UTC time
            'webserver': self.get_webserver_status()
        }
    
    def force_garbage_collection(self):
        """Manually trigger garbage collection"""
        collected = gc.collect()
        self.log.info(f"Manual garbage collection: collected {collected} objects")
        return collected
    
    
    def get_latest_data(self, n_frames=1, pol='I'):
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

        if pol == 'I':
            return self.ring_buffer[indices[::-1]], self.ring_arr_mjd[indices[::-1]]
        elif pol == 'V':
            return self.ring_buffer_v[indices[::-1]], self.ring_arr_mjd[indices[::-1]]
        else:
            raise ValueError(f"Invalid polarization: {pol}")  
            return None, None

    def start_webserver(self):
        """Start the Waitress web server for live spectrum display"""
        if not FLASK_AVAILABLE or not WAITRESS_AVAILABLE:
            self.log.warning("Web server dependencies not available, cannot start web server.")
            return

        try:
            self.app = Flask(__name__)
            CORS(self.app)

            @self.app.route('/')
            def index():
                return render_template('index.html')

            @self.app.route('/data')
            def get_data():
                # Get the latest data from the ring buffer
                if self.buffer_index > 0:
                    latest_data = self.get_latest_data(n_frames=1)[0]
                    if latest_data.size > 0:
                        # Return the most recent frame as a list
                        data_json = latest_data[0].tolist()
                        return jsonify(data_json)
                # Return empty list if no data
                return jsonify([])

            @self.app.route('/type3detect')
            def get_type3_detections():
                """Get latest Type 3 radio burst detections"""
                with self.detection_lock:
                    detections = self.latest_detections.copy()
                
                # Add time anchor information
                current_time = time.time()
                response_data = {
                    'detections': detections,
                    'timestamp': current_time,
                    'time_anchor': current_time,
                    'count': len(detections),
                    'last_detection_time': self.last_detection_time
                }
                #print(response_data)
                return jsonify(response_data)

            @self.app.route('/refresh')
            def refresh_data(n_frames=600):
                """Get the entire buffer array for refresh functionality"""
                if self.buffer_index > 0:
                    # Get all available data from the ring buffer
                    latest_data = self.get_latest_data(n_frames=n_frames)[0]
                    if latest_data.size > 0:
                        # Return the entire buffer as a list of lists
                        data_json = latest_data.tolist()
                        return jsonify({
                            'data': data_json,
                            'buffer_length': self.buffer_length,
                            'buffer_index': self.buffer_index,
                            'timestamp': time.time()
                        })
                # Return empty data if no data available
                return jsonify({
                    'data': [],
                    'buffer_length': self.buffer_length,
                    'buffer_index': self.buffer_index,
                    'timestamp': time.time()
                })

            # Use fixed port 9527 for web server
            web_port = 9527
            self.log.info(f"Web server starting on http://localhost:{web_port}")
            
            # Start waitress server in a separate thread
            self.webserver_thread = threading.Thread(
                target=lambda: serve(self.app, host='localhost', port=web_port, threads=4),
                daemon=True
            )
            self.webserver_thread.start()
            
            # Wait a moment for server to start
            time.sleep(2)
            
        except Exception as e:
            self.log.error(f"Failed to start web server: {e}")
            
    def stop_webserver(self):
        """Stop the Waitress web server"""
        if hasattr(self, 'webserver_thread') and self.webserver_thread.is_alive():
            self.log.info("Stopping web server...")
            # Waitress doesn't have a built-in shutdown method, so we just wait for the thread to finish
            self.webserver_thread.join(timeout=5.0)
            if self.webserver_thread.is_alive():
                self.log.warning("Web server thread did not stop gracefully")

    def is_webserver_running(self):
        """Check if the web server is running and accessible"""
        if not self.start_webshow or not hasattr(self, 'webserver_thread'):
            return False
        
        try:
            import requests
            response = requests.get("http://localhost:9527/", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_webserver_status(self):
        """Get web server status information"""
        if not self.start_webshow:
            return {
                'enabled': False,
                'running': False,
                'url': None,
                'status': 'disabled'
            }
        
        running = self.is_webserver_running()
        return {
            'enabled': True,
            'running': running,
            'url': 'http://localhost:9527' if running else None,
            'status': 'running' if running else 'failed'
        }


def main():
    """Main function for standalone testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stream Receiver for AvgStreamingOp")
    parser.add_argument('--addr', type=str, default='127.0.0.1',
                       help='Stream address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=30002,
                       help='Stream port (default: 30002)')
    parser.add_argument('--buffer-length', type=int, default=600,
                       help='Ring buffer length (default: 600)')
    parser.add_argument('--gc-interval', type=int, default=100,
                       help='Garbage collection interval in frames (default: 100)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    parser.add_argument('--start-webshow', action='store_true',
                       help='Start a web server on localhost:9898 to display live spectrum data')
    parser.add_argument('--streaming-interval', type=float, default=0.5,
                       help='Streaming interval in seconds per frame (default: 0.5)')
    parser.add_argument('--save-all-to-file', action='store_true',
                       help='Save all data to files', default=False)
    parser.add_argument('--save-dir', type=str, default='/common/lwa/stream_spec_npz/',
                       help='Directory to save data files (default: stream_spec_npz/)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
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
        gc_interval=args.gc_interval,
        start_webshow=args.start_webshow,
        streaming_interval=args.streaming_interval,
        verbose=args.verbose,
        save_dir=args.save_dir,
        save_all_to_file=args.save_all_to_file
    )
    
    try:
        receiver.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            # Print status every 30 seconds (reduced from 10)
            if int(time.time()) % 30 == 0:
                status = receiver.get_buffer_status()
                print(f"Status: buffer={status['buffer_index']}, delay={status['current_delay']:.1f}s")
                
                # Show web server status if enabled and verbose
                if receiver.start_webshow and args.verbose:
                    web_status = receiver.get_webserver_status()
                    print(f"Web Server: {web_status['status']} - {web_status['url']}")
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
