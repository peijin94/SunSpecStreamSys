#!/usr/bin/env python3

import zmq
import json
import numpy as np
import time
import threading
import logging
import gc
import os
import sqlite3
import uuid
import struct
from spectrum_time import time_raster, source_box
from radio_images import RadioImageCache, FREQUENCIES as IMAGE_FREQUENCIES
from antenna_flags import AntennaFlagCache
from imaging_status import ImagingStatusCache
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

# Web server imports (FastAPI + ASGI server)
try:
    from fastapi import FastAPI, Request, HTTPException, Query
    from fastapi.responses import JSONResponse, FileResponse, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from starlette.middleware.gzip import GZipMiddleware
    import uvicorn
    import threading
    WEB_AVAILABLE = True
except ImportError as e:
    WEB_AVAILABLE = False
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
                 start_webshow=False, streaming_interval=0.5, verbose=False, plot_dir='./figs/', save_dir='./stream_spec_npz/', save_all_to_file=True):
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
        self.buffer_lock = threading.RLock()
        self.frames_received = 0
        self.boot_id = uuid.uuid4().hex
        
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
        self.last_detection_attempt = 0
        self.detection_confidence = 0.55
        self.detection_model = None
        self.detection_window = {}
        self.latest_detections = []  # Store latest detection results
        self.detection_lock = threading.Lock()  # Thread safety for detections

        # AI summary (Gemini) of latest spectrum
        self.ai_summary_interval = 60.0  # seconds
        self.last_ai_summary_time = 0.0
        self.last_ai_attempt = 0.0
        self.latest_ai_summary = ""
        self.ai_summary_lock = threading.Lock()
        self.radio_images = RadioImageCache(
            os.environ.get('LWA_IMAGE_API_URL', 'http://127.0.0.1:18080'))
        self.antenna_flags = AntennaFlagCache(os.environ.get(
            'LWA_QA_API_URL', os.environ.get('LWA_IMAGE_API_URL', 'http://127.0.0.1:18080')))
        self.imaging_status = ImagingStatusCache(self.antenna_flags.base_url)
        
        # Setup logging
        self.setup_logging()

        # Visitor logging database
        self._init_visitor_db()
        self.restore_restart_snapshot()
        
        self.log.info(f"StreamReceiver initialized: {stream_addr}:{stream_port}")
        if self.start_webshow:
            self.log.info("Web server enabled")
    
    def restore_restart_snapshot(self):
        """Consume an optional deployment handoff to preserve the live time window."""
        path = os.path.join(self.plot_dir, 'restart-snapshot.npz')
        if not os.path.exists(path):
            return
        try:
            with np.load(path, allow_pickle=False) as saved:
                intensity, circular, times = saved['I'], saved['V'], saved['timestamps']
                count = len(times)
                if (intensity.shape != (count, 768) or circular.shape != intensity.shape
                        or count < 1 or count > self.buffer_length or not np.isfinite(times).all()
                        or np.any(np.diff(times) < 0) or abs(time.time() - times[-1]) > 300):
                    raise ValueError('Invalid or stale restart snapshot')
                self.ring_buffer[:count] = intensity
                self.ring_buffer_v[:count] = circular
                self.ring_arr_mjd[:count, 0] = times / 86400.0 + 40587.0
                self.buffer_index = count % self.buffer_length
                self.frames_received = count
            os.replace(path, os.path.join(self.plot_dir, 'restart-snapshot.used.npz'))
            self.log.info(f'Restored {count} timestamped source frames for deployment')
        except Exception as exc:
            self.log.warning(f'Restart snapshot skipped: {exc}')

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
    

    def save_latest_data_to_jpg(self, norm_factor=1e4*24, out_size=(640, 640),
                                snapshot=None, filename='latest_data.jpg'):
        """Render one immutable source-time snapshot in the model's orientation."""
        from skimage.transform import resize
        metadata, intensity, _, timestamps = snapshot or self._get_spectrum_arrays()
        raster, _, _ = time_raster(intensity, timestamps, metadata.get('previous_timestamp'))
        data_out = resize(raster, out_size, anti_aliasing=True)
        path = os.path.join(self.plot_dir, filename)
        paint_arr_to_jpg(data_out / norm_factor, filename=path, vmax=200, vmin=0.5, scaling='log')
        return path


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

    def _init_visitor_db(self):
        """Initialize local SQLite database for visitor logging."""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.visitor_db_path = os.path.join(base_dir, 'visitors.db')
            conn = sqlite3.connect(self.visitor_db_path)
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visited_at_utc TEXT NOT NULL,
                    path TEXT NOT NULL,
                    ip TEXT,
                    user_agent TEXT
                )
                """
            )
            conn.commit()
            conn.close()
        except Exception as e:
            # If visitor DB fails, log and continue; core functionality should not break.
            self.visitor_db_path = None
            self.log.error(f"Failed to initialize visitor DB: {e}")

    def record_visit(self, req):
        """Record a single page visit in the visitor database."""
        db_path = getattr(self, 'visitor_db_path', None)
        if not db_path:
            return
        try:
            headers = getattr(req, "headers", {}) or {}
            # Normalize header keys to lowercase for safety
            xff = None
            try:
                xff = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For")
            except Exception:
                pass

            if xff:
                ip = xff.split(",", 1)[0].strip()
            else:
                # FastAPI/Starlette style: request.client.host
                client = getattr(req, "client", None)
                ip = getattr(client, "host", "") if client else ""

            path = getattr(getattr(req, "url", None), "path", "/")
            user_agent = ""
            try:
                user_agent = headers.get("user-agent", "") or ""
            except Exception:
                user_agent = ""

            # Skip noisy internal probes (e.g. python-requests health checks)
            # and local 127.0.0.1 traffic, which are not real visitors.
            if "python-requests" in user_agent.lower():
                return
            if ip in ("127.0.0.1", "::1", ""):
                return

            visited_at = datetime.utcnow().isoformat() + 'Z'

            # Helpful debug log for IP issues
            self.log.info(
                f"record_visit: path={path} ip={ip} "
                f"xff={xff} "
            )

            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO visits (visited_at_utc, path, ip, user_agent) VALUES (?, ?, ?, ?)",
                (visited_at, path, ip, user_agent[:512]),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.log.error(f"Failed to record visit: {e}")

    def run_ai_summary(self):
        """Generate an AI summary of the latest spectrum using Gemini."""
        current_time = time.time()
        if current_time - self.last_ai_attempt < self.ai_summary_interval:
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            # Silent if not configured
            return

        snapshot = self._get_spectrum_arrays()
        if not len(snapshot[3]):
            return
        self.last_ai_attempt = current_time
        try:
            image_path = self.save_latest_data_to_jpg(snapshot=snapshot, filename="ai-input.jpg")
            if not os.path.exists(image_path):
                self.log.warning(f"AI summary: image not found at {image_path}")
                return

            # Gemini Python SDK (see https://ai.google.dev/gemini-api/docs)
            from google import genai

            client = genai.Client(api_key=api_key)

            with open(image_path, "rb") as f:
                img_bytes = f.read()

            prompt = (
                "read this spectrum from ovro-lwa, low frequency dynamic spectrum\n"
                "5min segment, top to down is 16MHz to 85MHz, left to right is early to late. duration is 5min\n"
                "describe what is in the spectrum whether it is radio burst or quiet sun\n"
                "describe short and precise"
            )

            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": img_bytes,
                                }
                            },
                        ],
                    }
                ],
            )

            text = (getattr(response, "text", "") or "").strip()
            if text:
                with self.ai_summary_lock:
                    self.latest_ai_summary = text
                    self.last_ai_summary_time = current_time
                if self.verbose:
                    self.log.info(f"AI summary updated: {text}")
        except Exception as e:
            # Log and continue; AI summaries are optional
            self.log.error(f"Error generating AI summary: {e}")


    def run_type3_detection(self):
        """Cached inference on a timestamped snapshot, outside the ingest thread."""
        current_time = time.time()
        if current_time - self.last_detection_attempt < self.detection_interval:
            return
        snapshot = self._get_spectrum_arrays()
        metadata, intensity, _, timestamps = snapshot
        if len(timestamps) < 2:
            return
        self.last_detection_attempt = current_time
        try:
            if self.detection_model is None:
                from ultralytics import YOLO
                model_path = os.path.join(os.path.dirname(__file__), 'model', 'bestv8.iw.pt')
                if not os.path.exists(model_path):
                    self.log.error(f"YOLO model not found at {model_path}")
                    return
                self.detection_model = YOLO(model_path)
            _, window_start, window_end = time_raster(
                intensity, timestamps, metadata.get('previous_timestamp'))
            image_path = self.save_latest_data_to_jpg(snapshot=snapshot, filename='detection-input.jpg')
            results = self.detection_model.predict(
                image_path, conf=self.detection_confidence, iou=0.6, verbose=False)
            detections = []
            if results:
                result = results[0]
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        geometry = source_box(boxes.xyxy[i].cpu().numpy(), result.orig_shape,
                                              window_start, window_end)
                        class_id = int(boxes.cls[i].cpu().numpy())
                        detections.append({
                            'id': i, 'class_id': class_id,
                            'class': 'type3' if class_id == 0 else 'type3b',
                            'confidence': float(boxes.conf[i].cpu().numpy()),
                            **geometry,
                            'timestamp': float(timestamps[-1]),
                            'boot_id': metadata['boot_id'],
                            'frame_sequence': metadata['sequence'],
                        })
            with self.detection_lock:
                self.latest_detections = detections
                self.last_detection_time = time.time()
                self.detection_window = {
                    'window_start': window_start, 'window_end': window_end,
                    'time_anchor': float(timestamps[-1]),
                    'boot_id': metadata['boot_id'], 'frame_sequence': metadata['sequence'],
                }
        except Exception as exc:
            self.log.error(f"Error in Type 3 detection: {exc}")

    def analysis_loop(self, callback):
        while not self.shutdown_event.is_set():
            try:
                callback()
            except Exception as exc:
                self.log.error(f"Analysis worker error: {exc}")
            self.shutdown_event.wait(0.25)

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

            string_time = header["last_block_time"]
            frame_mjd = Time(string_time, format='iso', scale='utc').mjd
            # Publish both polarizations and their timestamp as one integration.
            with self.buffer_lock:
                index = self.buffer_index
                self.ring_buffer[index, :] = averaged_data
                self.ring_buffer_v[index, :] = averaged_data_v
                self.ring_arr_mjd[index, 0] = frame_mjd
                self.buffer_index = (index + 1) % self.buffer_length
                self.frames_received += 1
            
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
        self.analysis_threads = []
        for callback in (self.run_type3_detection, self.run_ai_summary):
            worker = threading.Thread(target=self.analysis_loop, args=(callback,), daemon=True)
            worker.start()
            self.analysis_threads.append(worker)
        for cache, name in ((self.radio_images, 'fine-channel-images'),
                            (self.antenna_flags, 'antenna-flags'),
                            (self.imaging_status, 'imaging-status')):
            worker = threading.Thread(target=cache.run, args=(self.shutdown_event,),
                                      daemon=True, name=name)
            worker.start()
            self.analysis_threads.append(worker)
        

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
        
        for worker in getattr(self, "analysis_threads", []):
            worker.join(timeout=1.0)

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
        
        with self.buffer_lock:
            indices = (np.arange(self.buffer_index - n_frames,
                                 self.buffer_index) % self.buffer_length)
            if pol == 'I':
                values = self.ring_buffer[indices].copy()
            elif pol == 'V':
                values = self.ring_buffer_v[indices].copy()
            else:
                raise ValueError(f"Invalid polarization: {pol}")
            return values, self.ring_arr_mjd[indices].copy()

    def _get_spectrum_arrays(self, since=None, boot_id=None, limit=None, before=None):
        """Atomic paired samples; small recent snapshots and paged older history."""
        with self.buffer_lock:
            sequence = self.frames_received
            available = min(sequence, self.buffer_length)
            oldest = sequence - available + 1
            limit = self.buffer_length if limit is None else min(self.buffer_length, max(1, limit))
            history = before is not None and boot_id == self.boot_id
            reset = not history and (since is None or boot_id != self.boot_id or since < 0
                                     or since > sequence or sequence - since > available)
            end = min(before, sequence) if history else sequence
            count = max(0, min(limit, end - oldest + 1)) if history or reset else sequence - since
            first = end - count + 1
            indices = (np.arange(self.buffer_index - (sequence - end) - count,
                                 self.buffer_index - (sequence - end)) % self.buffer_length)
            intensity = self.ring_buffer[indices].copy()
            circular = self.ring_buffer_v[indices].copy()
            times = self.ring_arr_mjd[indices, 0].copy()
            previous = None
            if count and first > oldest:
                previous_index = (self.buffer_index - (sequence - first) - 2) % self.buffer_length
                previous = float((self.ring_arr_mjd[previous_index, 0] - 40587.0) * 86400.0)
            index = self.buffer_index
        has_more = bool(count and (history or reset) and first > oldest)
        metadata = {
            'sequence': sequence, 'first_sequence': first if count else None,
            'boot_id': self.boot_id, 'reset': reset, 'history': history,
            'has_more': has_more, 'next_before': first - 1 if has_more else None,
            'previous_timestamp': previous, 'buffer_length': self.buffer_length,
            'buffer_index': index, 'timestamp': time.time(),
        }
        return metadata, intensity, circular, (times - 40587.0) * 86400.0

    def get_spectrum_snapshot(self, since=None, boot_id=None, limit=None, before=None):
        metadata, intensity, circular, times = self._get_spectrum_arrays(since, boot_id, limit, before)
        # Corrupt/non-finite samples remain missing, rather than invalid JSON.
        def json_rows(values):
            if np.isfinite(values).all():
                return values.tolist()
            rows = values.astype(object)
            rows[~np.isfinite(values)] = None
            return rows.tolist()
        return {
            'I': json_rows(intensity),
            'V': json_rows(circular),
            'timestamps': times.tolist(),
            **metadata,
        }

    def get_spectrum_binary(self, since=None, boot_id=None, limit=None, before=None):
        """Aligned header + float64 times + float32 I/V, without JSON array copies."""
        metadata, intensity, circular, times = self._get_spectrum_arrays(since, boot_id, limit, before)
        metadata.update(schema='lwa-spectrum-iv-v1', rows=len(times), bins=768,
                        dtype='float32', byteorder='little', order=['I', 'V'])
        header = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
        header += b' ' * (-(4 + len(header)) % 8)
        return b''.join((
            struct.pack('<I', len(header)), header,
            times.astype('<f8', copy=False).tobytes(),
            intensity.astype('<f4', copy=False).tobytes(),
            circular.astype('<f4', copy=False).tobytes(),
        ))

    def start_webserver(self):
        """Start the FastAPI web server for live spectrum display (via Gunicorn)"""
        if not WEB_AVAILABLE:
            self.log.warning("Web server dependencies not available, cannot start web server.")
            return

        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            frontend_dist = os.path.join(base_dir, 'frontend', 'dist')

            app = FastAPI()
            app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=3)

            # CORS configuration similar to previous Flask+CORS
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            # Serve built frontend assets
            assets_dir = os.path.join(frontend_dist, "assets")
            if os.path.isdir(assets_dir):
                app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

            @app.get("/")
            async def index(request: Request):
                # Record visitor for main SPA entry point
                self.record_visit(request)
                index_path = os.path.join(frontend_dist, "index.html")
                if index_path and os.path.exists(index_path):
                    return FileResponse(index_path, media_type="text/html")
                return JSONResponse(
                    {"error": "frontend build not found"},
                    status_code=500,
                )

            @app.get("/data")
            async def get_data():
                if self.frames_received > 0:
                    latest_data = self.get_latest_data(n_frames=1)[0]
                    if latest_data.size > 0:
                        data_json = latest_data[0].tolist()
                        return JSONResponse(data_json)
                return JSONResponse([])

            @app.get("/spectrum")
            def get_spectrum(since: int = None, boot_id: str = None, format: str = 'json',
                             limit: int = Query(600, ge=1, le=600), before: int = Query(None, ge=0)):
                if since is not None and before is not None:
                    raise HTTPException(status_code=400, detail='Use since or before, not both')
                headers = {'Cache-Control': 'no-store'}
                if format == 'binary':
                    return Response(self.get_spectrum_binary(since, boot_id, limit, before),
                                    media_type='application/octet-stream', headers=headers)
                if format != 'json':
                    raise HTTPException(status_code=400, detail='Unknown spectrum format')
                return JSONResponse(self.get_spectrum_snapshot(since, boot_id, limit, before), headers=headers)

            @app.get('/radio-images')
            def get_radio_images():
                return JSONResponse(self.radio_images.metadata(), headers={'Cache-Control': 'no-store'})

            @app.get('/antenna-flags')
            def get_antenna_flags():
                return JSONResponse(self.antenna_flags.metadata(), headers={'Cache-Control': 'no-store'})

            @app.get('/imaging-status')
            def get_imaging_status():
                return JSONResponse(self.imaging_status.metadata(), headers={'Cache-Control': 'no-store'})

            @app.get('/radio-images/{target}.png')
            def get_radio_image(target: int, request: Request, version: str = None):
                if target not in IMAGE_FREQUENCIES:
                    raise HTTPException(status_code=404, detail='Unknown image channel')
                cached = self.radio_images.image(target, version)
                if cached is None:
                    raise HTTPException(status_code=404 if version else 503, detail='Image not available')
                png, stamp, selected_version = cached
                etag = f'"{selected_version}-{target}"'
                headers = {'ETag': etag, 'X-Observation-Timestamp': stamp,
                           'Cache-Control': 'public, max-age=86400, immutable' if version else 'no-store'}
                if request.headers.get('if-none-match') == etag:
                    return Response(status_code=304, headers=headers)
                return Response(png, media_type='image/png', headers=headers)

            @app.get("/type3detect")
            async def get_type3_detections():
                with self.detection_lock:
                    detections = self.latest_detections.copy()
                    window = self.detection_window.copy()

                current_time = time.time()
                response_data = {
                    "detections": detections,
                    "timestamp": current_time,
                    **window,
                    "count": len(detections),
                    "last_detection_time": self.last_detection_time,
                    "confidence_threshold": self.detection_confidence,
                }
                return JSONResponse(response_data)

            @app.get("/refresh")
            async def refresh_data(n_frames: int = 600):
                if self.frames_received > 0:
                    latest_data = self.get_latest_data(n_frames=n_frames)[0]
                    if latest_data.size > 0:
                        data_json = latest_data.tolist()
                        return JSONResponse(
                            {
                                "data": data_json,
                                "buffer_length": self.buffer_length,
                                "buffer_index": self.buffer_index,
                                "timestamp": time.time(),
                            }
                        )
                return JSONResponse(
                    {
                        "data": [],
                        "buffer_length": self.buffer_length,
                        "buffer_index": self.buffer_index,
                        "timestamp": time.time(),
                    }
                )

            @app.get("/ai-summary")
            async def get_ai_summary():
                with self.ai_summary_lock:
                    summary = self.latest_ai_summary
                    ts = self.last_ai_summary_time
                return JSONResponse(
                    {
                        "summary": summary,
                        "timestamp": ts,
                    }
                )

            @app.get("/visitors/recent")
            async def visitors_recent():
                db_path = getattr(self, "visitor_db_path", None)
                if not db_path:
                    return JSONResponse([])
                try:
                    conn = sqlite3.connect(db_path)
                    cur = conn.cursor()
                    cur.execute(
                        """
                        SELECT visited_at_utc, path, ip, user_agent
                        FROM visits
                        ORDER BY visited_at_utc DESC
                        LIMIT 100
                        """
                    )
                    rows = cur.fetchall()
                    conn.close()
                    data = [
                        {
                            "visited_at_utc": r[0],
                            "path": r[1],
                            "ip": r[2],
                            "user_agent": r[3],
                        }
                        for r in rows
                    ]
                    return JSONResponse(data)
                except Exception as e:
                    self.log.error(f"Failed to load recent visitors: {e}")
                    return JSONResponse(
                        {"error": "failed to load visitors"},
                        status_code=500,
                    )

            @app.get("/visitors/count")
            async def visitors_count():
                """Return total number of recorded visits."""
                db_path = getattr(self, "visitor_db_path", None)
                if not db_path:
                    return JSONResponse({"count": 0})
                try:
                    conn = sqlite3.connect(db_path)
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM visits")
                    row = cur.fetchone()
                    conn.close()
                    count = int(row[0]) if row and row[0] is not None else 0
                    return JSONResponse({"count": count})
                except Exception as e:
                    self.log.error(f"Failed to load visitor count: {e}")
                    return JSONResponse(
                        {"count": 0, "error": "failed to load visitor count"},
                        status_code=500,
                    )

            @app.get("/debug/request")
            async def debug_request(request: Request):
                """Debug endpoint to inspect client IP and headers."""
                try:
                    headers = {k: v for k, v in request.headers.items()}
                    client_host = request.client.host if request.client else None
                    info = {
                        "client_host": client_host,
                        "x_forwarded_for": headers.get("x-forwarded-for"),
                        "headers": headers,
                    }
                    return JSONResponse(info)
                except Exception as e:
                    self.log.error(f"/debug/request failed: {e}")
                    return JSONResponse({"error": "debug failed"}, status_code=500)

            self.app = app
            web_port = 9527
            self.log.info(f"Web server starting on http://localhost:{web_port} (FastAPI + Uvicorn)")

            def run_server():
                # Reduce noisy access logs from Uvicorn; keep only warnings/errors.
                uvicorn.run(
                    self.app,
                    host="127.0.0.1",
                    port=web_port,
                    log_level="warning",
                    access_log=False,
                )

            self.webserver_thread = threading.Thread(
                target=run_server,
                daemon=True,
            )
            self.webserver_thread.start()

            # Give the server a brief moment to start
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
    parser.add_argument('--save-all-to-file', action=argparse.BooleanOptionalAction,
                       help='Save completed ring buffers to NPZ files (default: enabled)',
                       default=True)
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
