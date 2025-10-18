#!/usr/bin/env python3
"""
Simple script to plot NPZ files from the stream receiver.

The NPZ files contain spectrum data with:
- data: 3D array (time_frames, frequency_bins, 2) with I and V polarizations
- mjd: Modified Julian Date timestamps
- freq: Frequency array in MHz

This script plots only the Stokes I waterfall plot (time vs frequency).

Usage Examples:
    # Plot the latest NPZ file
    conda activate solarml && python3 plot_npz.py --latest
    
    # Plot a specific file
    conda activate solarml && python3 plot_npz.py /path/to/file.npz
    
    # Plot all NPZ files in the directory
    conda activate solarml && python3 plot_npz.py --all
    
    # Plot without saving (just display)
    conda activate solarml && python3 plot_npz.py --latest --no-save --show
    
    # Plot with custom save directory
    conda activate solarml && python3 plot_npz.py --latest --save-dir /custom/path/
    
    # Live plot mode - continuously monitor and plot new data
    conda activate solarml && python3 plot_npz.py --live_plot
    
    # Live plot with custom chunk size
    conda activate solarml && python3 plot_npz.py --live_plot --chunk-size 12
    
    # Plot all files in a date directory
    conda activate solarml && python3 plot_npz.py --all-day --date-dir /common/lwa/stream_spec_npz/2025-10-18/
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import argparse
import os
import glob
from datetime import datetime

def load_npz_data(npz_file):
    """Load data from NPZ file and return structured data."""
    print(f"Loading data from: {npz_file}")
    
    data = np.load(npz_file)
    
    # Extract data arrays
    spectrum_data = data['data']  # Shape: (time_frames, frequency_bins, 2)
    mjd_times = data['mjd']       # Shape: (time_frames, 1)
    frequencies = data['freq']    # Shape: (frequency_bins,)
    
    # Close the file
    data.close()
    
    print(f"Data shape: {spectrum_data.shape}")
    print(f"Time range: {mjd_times.min():.6f} to {mjd_times.max():.6f} MJD")
    print(f"Frequency range: {frequencies.min():.2f} to {frequencies.max():.2f} MHz")
    return spectrum_data, mjd_times, frequencies

def plot_spectrums(npz_file_lst, save_plot=True, show_plot=False, output_path=None):
    """Plot the spectrum data from a list of NPZ files.
    
    Args:
        npz_file_lst: List of NPZ file paths to plot
        save_plot: Whether to save the plot
        show_plot: Whether to show the plot interactively
        output_path: Optional custom output path for the plot. If None, uses default naming.
    """
    
    # Load data
    spectrum_data_lst = []
    mjd_times_lst = []
    for npz_file in npz_file_lst:
        spectrum_data_tmp, mjd_times_tmp, frequencies_tmp = load_npz_data(npz_file)
        spectrum_data_lst.append(spectrum_data_tmp)
        mjd_times_lst.append(mjd_times_tmp)

    spectrum_data = np.concatenate(spectrum_data_lst, axis=0)
    mjd_times = np.concatenate(mjd_times_lst, axis=0)
    frequencies = frequencies_tmp

    # Extract I and V polarizations
    I_data = spectrum_data[:, :, 0]  # Stokes I
    V_data = spectrum_data[:, :, 1]  # Stokes V
    
    # Convert MJD to datetime for plotting
    times = Time(mjd_times.flatten(), format='mjd').datetime
    
    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), dpi=120)
    
    # Waterfall plot (Stokes I)
    vmax_plot = np.percentile(I_data, 99.5)
    vmin_plot = np.percentile(I_data, 2)
    if I_data.shape[0] > 1:
        im = ax.imshow(I_data.T, aspect='auto', origin='lower', 
                      cmap='viridis', norm=plt.matplotlib.colors.LogNorm(vmax=vmax_plot, vmin=vmin_plot),
                      extent=[0, len(times)-1, frequencies.min(), frequencies.max()])
        ax.set_ylabel('Frequency (MHz)')
        ax.set_xlabel('Time (UTC) on ' + times[0].strftime('%Y-%m-%d'))
        
        # Set custom x-axis labels for time
        n_ticks = min(5, len(times))  # Limit number of ticks
        tick_indices = np.linspace(1, len(times)-1, n_ticks, dtype=int)
        tick_labels = [times[i].strftime('%H:%M:%S') for i in tick_indices]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, aspect=22)
        # cbar title inside the colorbar
        cbar.set_label('[s.f.u.]', labelpad=-35)
    else:
        ax.text(0.5, 0.5, 'Insufficient data for waterfall plot', 
                ha='center', va='center', transform=ax.transAxes)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        if output_path is None:
            # Use default naming from the last file in the list
            plot_filename = npz_file_lst[-1].replace('.npz', '_plot.png')
        else:
            plot_filename = output_path
        
        # Ensure output directory exists
        output_dir = os.path.dirname(plot_filename)
        if output_dir:  # Only create if there's a directory path
            os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(plot_filename, dpi=120, bbox_inches='tight')
        full_path_filename = os.path.abspath(plot_filename)
        print(f"Plot saved as: {plot_filename}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig, full_path_filename

def find_latest_npz(directory='/common/lwa/stream_spec_npz/'):
    """Find the latest NPZ file in the directory."""
    npz_files = glob.glob(os.path.join(directory, '*.npz'))
    if not npz_files:
        return None
    
    # Sort by modification time and return the latest
    latest_file = max(npz_files, key=os.path.getmtime)
    return latest_file

def continues_plot(data_dir, out_dir, chunk_N=6, current_f_idx=0, make_latest_copy=True, 
    latest_spectra_abs_fname='/common/webplots/lwa-data/latest_spectrum.png'):
    """
    Plot the next chunk of NPZ files from data_dir.
    
    The function checks data_dir and plots the next chunk_N files starting from current_f_idx.
    Files are processed in chunks, and current_f_idx should always be a multiple of chunk_N.
    
    Examples:
        - 8 files, chunk_N=6, current_f_idx=0 → plot files 0-5, return 6
        - 13 files, chunk_N=6, current_f_idx=6 → plot files 6-11, return 12
        - 14 files, chunk_N=6, current_f_idx=12 → skip (need 18 files), return 12
    
    Args:
        data_dir: Directory containing NPZ files
        out_dir: Output directory for plots (e.g., /common/webplots/lwa-data/qlook_spectra)
        chunk_N: Number of files to plot together (default: 6)
        current_f_idx: Current file index (should be multiple of chunk_N)
        make_latest_copy: if yes, copy the output plot to /common/webplots/lwa-data/latest_spectrum.png

    Returns:
        int: Updated current_f_idx (index of next file to process), or current_f_idx if no plotting done
    """
    # Get all NPZ files in the directory, sorted by name (which includes timestamp)
    npz_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    total_files = len(npz_files)
    
    print(f"Found {total_files} NPZ files in {data_dir}")
    print(f"Current index: {current_f_idx}, need at least: {current_f_idx + chunk_N}")
    
    # Calculate the next chunk indices
    start_idx = current_f_idx
    end_idx = start_idx + chunk_N
    
    # Check if we have enough files
    if end_idx > total_files:
        print(f"Not enough files: have {total_files}, need at least {end_idx}. Skipping plot.")
        return current_f_idx
    
    # Get the files for this chunk
    files_to_plot = npz_files[start_idx:end_idx]
    
    if not files_to_plot:
        print(f"No files to plot")
        return current_f_idx
    
    # Get the timestamp from the last file for the output filename
    # Format: 2025-10-18_15_23_43.npz -> 20251018-152343-lwa_beam.png
    last_file = files_to_plot[-1]
    basename = os.path.basename(last_file)
    # Extract timestamp: 2025-10-18_15_23_43
    timestamp_str = basename.replace('.npz', '').replace('_', '').replace('-', '')
    # Format: YYYYMMDD-HHMMSS
    output_filename = f"{timestamp_str[:8]}-{timestamp_str[8:]}-lwa_beam.png"
    
    # Create output directory structure: out_dir/YYYY/MM/DD/
    # Extract date from the first file
    first_file = files_to_plot[0]
    first_basename = os.path.basename(first_file)
    # Extract date: 2025-10-18_...
    date_parts = first_basename.split('_')
    year, month, day = date_parts[0].split('-')
    
    output_subdir = os.path.join(out_dir, year, month, day)
    os.makedirs(output_subdir, exist_ok=True)
    
    output_path = os.path.join(output_subdir, output_filename)
    
    # Plot the spectrums
    print(f"\nPlotting files {start_idx} to {end_idx-1} (total: {chunk_N} files)")
    print(f"Files: {os.path.basename(files_to_plot[0])} ... {os.path.basename(files_to_plot[-1])}")
    print(f"Output: {output_path}")
    
    # Use plot_spectrums function
    fig, full_path_filename = plot_spectrums(files_to_plot, save_plot=True, show_plot=False, output_path=output_path)
    
    if make_latest_copy:
        shutil.copy(full_path_filename, latest_spectra_abs_fname)

    # Return the next index (should be multiple of chunk_N)
    return end_idx

def main():
    parser = argparse.ArgumentParser(description='Plot NPZ files from stream receiver')
    parser.add_argument('npz_file', nargs='?', help='NPZ file to plot')
    parser.add_argument('--latest', action='store_true', 
                       help='Plot the latest NPZ file', default=False)
    parser.add_argument('--all', action='store_true', 
                       help='Plot all NPZ files in the directory')
    parser.add_argument('--save-dir', default='./', 
                       help='Directory containing NPZ files (default: /common/lwa/stream_spec_npz/)')
    parser.add_argument('--no-save', action='store_true', 
                       help='Do not save plots to files')
    parser.add_argument('--show', action='store_true', 
                       help='Show plots interactively')
    parser.add_argument('--live_plot', action='store_true',
                       help='Continuously monitor and plot new data every N minutes')
    parser.add_argument('--all-day', action='store_true',
                       help='Plot all files in a date directory in chunks')
    parser.add_argument('--date-dir', default=None,
                       help='Date directory path for --all-day mode (e.g., /common/lwa/stream_spec_npz/2025-10-18/)')
    parser.add_argument('--chunk-size', type=int, default=6,
                       help='Number of files to plot together (default: 6)')
    parser.add_argument('--data-dir-root', default='/common/lwa/stream_spec_npz/',
                       help='Root directory for data (default: /common/lwa/stream_spec_npz/)')
    parser.add_argument('--out-dir', default='/common/webplots/lwa-data/qlook_spectra',
                       help='Output directory for plots (default: /common/webplots/lwa-data/qlook_spectra)')
    parser.add_argument('--wait-minutes', type=int, default=3,
                       help='Wait time in minutes between checks (default: 5)')
    parser.add_argument('--latest05min-plot-fname', default='/common/webplots/lwa-data/05min_plot.png',
                       help='Latest 5 minutes plot file name (default: /common/webplots/lwa-data/05min_plot.png)')
    
    args = parser.parse_args()
    

    if not args.date_dir:
        args.date_dir = os.path.join(args.data_dir_root, datetime.now().strftime('%Y-%m-%d'))
        if not os.path.exists(args.date_dir):
            print(f"Error: Directory does not exist: {args.date_dir}")
            return
        print(f"Date directory: {args.date_dir}")
        print("=" * 60)

    # Handle all-day mode
    if args.all_day:
        if not args.date_dir:
            print("Error: --date-dir is required when using --all-day")
            print("Example: python3 plot_npz.py --all-day --date-dir /common/lwa/stream_spec_npz/2025-10-18/")
            return
        
        if not os.path.exists(args.date_dir):
            print(f"Error: Directory does not exist: {args.date_dir}")
            return
        
        print(f"All-day mode: Processing all files in {args.date_dir}")
        print(f"Chunk size: {args.chunk_size}")
        print(f"Output directory: {args.out_dir}")
        print("=" * 60)
        
        current_idx = 0
        chunks_plotted = 0
        
        while True:
            new_idx = continues_plot(args.date_dir, args.out_dir, 
                                    chunk_N=args.chunk_size, 
                                    current_f_idx=current_idx)
            
            if new_idx == current_idx:
                # No more complete chunks available
                print(f"\nCompleted: Plotted {chunks_plotted} chunk(s)")
                
                # Check if there are remaining files
                npz_files = sorted(glob.glob(os.path.join(args.date_dir, '*.npz')))
                remaining = len(npz_files) - current_idx
                if remaining > 0:
                    print(f"Note: {remaining} file(s) remaining (less than chunk size of {args.chunk_size})")
                break
            
            current_idx = new_idx
            chunks_plotted += 1
            print(f"Progress: Completed chunk {chunks_plotted}, index now at {current_idx}")
            print("-" * 60)
        
        print("=" * 60)
        return
    
    # Handle live plot mode
    if args.live_plot:
        import time
        from datetime import datetime as dt
        
        print(f"Starting live plot mode with chunk size {args.chunk_size}")
        print(f"Data directory root: {args.data_dir_root}")
        print(f"Output directory: {args.out_dir}")
        print(f"Checking every 10 minutes...")
        
        n_files_in_dir = len(glob.glob(os.path.join(args.date_dir, '*.npz')))
        current_f_idx = int(int(n_files_in_dir / args.chunk_size) * args.chunk_size)
        
        while True:
            try:
                # Get current UTC date for directory path
                utc_now = dt.utcnow()
                date_str = utc_now.strftime('%Y-%m-%d')
                data_dir = os.path.join(args.data_dir_root, date_str)
                
                print(f"\n{'='*60}")
                print(f"[{utc_now.strftime('%Y-%m-%d %H:%M:%S')} UTC] Checking {data_dir}")
                
                # Check if directory exists
                if not os.path.exists(data_dir):
                    print(f"Directory does not exist yet: {data_dir}")
                else:
                    # Try to plot the next chunk
                    new_idx = continues_plot(data_dir, args.out_dir, 
                                            chunk_N=args.chunk_size, 
                                            current_f_idx=current_f_idx)
                    
                    if new_idx > current_f_idx:
                        print(f"Successfully plotted chunk. Updated index: {current_f_idx} → {new_idx}")
                        current_f_idx = new_idx
                    else:
                        print(f"No new chunks available yet.")

                    newsest_npz_file = sorted(glob.glob(os.path.join(data_dir, '*.npz')))[-1]
                    fig, full_path_filename = plot_spectrums(
                        [newsest_npz_file], save_plot=True, show_plot=False, output_path=args.latest05min_plot_fname)
                
                # Wait N minutes
                print(f"Waiting {args.wait_minutes} minutes until next check...")
                time.sleep(args.wait_minutes * 60)  # N minutes = N * 60 seconds
                
            except KeyboardInterrupt:
                print("\n\nStopping live plot mode...")
                break
            except Exception as e:
                print(f"Error in live plot: {e}")
                import traceback
                traceback.print_exc()
                print(f"Waiting {args.wait_minutes} minutes before retry...")
                time.sleep(args.wait_minutes * 60)
        
        return
    
    # Determine which files to plot
    files_to_plot = []
    
    if args.all:
        # Plot all NPZ files
        npz_files = glob.glob(os.path.join(args.save_dir, '*.npz'))
        files_to_plot = sorted(npz_files)
        print(f"Found {len(files_to_plot)} NPZ files to plot")
    
    elif args.latest:
        # Plot the latest NPZ file
        latest_file = find_latest_npz(args.save_dir)
        if latest_file:
            files_to_plot = [latest_file]
            print(f"Latest NPZ file: {latest_file}")
        else:
            print(f"No NPZ files found in {args.save_dir}")
            return
    
    elif args.npz_file:
        # Plot specific file
        if os.path.exists(args.npz_file):
            files_to_plot = [args.npz_file]
        else:
            print(f"File not found: {args.npz_file}")
            return
    
    else:
        # Default: plot the latest file
        latest_file = find_latest_npz(args.save_dir)
        if latest_file:
            files_to_plot = [latest_file]
            print(f"Plotting latest NPZ file: {latest_file}")
        else:
            print(f"No NPZ files found in {args.save_dir}")
            print("Usage: python3 plot_npz.py <npz_file> or --latest or --all")
            return
    
    # Plot each file
    for npz_file in files_to_plot:
        try:
            print(f"\n{'='*60}")
            plot_spectrums([npz_file], save_plot=not args.no_save, show_plot=args.show)
        except Exception as e:
            print(f"Error plotting {npz_file}: {e}")
            continue
    
    print(f"\nCompleted plotting {len(files_to_plot)} file(s)")

if __name__ == "__main__":
    main()
