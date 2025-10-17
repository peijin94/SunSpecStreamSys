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

def plot_spectrum(npz_file, save_plot=True, show_plot=False):
    """Plot the spectrum data from an NPZ file."""
    
    # Load data
    spectrum_data, mjd_times, frequencies = load_npz_data(npz_file)
    
    # Extract I and V polarizations
    I_data = spectrum_data[:, :, 0]  # Stokes I
    V_data = spectrum_data[:, :, 1]  # Stokes V
    
    # Convert MJD to datetime for plotting
    times = Time(mjd_times.flatten(), format='mjd').datetime
    
    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
    
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
        plot_filename = npz_file.replace('.npz', '_plot.png')
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def find_latest_npz(directory='/common/lwa/stream_spec_npz/'):
    """Find the latest NPZ file in the directory."""
    npz_files = glob.glob(os.path.join(directory, '*.npz'))
    if not npz_files:
        return None
    
    # Sort by modification time and return the latest
    latest_file = max(npz_files, key=os.path.getmtime)
    return latest_file

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
    
    args = parser.parse_args()
    
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
            plot_spectrum(npz_file, save_plot=not args.no_save, show_plot=args.show)
        except Exception as e:
            print(f"Error plotting {npz_file}: {e}")
            continue
    
    print(f"\nCompleted plotting {len(files_to_plot)} file(s)")

if __name__ == "__main__":
    main()
