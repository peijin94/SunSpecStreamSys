#!/usr/bin/env python3
"""
Test script to demonstrate the continues_plot function.

Usage:
    conda activate solarml && python3 test_continues_plot.py
"""

from plot_npz import continues_plot
import os

def main():
    # Configuration
    data_dir = '/common/lwa/stream_spec_npz/2025-10-18'
    out_dir = '/common/webplots/lwa-data/qlook_spectra'
    chunk_size = 6
    
    print("=" * 60)
    print("Testing continues_plot function")
    print("=" * 60)
    
    # Test 1: Plot first chunk (files 0-5)
    print("\nTest 1: Plot first chunk (indices 0-5)")
    print("-" * 60)
    current_idx = 0
    new_idx = continues_plot(data_dir, out_dir, chunk_N=chunk_size, current_f_idx=current_idx)
    print(f"Result: Index changed from {current_idx} to {new_idx}")
    
    # Test 2: Plot second chunk (files 6-11)
    print("\nTest 2: Plot second chunk (indices 6-11)")
    print("-" * 60)
    current_idx = new_idx
    new_idx = continues_plot(data_dir, out_dir, chunk_N=chunk_size, current_f_idx=current_idx)
    print(f"Result: Index changed from {current_idx} to {new_idx}")
    
    # Test 3: Try to plot beyond available files
    print("\nTest 3: Attempt to plot beyond available files")
    print("-" * 60)
    current_idx = 180  # Near the end
    new_idx = continues_plot(data_dir, out_dir, chunk_N=chunk_size, current_f_idx=current_idx)
    print(f"Result: Index {'changed' if new_idx != current_idx else 'unchanged'} - from {current_idx} to {new_idx}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()


