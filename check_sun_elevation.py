#!/usr/bin/env python3
"""
Simple script to check if the Sun is above the horizon at OVRO.

Usage:
    # Check current Sun elevation
    python3 check_sun_elevation.py
    
    # Quiet mode (only exit code: 0 if sun up, 1 if sun down)
    python3 check_sun_elevation.py --quiet
    
    # Use in scripts
    if python3 check_sun_elevation.py --quiet; then
        echo "Sun is up!"
    fi
"""

from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u
from datetime import datetime
import sys
import argparse

# OVRO location (Owens Valley Radio Observatory)
# Coordinates: 37.2339° N, 118.2817° W, elevation ~1222 m
OVRO = EarthLocation(lat=37.2339*u.deg, lon=-118.2817*u.deg, height=1222*u.m)

def get_sun_elevation(location=OVRO, time=None):
    """
    Get the Sun's elevation at a given location and time.
    
    Args:
        location: EarthLocation object (default: OVRO)
        time: astropy Time object or None for current time
    
    Returns:
        float: Sun's elevation in degrees
    """
    if time is None:
        time = Time.now()
    
    # Get Sun position
    sun = get_sun(time)
    
    # Transform to altitude-azimuth frame at the location
    altaz = AltAz(obstime=time, location=location)
    sun_altaz = sun.transform_to(altaz)
    
    return sun_altaz.alt.deg

def is_sun_up(location=OVRO, time=None):
    """
    Check if the Sun is above the horizon.
    
    Args:
        location: EarthLocation object (default: OVRO)
        time: astropy Time object or None for current time
    
    Returns:
        bool: True if Sun elevation > 0, False otherwise
    """
    elevation = get_sun_elevation(location, time)
    return elevation > 0

def main():
    parser = argparse.ArgumentParser(description='Check if Sun is above horizon at OVRO')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode: no output, exit code 0 if sun up, 1 if down')
    args = parser.parse_args()
    
    # Get current time
    now = Time.now()
    now_dt = now.datetime
    
    # Get Sun elevation
    elevation = get_sun_elevation(OVRO, now)
    sun_up = is_sun_up(OVRO, now)
    
    if not args.quiet:
        print(f"OVRO Sun Check")
        print(f"=" * 50)
        print(f"Time (UTC):       {now_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Sun elevation:    {elevation:.2f}°")
        print(f"Sun above horizon: {'YES' if sun_up else 'NO'}")
        print(f"=" * 50)
    
    # Exit with appropriate code
    sys.exit(0 if sun_up else 1)

if __name__ == "__main__":
    main()

