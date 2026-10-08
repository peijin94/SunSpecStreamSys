"""Generate the static browser template; PyYAML is needed only by this tool.

Usage: python tools/generate_antenna_layout.py /path/to/lwa-antpos
No lwa-antpos import, installation, or runtime dependency is used.
"""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import yaml


def ecef(lat, lon, height):
    lat, lon = math.radians(lat), math.radians(lon)
    eccentricity_squared = 6.6943799901413165e-3
    radius = 6378137.0 / math.sqrt(1 - eccentricity_squared * math.sin(lat) ** 2)
    return ((radius + height) * math.cos(lat) * math.cos(lon),
            (radius + height) * math.cos(lat) * math.sin(lon),
            (radius * (1 - eccentricity_squared) + height) * math.sin(lat))


def generate(source):
    path = source / 'lwa_antpos/data/system_cfg.yml'
    rows = yaml.safe_load(path.read_text())
    reference = rows['LWA-000']
    lat, lon, height = reference['latitude'], reference['longitude'], 1222.0
    origin = ecef(lat, lon, height)
    phi, lam = math.radians(lat), math.radians(lon)
    antennas = []
    for row in sorted((r for r in rows.values() if r['corr_num'] >= 0), key=lambda r: r['corr_num']):
        elevation = float(row['elevation'])
        if not math.isfinite(elevation) or elevation > 999990:
            elevation = height
        xyz = ecef(float(row['latitude']), float(row['longitude']), elevation)
        x, y, z = (v - o for v, o in zip(xyz, origin))
        east = -math.sin(lam) * x + math.cos(lam) * y
        north = -math.sin(phi) * math.cos(lam) * x - math.sin(phi) * math.sin(lam) * y + math.cos(phi) * z
        antennas.append({'id': row['corr_num'], 'name': row['antname'],
                         'east_m': round(east, 3), 'north_m': round(north, 3)})
    assert [a['id'] for a in antennas] == list(range(352))
    assert len({a['name'] for a in antennas}) == 352
    assert all(math.isfinite(a['east_m']) and math.isfinite(a['north_m']) for a in antennas)
    return {'source': 'https://github.com/ovro-lwa/lwa-antpos',
            'source_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
            'source_file': str(path.relative_to(source)),
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'coordinate_system': 'WGS84 ECEF rotated to local east/north, relative to LWA-000',
            'reference': {'latitude_deg': lat, 'longitude_deg': lon, 'height_m': height},
            'id_system': 'zero-based MS ANTENNA index, matching corr_num',
            'antennas': antennas}


if __name__ == '__main__':
    layout = generate(Path(sys.argv[1]).resolve())
    output = Path(__file__).resolve().parents[1] / 'frontend/src/features/antennas/layout.json'
    output.write_text(json.dumps(layout, indent=2) + '\n')
    print(f'Generated {len(layout["antennas"])} antennas at {output}')
