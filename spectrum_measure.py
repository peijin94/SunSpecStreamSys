"""Small, CPU-only radial plasma-emission measurements (no model fitting).

Physics follows peijin94/PhySynthTrainer, commit
6e20e0c6636557ef8032d4d9a03287273ea886d8, freqDrift.py.
Frequencies are MHz, times are Unix UTC seconds, density is cm^-3.
"""
import math

SOLAR_RADIUS_KM=696000.0
MAX_POINTS=128
REFERENCE_COMMIT='6e20e0c6636557ef8032d4d9a03287273ea886d8'


def _frequency(r,scale=1.,harmonic=1,model='saito77'):
    if model=='saito77':density=1.36e6*r**-2.14+1.68e8*r**-6.13
    elif model=='leblanc98':density=3.3e5*r**-2+4.1e6*r**-4+8e7*r**-6
    else:density=4.2e4*10**(4.32/r)
    return harmonic*8.93e-3*math.sqrt(scale*density)


def _radius(frequency,scale,harmonic,model):
    lo,hi=1.,100.
    if not _frequency(hi,scale,harmonic,model)<=frequency<=_frequency(lo,scale,harmonic,model):
        raise ValueError('Frequency lies outside this density model at 1–100 solar radii')
    for _ in range(80):
        middle=(lo+hi)/2
        if _frequency(middle,scale,harmonic,model)>frequency:lo=middle
        else:hi=middle
    return (lo+hi)/2


def measure_spectrum(payload):
    if not isinstance(payload,dict) or set(payload)-{'points','model','harmonic','density_scale'}:
        raise ValueError('Provide points and only supported model options')
    raw=payload.get('points')
    if not isinstance(raw,list) or not 2<=len(raw)<=MAX_POINTS:
        raise ValueError('Provide 2–128 [Unix UTC seconds, frequency MHz] points')
    for point in raw:
        if not isinstance(point,(list,tuple)) or len(point)!=2 or any(
                type(x) not in (int,float) or abs(x)>1e100 or not math.isfinite(x) for x in point):
            raise ValueError('Each point needs two finite numbers, not booleans or strings')
        if not 0<=point[0]<=1e11 or not 0<point[1]<=1000:
            raise ValueError('Time must be Unix UTC seconds; frequency must be positive MHz up to 1000')
    ordered=sorted(raw)
    if any(b[0]-a[0]<1e-6 for a,b in zip(ordered,ordered[1:])):
        raise ValueError('Point times must be distinct by at least one microsecond')
    scale=payload.get('density_scale',1.)
    if type(scale) not in (int,float) or not .01<=scale<=100:
        raise ValueError('Density multiplier must be finite and between 0.01 and 100')
    model=payload.get('model','saito77')
    if model not in ('saito77','leblanc98','newkirk'):
        raise ValueError('Unsupported density model')
    harmonic=payload.get('harmonic',1)
    if type(harmonic) is not int or harmonic not in (1,2):
        raise ValueError('Emission harmonic must be integer 1 or 2')
    points=[{'time':time,'frequency_mhz':frequency,'r_sun':_radius(frequency,scale,harmonic,model)}
            for time,frequency in sorted(payload['points'])]
    segments=[]
    for left,right in zip(points,points[1:]):
        dt=right['time']-left['time']
        velocity=(right['r_sun']-left['r_sun'])*SOLAR_RADIUS_KM/dt
        segments.append({'time':left['time']+dt/2,'duration_s':dt,
                         'frequency_mhz':(left['frequency_mhz']+right['frequency_mhz'])/2,
                         'drift_mhz_s':(right['frequency_mhz']-left['frequency_mhz'])/dt,
                         'r_sun':(left['r_sun']+right['r_sun'])/2,
                         'radial_velocity_km_s':velocity,'speed_km_s':abs(velocity)})
    return {'points':points,'segments':segments,'model':model,'harmonic':harmonic,
            'density_scale':scale,'solar_radius_km':SOLAR_RADIUS_KM,
            'method':'adjacent_secants','reference_commit':REFERENCE_COMMIT,
            'assumptions':['Radial plasma emission; no propagation, projection or light-travel-time correction.',
                           'Adjacent-point secants, evaluated at segment mid-times and mean radii.',
                           'r_sun is distance from solar centre, not height above the photosphere.'],
            'warnings':(['Inferred speed exceeds the speed of light; check points and emission/model assumptions.']
                        if any(s['speed_km_s']>=299800 for s in segments) else [])}
