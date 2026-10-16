"""Bounded, read-only loading of recent normalized beam NPZ archives."""
from datetime import datetime, timezone
from pathlib import Path
import zipfile
import numpy as np

ARCHIVE_NORMALIZATION = 1e4 * 24


def load_npz_history(directory, now, capacity, log):
    cutoff=now-3600
    days={datetime.fromtimestamp(cutoff,timezone.utc).strftime('%Y-%m-%d'),
          datetime.fromtimestamp(now,timezone.utc).strftime('%Y-%m-%d')}
    candidates=[]
    for day in sorted(days):
        for path in (Path(directory)/day).glob('*.npz'):
            try:
                stamp=datetime.strptime(path.stem,'%Y-%m-%d_%H_%M_%S').replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
            # File names describe save time, not the earliest row in a chunk.
            if cutoff-600 <= stamp <= now+1:
                candidates.append((stamp,path))
    i=np.empty((0,768),np.float32);v=i.copy();times=np.empty(0)
    info={'source':'archive','frames':0,'files':0,'legacy_mjd_rotations':0,'skipped_files':0}
    for _,path in sorted(candidates)[-32:]:
        try:
            with zipfile.ZipFile(path) as zipped:
                if sum(item.file_size for item in zipped.infolist()) > 64*1024*1024:
                    raise ValueError('archive exceeds per-file size limit')
            with np.load(path,allow_pickle=False) as archive:
                data=archive['data'];mjd=np.asarray(archive['mjd'],dtype=np.float64).reshape(-1)
                frequency=np.asarray(archive['freq'],dtype=np.float64)
                expected=np.linspace(196*(600/8192),196*((600+3072-1)/8192),768)
                if frequency.shape!=(768,) or not np.allclose(frequency,expected,rtol=0,atol=1e-5):
                    raise ValueError('incompatible frequency grid')
                if data.shape != (len(mjd),768,2) or not 0<len(mjd)<=7201:
                    raise ValueError('incompatible paired spectrum shape')
                if not np.issubdtype(data.dtype,np.number) or not np.isfinite(mjd).all():
                    raise ValueError('non-numeric or invalid archive')
                version=None
                if 'archive_version' in archive:
                    metadata=np.asarray(archive['archive_version'])
                    if (metadata.shape!=() or metadata.dtype.kind not in 'iuf' or
                            not np.isfinite(metadata).item() or metadata.item() not in (1,2)):
                        raise ValueError('invalid or unsupported archive version')
                    version=int(metadata.item())
                rotated=False
                if np.any(np.diff(mjd)<0):
                    if version is not None and version>=2:
                        raise ValueError('non-chronological versioned archive')
                    # Legacy writer saved chronological I/V but physical-ring MJD.
                    # Rotate ONLY the cyclic time array, never the power planes.
                    mjd=np.roll(mjd,-int(np.argmin(mjd)))
                    if np.any(np.diff(mjd)<0):
                        raise ValueError('non-cyclic timestamp corruption')
                    rotated=True
                normalized_by=float(archive['normalization_factor']) if 'normalization_factor' in archive else ARCHIVE_NORMALIZATION
                if not np.isfinite(normalized_by) or normalized_by<=0:
                    raise ValueError('invalid normalization factor')
                stamps=(mjd-40587)*86400
                keep=(stamps>=cutoff-1e-6)&(stamps<=now+1e-6)
                if not np.any(keep):continue
                incoming_i=np.asarray(data[keep,:,0]*np.float32(normalized_by),np.float32)
                incoming_v=np.asarray(data[keep,:,1]*np.float32(normalized_by),np.float32)
                times=np.concatenate((times,stamps[keep]));i=np.concatenate((i,incoming_i));v=np.concatenate((v,incoming_v))
                # Later files win overlap. Unique indices are returned in UTC order.
                _,backward=np.unique(times[::-1],return_index=True)
                order=(len(times)-1-backward)[-capacity:]
                times,i,v=times[order],i[order],v[order]
                info['files']+=1
                info['legacy_mjd_rotations']+=int(rotated)
        except Exception as exc:
            info['skipped_files']+=1
            log.warning('Archive prefill skipped %s (%s)',path.name,type(exc).__name__)
    info['frames']=len(times)
    return i,v,times,info
