"""Cold-start archive bootstrap; every fixture is confined to a temp directory."""
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import test_polarization_api as polarization

NORM=1e4*24

class ArchivePrefillTests(unittest.TestCase):
    def receiver(self, root, capacity=7201):
        case=polarization.PolarizationTests();case.setUp();r=case.receiver
        r.buffer_length=capacity;r.ring_buffer=np.zeros((capacity,768),np.float32)
        r.ring_buffer_v=np.zeros_like(r.ring_buffer);r.ring_arr_mjd=np.zeros((capacity,1))
        r.archive_frames=0;r.save_dir=str(root)
        return case,r

    def archive(self, root, times, values, legacy=False, name=None):
        times=np.asarray(times,float);i=np.repeat(np.asarray(values,dtype=np.float32)[:,None],768,axis=1)
        values=np.stack((i,-.25*i),axis=2)/NORM
        mjd=times/86400+40587
        if legacy: mjd=np.roll(mjd,-1)
        stamp=datetime.fromtimestamp(times[-1],timezone.utc)
        folder=Path(root)/stamp.strftime('%Y-%m-%d');folder.mkdir(parents=True,exist_ok=True)
        path=folder/(name or stamp.strftime('%Y-%m-%d_%H_%M_%S.npz'))
        np.savez(path,data=values,mjd=mjd[:,None],freq=np.linspace(14.35546875,87.83154296875,768))
        return path

    def test_empty_receiver_prefills_paired_raw_jy_without_rearchiving(self):
        now=datetime(2026,10,16,1,10,40,tzinfo=timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as root:
            self.archive(root,[now-20,now-19.5],[240000,480000])
            _,r=self.receiver(root)
            method=getattr(r,'restore_archive_buffer',None)
            self.assertTrue(callable(method),'NPZ cold-start bootstrap is missing')
            with patch('time.time',return_value=now): method()
            data=r.get_spectrum_snapshot()
            self.assertEqual(r.frames_received,2)
            np.testing.assert_allclose(data['I'],np.repeat([[240000],[480000]],768,axis=1),rtol=2e-7)
            np.testing.assert_allclose(data['V'],-.25*np.asarray(data['I']))
            np.testing.assert_allclose(data['timestamps'],[now-20,now-19.5],atol=1e-5,rtol=0)
            self.assertEqual(r.archive_frames,0)

    def test_legacy_physical_ring_mjd_rotation_is_repaired_without_rotating_power(self):
        now=datetime(2026,10,16,1,10,40,tzinfo=timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as root:
            times=now-20+np.arange(4)*.5
            self.archive(root,times,[240000,480000,720000,960000],legacy=True)
            _,r=self.receiver(root)
            with patch('time.time',return_value=now): r.restore_archive_buffer()
            self.assertEqual(r.frames_received,4,'Legacy cyclic MJD arrays must be repaired in memory')
            data=r.get_spectrum_snapshot()
            np.testing.assert_allclose(data['timestamps'],times,rtol=0,atol=1e-5)
            np.testing.assert_allclose(np.asarray(data['I'])[:,0],[240000,480000,720000,960000])
            self.assertEqual(r.prefill_info['legacy_mjd_rotations'],1)

    def test_queued_live_overlap_is_not_counted_as_new_archive_frames(self):
        now=datetime(2026,10,16,1,10,40,tzinfo=timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as root:
            self.archive(root,[now-10.5,now-10],[240000,480000])
            case,r=self.receiver(root)
            with patch('time.time',return_value=now): r.restore_archive_buffer()
            case.frame(99,now-10)
            self.assertEqual(r.frames_received,2,'A queued integration already loaded from NPZ must not duplicate')
            self.assertEqual(r.archive_frames,0)
            case.frame(100,now-9.5)
            self.assertEqual(r.frames_received,3)
            self.assertEqual(r.archive_frames,1)

    def test_midnight_overlap_capacity_stale_and_corrupt_files(self):
        now=datetime(2026,10,16,0,1,tzinfo=timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as root:
            self.archive(root,[now-90,now-89.5],[240000,480000])
            # Newer chunk overlaps one integration and must take precedence.
            self.archive(root,[now-89.5,now-1,now-.5],[720000,960000,1200000])
            # A recently named file containing ancient source times is not live history.
            self.archive(root,[now-7200],[9999999],name='2026-10-16_00_00_59.npz')
            bad=Path(root)/'2026-10-16'/'2026-10-16_00_01_00.npz';bad.write_bytes(b'not an NPZ')
            _,r=self.receiver(root,capacity=3)
            with patch('time.time',return_value=now):r.restore_archive_buffer()
            data=r.get_spectrum_snapshot()
            self.assertEqual(r.frames_received,3)
            np.testing.assert_allclose(np.asarray(data['I'])[:,0],[720000,960000,1200000],rtol=2e-7)
            self.assertTrue(np.all(np.diff(data['timestamps'])>0))
            self.assertEqual(r.prefill_info['skipped_files'],1)

    def test_existing_handoff_is_not_replaced_by_archives(self):
        now=datetime(2026,10,16,1,10,40,tzinfo=timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as root:
            self.archive(root,[now-10],[240000])
            case,r=self.receiver(root);case.frame(100,now-1)
            with patch('time.time',return_value=now):r.restore_archive_buffer()
            self.assertEqual(r.frames_received,1)
            self.assertEqual(r.get_spectrum_snapshot()['I'][0][0],300)

    def test_wrong_frequency_grid_is_not_silently_loaded_as_live_band(self):
        now=datetime(2026,10,16,1,10,40,tzinfo=timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as root:
            path=self.archive(root,[now-10],[240000])
            with np.load(path) as original:
                data,mjd=original['data'].copy(),original['mjd'].copy()
            np.savez(path,data=data,mjd=mjd,freq=np.linspace(100,200,768))
            _,r=self.receiver(root)
            with patch('time.time',return_value=now):r.restore_archive_buffer()
            self.assertEqual(r.frames_received,0,'A different band must not receive the live frequency labels')

    def test_nonfinite_archive_version_cannot_abort_cold_start(self):
        now=datetime(2026,10,16,1,10,40,tzinfo=timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as root:
            path=self.archive(root,[now-20,now-19.5],[240000,480000],legacy=True)
            with np.load(path) as original:payload={key:original[key].copy() for key in original.files}
            np.savez(path,**payload,archive_version=np.inf)
            self.archive(root,[now-2,now-1],[720000,960000])
            _,r=self.receiver(root)
            with patch('time.time',return_value=now):
                try:r.restore_archive_buffer()
                except (OverflowError,EOFError):self.fail('Malformed archive metadata must not abort receiver startup')
            self.assertEqual(r.frames_received,2)
            self.assertEqual(r.prefill_info['skipped_files'],1)
            np.testing.assert_allclose(np.asarray(r.get_spectrum_snapshot()['I'])[:,0],[720000,960000])

    def test_unsupported_or_encrypted_zip_members_cannot_abort_cold_start(self):
        import struct,zipfile
        now=datetime(2026,10,16,1,10,40,tzinfo=timezone.utc).timestamp()
        for kind in ('unsupported_compression','encrypted_flag'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                path=self.archive(root,[now-20,now-19.5],[240000,480000])
                with zipfile.ZipFile(path) as zipped:
                    local=zipped.infolist()[0].header_offset;central=zipped.start_dir
                raw=bytearray(path.read_bytes())
                if kind=='unsupported_compression':
                    struct.pack_into('<H',raw,local+8,99);struct.pack_into('<H',raw,central+10,99)
                else:
                    struct.pack_into('<H',raw,local+6,1);struct.pack_into('<H',raw,central+8,1)
                path.write_bytes(raw)
                self.archive(root,[now-2,now-1],[720000,960000])
                _,r=self.receiver(root)
                with patch('time.time',return_value=now):
                    try:r.restore_archive_buffer()
                    except (RuntimeError,NotImplementedError):self.fail('An unreadable ZIP member must not abort live startup')
                self.assertEqual(r.frames_received,2)
                self.assertEqual(r.prefill_info['skipped_files'],1)

if __name__=='__main__':unittest.main()
