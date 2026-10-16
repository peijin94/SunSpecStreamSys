"""Verify the public/dev services cannot share ports or writable runtime state."""
import configparser
from pathlib import Path
import shlex
import unittest

DEPLOY=Path(__file__).resolve().parents[1]/'deploy'

def service(name):
    config=configparser.ConfigParser(interpolation=None,strict=False)
    assert config.read(DEPLOY/name),name
    return config['Service']

class LiveServiceUnitTests(unittest.TestCase):
    def test_public_keeps_existing_address_and_is_the_only_archive_writer(self):
        unit=service('lwa-stream-receiver.service');args=shlex.split(unit['ExecStart'])
        self.assertEqual(unit['WorkingDirectory'],'%h/SunSpecStreamSys')
        self.assertEqual(args[args.index('--web-port')+1],'9527')
        self.assertEqual(args[args.index('--app-mode')+1],'public')
        self.assertEqual(args[args.index('--port')+1],'30005')
        self.assertNotIn('--no-archive',args)

    def test_dev_has_an_independent_port_and_private_runtime_without_archive_writes(self):
        unit=service('lwa-stream-receiver-dev.service');args=shlex.split(unit['ExecStart'])
        self.assertEqual(unit['WorkingDirectory'],'%h/SunSpecStreamSys-live-dev')
        self.assertEqual(args[args.index('--web-port')+1],'9529')
        self.assertEqual(args[args.index('--app-mode')+1],'dev')
        self.assertEqual(args[args.index('--port')+1],'30005')
        self.assertIn('--no-archive',args)
        self.assertIn('--no-save-all-to-file',args)
        self.assertEqual(unit['UMask'],'0077')
        self.assertEqual(unit['ReadOnlyPaths'],'/common/lwa/stream_spec_npz')
        self.assertNotEqual(unit['WorkingDirectory'],service('lwa-stream-receiver.service')['WorkingDirectory'])

if __name__=='__main__':unittest.main()
