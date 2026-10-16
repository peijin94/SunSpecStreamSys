"""Luna subprocess contract tests (fixture CLI), plus receiver integration."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

class LunaSummaryTests(unittest.TestCase):
    def test_gpt6_luna_cli_receives_image_in_ephemeral_tool_free_job(self):
        self.assertIsNotNone(importlib.util.find_spec('spectrum_ai'),'Luna vision adapter is missing')
        from spectrum_ai import CodexSpectrumClient
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);image=root/'spectrum.png';image.write_bytes(b'image fixture for CLI argument test')
            cli=root/'fixture-cli';cli.write_text('#!'+sys.executable+'\n'+
                'import sys,json,os\nfrom pathlib import Path\n'+
                'Path(sys.argv[0]+".args").write_text(json.dumps(sys.argv[1:]))\n'+
                'Path(sys.argv[0]+".env").write_text(json.dumps({k:k in os.environ for k in ["GEMINI_API_KEY","OPENAI_API_KEY","HOME"]}))\n'+
                'Path(sys.argv[sys.argv.index("--output-last-message")+1]).write_text(json.dumps({"summary":"Fixture summary: no model was called in this unit test."}))\n')
            cli.chmod(0o700)
            client=CodexSpectrumClient(executable=str(cli),work_dir=os.path.relpath(root,Path.cwd()),timeout=5)
            text=client.describe(image,'Describe this test spectrum.')
            self.assertIn('Fixture summary',text)
            args=json.loads(Path(str(cli)+'.args').read_text())
            self.assertEqual(args[args.index('--model')+1],'gpt-6-luna')
            self.assertEqual(args[args.index('--image')+1],str(image.resolve()))
            self.assertIn('--ephemeral',args);self.assertIn('--ignore-user-config',args)
            self.assertIn('features.shell_tool=false',args)
            self.assertIn('features.plugins=false',args)
            self.assertIn('features.browser_use=false',args)
            self.assertIn('features.skip_host_skill_discovery=true',args)
            self.assertEqual(args[args.index('--sandbox')+1],'read-only')
            env=json.loads(Path(str(cli)+'.env').read_text())
            self.assertFalse(env['GEMINI_API_KEY']);self.assertFalse(env['OPENAI_API_KEY'])
            self.assertEqual(list(root.glob('lwa-ai-*')),[],'Job files must be cleaned up')

    def test_rendered_vision_input_has_paired_panels_real_utc_and_night_context(self):
        from spectrum_ai import render_ai_spectrum
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'vision.png'
            # A genuinely night-time source window, not a forced Sun-down flag.
            times=np.array([1792113000.,1792113000.5,1792113060.])+7200
            i=np.full((3,768),240000.,dtype=np.float32)
            prompt,window=render_ai_spectrum(({},i,-.1*i,times),path)
            with Image.open(path) as image:self.assertGreaterEqual(image.width,800)
            self.assertIn('UTC',prompt);self.assertIn('Stokes I',prompt);self.assertIn('V/I',prompt)
            self.assertIn('night',prompt.lower());self.assertIn('2026-10-16',prompt)
            self.assertEqual(window['source_end'],times[-1]);self.assertEqual(window['source_start'],times[0])

    def test_prompt_has_source_ut_time_and_matching_solar_elevation_azimuth(self):
        from spectrum_ai import render_ai_spectrum
        from datetime import datetime, timezone
        from unittest.mock import patch
        import numpy as np
        times=np.array([1792113000.,1792113000.5,1792113060.])
        i=np.full((3,768),240000.,dtype=np.float32)
        expected_time=datetime.fromtimestamp(times[-1],timezone.utc).isoformat(timespec='seconds')
        with tempfile.TemporaryDirectory() as root, patch('check_sun_elevation.get_sun_position',
                return_value={'elevation_deg':-12.25,'azimuth_deg':74.5}) as position:
            prompt,window=render_ai_spectrum(({},i,-.1*i,times),Path(root)/'vision.png')
        self.assertIn('Observation time (UT/UTC): '+expected_time,prompt)
        self.assertIn('Sun elevation: -12.25 deg',prompt)
        self.assertIn('Sun azimuth: 74.50 deg',prompt)
        self.assertIn('clockwise from true north',prompt)
        self.assertAlmostEqual(position.call_args.kwargs['time'].unix,times[-1],places=5)
        self.assertEqual(window['observation_time_utc'],expected_time)
        self.assertEqual(window['sun_elevation_deg'],-12.25)
        self.assertEqual(window['sun_azimuth_deg'],74.5)
        self.assertFalse(window['sun_up'])

    def test_receiver_updates_luna_metadata_and_rate_limits_without_gemini_key(self):
        import threading,time
        from unittest.mock import patch
        import test_polarization_api as polarization
        case=polarization.PolarizationTests();case.setUp();r=case.receiver
        class FixtureClient:
            calls=0
            def describe(self,image,prompt):
                self.calls+=1
                assert Path(image).is_file()
                assert 'night' in prompt.lower()
                return 'Fixture Luna result.'
        with tempfile.TemporaryDirectory() as root:
            r.plot_dir=root;r.ai_model='gpt-6-luna';r.ai_client=FixtureClient()
            r.last_ai_attempt=0;r.ai_summary_interval=60;r.latest_ai_summary=''
            r.ai_summary_lock=threading.Lock();r.last_ai_summary_time=0
            now=time.time();case.frame(10,now-1);case.frame(11,now-.5)
            with patch.dict(os.environ,{'GEMINI_API_KEY':''}),patch('check_sun_elevation.get_sun_position',return_value={'elevation_deg':-12.25,'azimuth_deg':74.5}):
                r.run_ai_summary();r.run_ai_summary()
            self.assertEqual(r.latest_ai_summary,'Fixture Luna result.')
            self.assertEqual(r.ai_client.calls,1)
            result=r.ai_summary_payload()
            self.assertEqual(result['model'],'gpt-6-luna');self.assertEqual(result['provider'],'openai-codex')
            self.assertEqual(result['status'],'ready')
            self.assertAlmostEqual(result['source_end'],now-.5,places=2)

    def test_cli_timeout_cleans_job_and_does_not_leak_stderr(self):
        from spectrum_ai import CodexSpectrumClient,SpectrumSummaryError
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);image=root/'image.png';image.write_bytes(b'fixture')
            cli=root/'fixture-cli';cli.write_text('#!'+sys.executable+'\nimport time,sys\nprint("fixture credential must stay private",file=sys.stderr)\ntime.sleep(10)\n');cli.chmod(0o700)
            with self.assertRaisesRegex(SpectrumSummaryError,'timed out') as raised:
                CodexSpectrumClient(executable=str(cli),work_dir=root,timeout=.2).describe(image,'fixture')
            self.assertNotIn('credential',str(raised.exception))
            self.assertEqual(list(root.glob('lwa-ai-*')),[])

    def test_failed_cli_keeps_prior_interpretation_and_marks_it_stale(self):
        from spectrum_ai import SpectrumSummaryError
        from unittest.mock import patch
        import threading,time
        import test_polarization_api as polarization
        case=polarization.PolarizationTests();case.setUp();r=case.receiver
        class UnavailableClient:
            def describe(self,*args):raise SpectrumSummaryError('Codex image interpretation failed (exit 1)')
        with tempfile.TemporaryDirectory() as root:
            now=time.time();case.frame(10,now-1);case.frame(11,now-.5)
            r.plot_dir=root;r.ai_client=UnavailableClient();r.ai_model='gpt-6-luna'
            r.ai_summary_interval=60;r.last_ai_attempt=0;r.latest_ai_summary='Prior real interpretation.'
            r.last_ai_summary_time=now-30;r.ai_summary_lock=threading.Lock()
            r.ai_window={'source_start':now-60,'source_end':now-30};r.ai_error=''
            with patch('check_sun_elevation.get_sun_position',return_value={'elevation_deg':-12.25,'azimuth_deg':74.5}),self.assertLogs(r.log,level='WARNING'):
                r.run_ai_summary()
            result=r.ai_summary_payload()
            self.assertEqual(result['summary'],'Prior real interpretation.')
            self.assertEqual(result['status'],'stale')
            self.assertEqual(result['source_end'],now-30)

    def test_dropped_integrations_remain_blank_in_vision_input(self):
        from spectrum_ai import render_ai_spectrum
        from matplotlib.figure import Figure
        from unittest.mock import patch
        import numpy as np
        captured=[]
        def capture(figure,*args,**kwargs):
            captured.append(np.ma.getmaskarray(figure.axes[0].images[0].get_array()).copy())
        i=np.full((3,768),240000.,dtype=np.float32)
        times=np.array([1792113000.,1792113001.5,1792113003.])
        with patch.object(Figure,'savefig',capture):
            render_ai_spectrum(({'integration_seconds':.5},i,-.1*i,times),'unused.png')
        self.assertTrue(captured[0].any(),'Dropped frames must not be painted as continuous emission')

    def test_vision_columns_are_time_and_rows_are_frequency(self):
        from spectrum_ai import render_ai_spectrum
        from matplotlib.figure import Figure
        from unittest.mock import patch
        import numpy as np
        captured=[]
        def capture(figure,*args,**kwargs):captured.append(figure.axes[0].images[0].get_array().copy())
        i=np.tile(np.arange(768,dtype=np.float32)+240000,(3,1))
        times=np.array([1792113000.,1792113000.5,1792113001.])
        with patch.object(Figure,'savefig',capture):
            render_ai_spectrum(({},i,-.1*i,times),'unused.png')
        self.assertEqual(captured[0].shape,(768,600),'Vision image must not swap time and frequency')
        np.testing.assert_allclose(captured[0][:,-1],i[-1]/1e4)

if __name__=='__main__':unittest.main()
