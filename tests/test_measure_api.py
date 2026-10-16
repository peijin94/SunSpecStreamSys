"""Real ASGI tests; no listener is started and no live buffer is modified."""
import asyncio,json,unittest
from unittest.mock import patch
import test_polarization_api as fixture_module

class MeasureApiTests(unittest.TestCase):
    def setUp(self):
        fixture=fixture_module.PolarizationTests();fixture.setUp();self.receiver=fixture.receiver
        with patch('threading.Thread.start'),patch('time.sleep'):self.receiver.start_webserver()
    def request(self,body):
        async def run():
            messages=[]
            async def receive():return {'type':'http.request','body':body,'more_body':False}
            async def send(message):messages.append(message)
            await self.receiver.app({'type':'http','asgi':{'version':'3.0'},'http_version':'1.1',
                'method':'POST','scheme':'http','path':'/spectrum-measure','raw_path':b'/spectrum-measure',
                'query_string':b'','root_path':'','headers':[(b'content-type',b'application/json')],
                'client':('127.0.0.1',1),'server':('127.0.0.1',9527)},receive,send)
            start=next(m for m in messages if m['type']=='http.response.start')
            return start,b''.join(m.get('body',b'') for m in messages if m['type']=='http.response.body')
        return asyncio.run(run())
    def test_post_returns_physical_results_without_changing_receiver(self):
        status,body=self.request(json.dumps({'points':[[1000,60],[1010,40]]}).encode())
        self.assertEqual(status['status'],200)
        result=json.loads(body)
        self.assertAlmostEqual(result['segments'][0]['drift_mhz_s'],-2)
        self.assertGreater(result['segments'][0]['speed_km_s'],0)
        self.assertEqual(self.receiver.frames_received,0)
        self.assertIn((b'cache-control',b'no-store'),status['headers'])

    def test_malformed_and_excessive_requests_are_controlled_errors(self):
        for raw,expected in [(b'not json',400),(b'{"points":[[1,30],[1,20]]}',400),
                (b'['*2000+b']'*2000,400),(b' '*32769,413)]:
            with self.subTest(size=len(raw)):
                start,_=self.request(raw)
                self.assertEqual(start['status'],expected)

if __name__=='__main__':unittest.main()
