"""No root or server writes: verify the exact privileged Apache payload."""
import hashlib
import importlib.util
from pathlib import Path
from unittest import mock
import unittest

path=Path(__file__).resolve().parents[1]/'deploy/install_live_dev_apache.py'
spec=importlib.util.spec_from_file_location('live_dev_apache',path)
assert spec is not None and spec.loader is not None
apache=importlib.util.module_from_spec(spec)
spec.loader.exec_module(apache)

ORIGINAL='''<IfModule mod_ssl.c>
<VirtualHost *:443>
    ProxyPass        /live/ http://127.0.0.1:9527/
    ProxyPassReverse /live/ http://127.0.0.1:9527/
    <Location /live/>
        Require all granted
    </Location>
    # Unrelated operator rules must remain intact.
    ProxyPass /elive/api/ http://127.0.0.1:8771/elive/api/
</VirtualHost>
</IfModule>
'''

class ApachePayloadTests(unittest.TestCase):
    def test_adds_protected_dev_proxy_without_changing_existing_routes(self):
        result=apache.render_vhost(ORIGINAL)
        self.assertEqual(result.replace(apache.BLOCK,''),ORIGINAL)
        self.assertIn('http://127.0.0.1:9529/',result)
        self.assertIn('<Location /live-dev/>',result)
        self.assertNotIn('<LocationMatch "^/live-dev(?:/|$)">',result)
        for directive in ['AuthType Basic','Require user dev','AuthMerging Off',
                'RequestHeader unset Authorization','RequestHeader unset Proxy-Authorization',
                'Header unset Cache-Control','Header always set Cache-Control "private, no-store"']:
            self.assertIn(directive,result)
        self.assertNotIn('RequestHeader unset Authorization early',result)
        self.assertIn('RedirectMatch 308 ^/live-dev$ /live-dev/',result)
        # Bare redirect is vhost-level BEFORE any challenge, not inside Location.
        loc_start=result.index('<Location /live-dev/>')
        redir=result.index('RedirectMatch 308 ^/live-dev$ /live-dev/')
        self.assertLess(redir,loc_start,'Bare /live-dev redirect must run before any challenge')
        loc_end=result.index('</Location>',loc_start)
        self.assertFalse(redir > loc_start and redir < loc_end,'Redirect must not be inside Location')
        self.assertEqual(result.count('ProxyPass        /live/ '),1)

    def test_inserts_block_last_before_tls_close(self):
        result=apache.render_vhost(ORIGINAL)
        block_at=result.index(apache.BLOCK)
        close_at=result.index('</VirtualHost>',block_at)
        self.assertEqual(block_at+len(apache.BLOCK),close_at)
        # New auth is last among Location sections.
        self.assertLess(result.index('</Location>'),block_at)

    def test_rejects_non_tls_or_multiple_vhosts_or_anchor_outside(self):
        two=ORIGINAL+'<VirtualHost *:80>\n</VirtualHost>\n'
        only80=ORIGINAL.replace('*:443','*:80')
        mixed=ORIGINAL.replace('*:443','*:80 *:443')
        anchor_outside=ORIGINAL.replace('ProxyPass        /live/ http://127.0.0.1:9527/','')+'ProxyPass        /live/ http://127.0.0.1:9527/\n'
        for original in [only80,mixed,two,anchor_outside]:
            with self.subTest(original=original[-80:]):
                with self.assertRaises(ValueError):apache.render_vhost(original)

    def test_rejects_active_dynamic_if_but_allows_ifmodule(self):
        with_if=ORIGINAL.replace('</VirtualHost>','    <If "%{REQUEST_URI} =~ m#^/live-dev/#">\n        Require all granted\n    </If>\n</VirtualHost>')
        with_else=ORIGINAL.replace('</VirtualHost>','    <Else>\n        Require all granted\n    </Else>\n</VirtualHost>')
        with_commented_if=ORIGINAL.replace('</VirtualHost>','    # <If "%{REQUEST_URI} =~ m#^/x/#">\n</VirtualHost>')
        self.assertEqual(apache.render_vhost(with_commented_if).replace(apache.BLOCK,''),with_commented_if)
        for original in [with_if,with_else]:
            with self.subTest(original=original[-60:]):
                with self.assertRaises(ValueError):apache.render_vhost(original)

    def test_reapplication_is_idempotent(self):
        configured=apache.render_vhost(ORIGINAL)
        self.assertEqual(apache.render_vhost(configured),configured)

    def test_refuses_conflicting_or_partial_configuration(self):
        for original in [ORIGINAL.replace('/live/','/other/'),ORIGINAL.replace('*:443','*:80'),
                ORIGINAL+'ProxyPass /live-dev/ http://127.0.0.1:9999/\n',
                ORIGINAL+apache.BEGIN+'\n']:
            with self.subTest(original=original[-100:]):
                with self.assertRaises(ValueError):apache.render_vhost(original)

    def test_only_accepts_a_single_dev_bcrypt_verifier(self):
        valid=b'dev:$2y$12$'+b'A'*53+b'\n'
        self.assertEqual(apache.validate_verifier(valid),valid)
        for value in [b'dev:plaintext\n',valid+b'admin:hash\n',valid.replace(b'dev:',b'admin:'),
                valid.replace(b'$12$',b'$04$'),valid.replace(b'$12$',b'$99$')]:
            with self.subTest(value=value[:12]):
                with self.assertRaises(ValueError):apache.validate_verifier(value)

    def test_rollback_failure_is_not_reported_as_restored(self):
        clean=apache.rollback_message(Path('/root/backup'),[])
        self.assertIn('restored',clean)
        failed=apache.rollback_message(Path('/root/backup'),['rollback reload failed'])
        self.assertNotIn('restored',failed)
        self.assertIn('rollback incomplete',failed)
        self.assertIn('rollback reload failed',failed)

    def test_prerequisite_modules_include_alias_and_ssl(self):
        self.assertIn('alias_module',apache.REQUIRED_MODULES)
        self.assertIn('ssl_module',apache.REQUIRED_MODULES)
        for module in ['auth_basic_module','authn_file_module','authz_user_module','proxy_module','proxy_http_module','headers_module']:
            self.assertIn(module,apache.REQUIRED_MODULES)

    def test_stripped_baseline_recovers_original(self):
        configured=apache.render_vhost(ORIGINAL)
        self.assertEqual(apache.stripped_baseline(configured),ORIGINAL)
        digest=hashlib.sha256(ORIGINAL.encode()).hexdigest()
        self.assertEqual(hashlib.sha256(apache.stripped_baseline(configured).encode()).hexdigest(),digest)

    def test_anonymous_probe_requires_401_with_safe_curl_args(self):
        seen={}
        def fake_run(cmd,**kwargs):
            seen['cmd']=cmd
            seen['kwargs']=kwargs
            class Done:
                returncode=0
                stdout='401'
                stderr=''
            return Done()
        with mock.patch.object(apache.subprocess,'run',side_effect=fake_run):
            apache.probe_anonymous_401()
        cmd=seen['cmd']
        self.assertIsInstance(cmd,list)
        self.assertNotIn('shell',seen['kwargs'])
        joined=' '.join(cmd)
        self.assertIn('--resolve',joined)
        self.assertIn('--noproxy',cmd)
        self.assertEqual(cmd[cmd.index('--noproxy')+1],'*')
        self.assertIn('ovsa.njit.edu:443:127.0.0.1',joined)
        self.assertNotIn('-k',cmd)
        self.assertNotIn('--insecure',joined)
        self.assertFalse(any('Authorization' in part for part in cmd))
        with mock.patch.object(apache.subprocess,'run',return_value=type('D',(),{'returncode':0,'stdout':'200','stderr':''})()):
            with self.assertRaises(RuntimeError):apache.probe_anonymous_401()

if __name__=='__main__':unittest.main()
