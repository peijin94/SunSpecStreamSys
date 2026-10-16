#!/usr/bin/env python3
"""One-time, guarded Apache live-dev install. Run explicitly with sudo.

No plaintext credential is accepted or stored here. The caller supplies a private
htpasswd bcrypt verifier produced separately; this script prints metadata only.
The existing /live/ and unrelated vhost rules are preserved byte-for-byte.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile

BEGIN='    # BEGIN SUNSPEC LIVE-DEV BASIC AUTH'
END='    # END SUNSPEC LIVE-DEV BASIC AUTH'
BLOCK='''    # BEGIN SUNSPEC LIVE-DEV BASIC AUTH
    RedirectMatch 308 ^/live-dev$ /live-dev/
    ProxyPass        /live-dev/ http://127.0.0.1:9529/
    ProxyPassReverse /live-dev/ http://127.0.0.1:9529/
    <Location /live-dev/>
        AuthType Basic
        AuthName "OVRO-LWA live-dev"
        AuthBasicProvider file
        AuthUserFile /etc/apache2/lwa-live-dev.htpasswd
        AuthMerging Off
        Require user dev
        # Default/fixup timing: authenticate first, then remove before proxying.
        RequestHeader unset Authorization
        RequestHeader unset Proxy-Authorization
        Header unset Cache-Control
        Header always set Cache-Control "private, no-store"
        ProxyPreserveHost On
    </Location>
    # END SUNSPEC LIVE-DEV BASIC AUTH
'''

REQUIRED_MODULES={'auth_basic_module','authn_file_module','authz_user_module','proxy_module','proxy_http_module','headers_module','alias_module','ssl_module'}

PROBE_PATHS=['/live-dev/','/live-dev/spectrum-range','/live-dev/radio-images/']


def _find_single_tls_vhost(original):
    opens=[]
    closes=[]
    offset=0
    for line in original.splitlines(keepends=True):
        stripped=line.lstrip()
        if not stripped.startswith('#'):
            open_match=re.match(r'[ \t]*<VirtualHost\s+([^>]+)>',line,re.I)
            if open_match:
                end=line.find('>')
                opens.append((offset,offset+end+1,open_match.group(1)))
            close_match=re.match(r'[ \t]*</VirtualHost\s*>',line,re.I)
            if close_match:
                start=line.find('<')
                closes.append((offset+start,offset+len(line)))
        offset+=len(line)
    if len(opens)!=1 or len(closes)!=1:
        raise ValueError('Expected exactly one active VirtualHost')
    open_start,open_end,args=opens[0]
    close_start,close_end=closes[0]
    if not open_start<close_start:
        raise ValueError('Malformed VirtualHost ordering')
    tokens=args.split()
    if not tokens or any(not token.endswith(':443') for token in tokens):
        raise ValueError('Refuse a vhost without an exclusive TLS :443 block')
    return original[open_end:close_start],close_start


def _reject_dynamic_sections(inner):
    for line in inner.splitlines():
        stripped=line.lstrip()
        if stripped.startswith('#'):
            continue
        if re.match(r'<\s*/?\s*(If|ElseIf|Else)\b',stripped,re.I):
            raise ValueError('Refuse dynamic <If>/<Else> sections that could merge after Location auth')


def stripped_baseline(configured):
    if configured.count(BLOCK)!=1:
        raise ValueError('Expected exactly one live-dev block')
    return configured.replace(BLOCK,'',1)


def probe_anonymous_401(timeout=10):
    base='https://ovsa.njit.edu'
    for path in PROBE_PATHS:
        url=base+path
        cmd=['curl','--noproxy','*','--silent','--output','/dev/null','--write-out','%{http_code}','--max-time',str(timeout),'--resolve','ovsa.njit.edu:443:127.0.0.1',url]
        completed=subprocess.run(cmd,capture_output=True,text=True)
        if completed.returncode!=0:
            detail=(completed.stderr or '').strip()[:200]
            raise RuntimeError('Anonymous probe failed for '+path+': '+detail)
        code=(completed.stdout or '').strip()
        if code!='401':
            raise RuntimeError('Expected 401 for anonymous '+path+', got '+code)


def render_vhost(original):
    inner,close_start=_find_single_tls_vhost(original)
    _reject_dynamic_sections(inner)
    if BEGIN in original or END in original:
        if original.count(BEGIN)!=1 or original.count(END)!=1 or BLOCK not in original:
            raise ValueError('Conflicting or partial live-dev configuration')
        if BLOCK not in inner:
            raise ValueError('Conflicting or partial live-dev configuration')
        if original[close_start-len(BLOCK):close_start]!=BLOCK:
            raise ValueError('Conflicting or partial live-dev configuration')
        return original
    if re.search(r'^\s*ProxyPass\s+\S*/live-dev', original, re.M):
        raise ValueError('An unrelated live-dev proxy already exists')
    anchors=list(re.finditer(r'^[ \t]*ProxyPass[ \t]+/live/[ \t]+http://127\.0\.0\.1:9527/[ \t]*$',inner,re.M))
    if len(anchors)!=1:
        raise ValueError('Expected exactly one unchanged public /live/ proxy')
    return original[:close_start]+BLOCK+original[close_start:]


def validate_verifier(value):
    match=re.fullmatch(rb'dev:\$2[aby]\$(\d{2})\$[./A-Za-z0-9]{53}\n',value)
    if not match or not 10<=int(match.group(1))<=16:
        raise ValueError('Expected one dev bcrypt verifier at cost 10–16, not plaintext')
    return value


def atomic_write(path, value, mode, uid, gid):
    descriptor, temporary=tempfile.mkstemp(prefix='.lwa-live-dev-',dir=str(path.parent))
    try:
        with os.fdopen(descriptor,'wb') as stream:
            stream.write(value);stream.flush();os.fsync(stream.fileno())
        os.chmod(temporary,mode);os.chown(temporary,uid,gid)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def rollback_message(backup, notes):
    if notes:
        return 'Install failed; rollback incomplete. Inspect '+str(backup)+'. Rollback issues: '+'; '.join(notes)
    return 'Install failed; prior routing/auth files restored and Apache reloaded. Inspect '+str(backup)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials-file',required=True)
    parser.add_argument('--expected-sha256',required=True)
    args=parser.parse_args()
    if os.geteuid()!=0:raise SystemExit('Run this reviewed installer explicitly with sudo')
    if not re.fullmatch(r'[a-f0-9]{64}',args.expected_sha256):raise SystemExit('Invalid config fingerprint')
    import grp
    import pwd
    credential_source=Path(args.credentials_file)
    if credential_source.is_symlink():raise SystemExit('Verifier source must not be a symlink')
    info=credential_source.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size>256 or info.st_uid not in (0,pwd.getpwnam('peijin').pw_uid) or info.st_mode & 0o077:
        raise SystemExit('Verifier source must be a private, owner-only regular file')
    verifier=validate_verifier(credential_source.read_bytes())
    target=Path('/etc/apache2/sites-enabled/000-default-le-ssl.conf').resolve(strict=True)
    if target.parent!=Path('/etc/apache2/sites-available') or target.name!='000-default-le-ssl.conf':
        raise SystemExit('Unexpected TLS vhost target')
    original=target.read_bytes();proposed=render_vhost(original.decode()).encode()
    password_file=Path('/etc/apache2/lwa-live-dev.htpasswd')
    prior_password=password_file.read_bytes() if password_file.exists() else None
    if hashlib.sha256(original).hexdigest()!=args.expected_sha256:
        if proposed==original and prior_password==verifier:
            baseline=stripped_baseline(original.decode()).encode()
            if hashlib.sha256(baseline).hexdigest()!=args.expected_sha256:
                raise SystemExit('Vhost drift detected; refuse overwrite and regenerate the command')
            print(json.dumps({'status':'already_configured','username':'dev','dev_port':9529}));return
        raise SystemExit('Vhost changed since preparation; refuse overwrite and regenerate the command')
    backup=Path('/root')/('lwa-live-dev-apache-backup-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    backup.mkdir(mode=0o700)
    shutil.copy2(target,backup/'vhost.conf')
    if prior_password is not None:
        (backup/'prior.htpasswd').write_bytes(prior_password);os.chmod(backup/'prior.htpasswd',0o600)
    old=target.stat();old_password=password_file.stat() if password_file.exists() else None
    # Check prerequisites without enabling modules or changing unrelated state.
    enabled=subprocess.run(['apache2ctl','-M'],capture_output=True,text=True)
    (backup/'module-check.log').write_text(enabled.stdout+enabled.stderr)
    missing=REQUIRED_MODULES-set(re.findall(r'\b[a-z_]+_module\b',enabled.stdout))
    if enabled.returncode or missing:
        raise SystemExit('Required Apache modules unavailable: '+', '.join(sorted(missing))+'; no routing/auth files changed')
    try:
        atomic_write(password_file,verifier,0o640,0,grp.getgrnam('www-data').gr_gid)
        atomic_write(target,proposed,stat.S_IMODE(old.st_mode),old.st_uid,old.st_gid)
        checked=subprocess.run(['apache2ctl','configtest'],capture_output=True,text=True)
        (backup/'configtest.log').write_text(checked.stdout+checked.stderr)
        if checked.returncode:raise RuntimeError('Apache configtest failed')
        reloaded=subprocess.run(['systemctl','reload','apache2'],capture_output=True,text=True)
        (backup/'reload.log').write_text(reloaded.stdout+reloaded.stderr)
        if reloaded.returncode:raise RuntimeError('Apache reload failed')
        probe_anonymous_401()
        if target.read_bytes()!=proposed or password_file.read_bytes()!=verifier:
            raise RuntimeError('Post-write verification failed')
        manifest={'status':'installed','username':'dev','dev_url':'https://ovsa.njit.edu/live-dev/',
              'dev_port':9529,'public_port':9527,'vhost_sha256':hashlib.sha256(proposed).hexdigest(),'backup':str(backup)}
        (backup/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print(json.dumps(manifest))
    except Exception as error:
        notes=[]
        try:
            atomic_write(target,original,stat.S_IMODE(old.st_mode),old.st_uid,old.st_gid)
        except Exception as restore_error:
            notes.append('vhost restore failed: '+str(restore_error)[:200])
        try:
            if prior_password is None:
                try:
                    password_file.unlink()
                except FileNotFoundError:
                    pass
            else:
                assert old_password is not None
                atomic_write(password_file,prior_password,stat.S_IMODE(old_password.st_mode),old_password.st_uid,old_password.st_gid)
        except Exception as password_error:
            notes.append('password restore failed: '+str(password_error)[:200])
        rollback_check=subprocess.run(['apache2ctl','configtest'],capture_output=True,text=True)
        try:
            (backup/'rollback-configtest.log').write_text(rollback_check.stdout+rollback_check.stderr)
        except Exception:
            pass
        if rollback_check.returncode:
            notes.append('rollback configtest failed: '+(rollback_check.stderr or rollback_check.stdout or '').strip()[:200])
        rollback_reload=subprocess.run(['systemctl','reload','apache2'],capture_output=True,text=True)
        try:
            (backup/'rollback-reload.log').write_text(rollback_reload.stdout+rollback_reload.stderr)
        except Exception:
            pass
        if rollback_reload.returncode:
            notes.append('rollback reload failed: '+(rollback_reload.stderr or rollback_reload.stdout or '').strip()[:200])
        raise SystemExit(rollback_message(backup,notes)) from error

if __name__=='__main__':main()
