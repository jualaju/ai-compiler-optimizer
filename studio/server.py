"""Loopback-only, standard-library HTTP adapter for Compiler Studio."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import traceback
from urllib.parse import unquote, urlsplit

import torch
from .serialization import json_safe
from .service import StudioService

STATIC = Path(__file__).resolve().parent / 'static'


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address=('127.0.0.1', 8765), service=None):
        self.service = service or StudioService()
        super().__init__(address, StudioHandler)


class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, message, *args):
        # No request payloads or tensor contents in logs.
        super().log_message(message, *args)

    def send_body(self, status, body, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        try:
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Closing a browser tab cancels delivery, not the compiler's completed work.
            return

    def json(self, status, data):
        self.send_body(status, json.dumps(json_safe(data), allow_nan=False).encode(), 'application/json; charset=utf-8')

    def check_local(self):
        expected = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        host = self.headers.get('Host', '')
        if host not in expected:
            self.json(403, {'error': 'Use the local URL printed by Studio'})
            return False
        origin = self.headers.get('Origin')
        if origin is not None and origin not in {'http://' + value for value in expected}:
            self.json(403, {'error': 'Studio accepts same-origin local requests only'})
            return False
        return True

    def do_GET(self):
        if not self.check_local(): return
        path = urlsplit(self.path).path
        try:
            if path == '/api/catalog': return self.json(200, self.server.service.catalog())
            if path == '/api/support': return self.json(200, self.server.service.support())
            if path == '/api/runs': return self.json(200, self.server.service.history())
            if path.startswith('/api/runs/'):
                with self.server.service.lock:
                    run = self.server.service.get(path.removeprefix('/api/runs/'))
                    return self.json(200, self.server.service.snapshot(run))
            if path.startswith('/api/'): return self.json(404, {'error': 'Unknown API endpoint'})
            file = STATIC / ('index.html' if path == '/' else unquote(path).lstrip('/'))
            if not file.resolve().is_relative_to(STATIC) or not file.is_file():
                return self.json(404, {'error': 'Asset not found'})
            content_type = mimetypes.guess_type(str(file))[0] or 'application/octet-stream'
            return self.send_body(200, file.read_bytes(), content_type)
        except KeyError as exc: self.json(404, {'error': str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self.json(500, {'error': str(exc)})

    def do_POST(self):
        if not self.check_local(): return
        try:
            if self.headers.get_content_type() != 'application/json':
                return self.json(415, {'error': 'Send an application/json request'})
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 2 * 1024 * 1024:
                return self.json(413, {'error': 'Request body must be 1 byte–2 MiB'})
            def reject_constant(value): raise ValueError(f'Non-finite JSON literal: {value}')
            data = json.loads(self.rfile.read(length), parse_constant=reject_constant)
            if not isinstance(data, dict): raise ValueError('JSON body must be an object')
            path = urlsplit(self.path).path
            if path == '/api/runs': return self.json(201, self.server.service.create(data))
            parts = path.strip('/').split('/')
            if len(parts) == 4 and parts[:2] == ['api', 'runs']:
                return self.json(200, self.server.service.action(parts[2], parts[3], data))
            self.json(404, {'error': 'Unknown API endpoint'})
        except KeyError as exc: self.json(404, {'error': str(exc)})
        except (ValueError, RuntimeError, OSError, TypeError) as exc:
            self.json(400, {'error': str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self.json(500, {'error': str(exc)})


def main(argv=None):
    parser = argparse.ArgumentParser(description='Local Compiler Studio — real CPU compiler and IR graph explorer')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535: parser.error('port must be between 0 and 65535')
    torch.set_num_threads(1)
    try:
        server = StudioServer(('127.0.0.1', args.port))
    except OSError as exc:
        parser.exit(1, f'Cannot start Studio: {exc}. Try --port 8766.\n')
    print(f'Compiler Studio: http://127.0.0.1:{server.server_port}', flush=True)
    print('Local CPU execution • session history in memory • Ctrl+C to stop', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__': main()
