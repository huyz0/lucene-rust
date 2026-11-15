"""benchhttp against a local server: content-length, chunked, Connection: close, large bodies, errors, and the per-request floor."""
import sys, threading, json, time
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../opensearch-plugin/e2e'))
import benchhttp as bh
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    wbufsize=65536
    def handle_one_request(self):
        BaseHTTPRequestHandler.handle_one_request(self); self.wfile.flush()
    protocol_version="HTTP/1.1"
    def do_POST(self):
        n=int(self.headers.get("content-length",0)); b=self.rfile.read(n)
        if self.path=="/chunk":
            self.send_response(200); self.send_header("Transfer-Encoding","chunked"); self.end_headers()
            for part in (b'{"a":', b'1}'): self.wfile.write(b"%x\r\n%s\r\n"%(len(part),part))
            self.wfile.write(b"0\r\n\r\n"); return
        if self.path=="/err":
            d=b'{"x":1}'; self.send_response(500); self.send_header("Content-Length",str(len(d))); self.end_headers(); self.wfile.write(d); return
        if self.path=="/close":
            d=b'{"c":1}'; self.send_response(200); self.send_header("Content-Length",str(len(d))); self.send_header("Connection","close"); self.end_headers(); self.wfile.write(d); self.close_connection=True; return
        d=json.dumps({"len":len(b)}).encode(); self.send_response(200); self.send_header("Content-Length",str(len(d))); self.end_headers(); self.wfile.write(d)
    do_GET=do_POST
    def log_message(self,*a): pass
s=ThreadingHTTPServer(("127.0.0.1",0),H); port=s.server_address[1]
threading.Thread(target=s.serve_forever,daemon=True).start()
base=f"http://localhost:{port}"
assert bh.req(base,"POST","/x",{"a":1})=={"len":8}
assert bh.req(base,"POST","/chunk",{})=={"a":1}
assert bh.req(base,"POST","/close",{})=={"c":1}
assert bh.req(base,"POST","/x",b"x"*5_000_000)=={"len":5_000_000}
try: bh.req(base,"POST","/err",{}); assert 0
except RuntimeError as e: assert "500" in str(e)
assert bh.req(base,"GET","/x")=={"len":0}
ms=bh.measure(lambda: bh.req(base,"POST","/x",{"a":1}),500,50,0.2)
print(bh.summarize(ms))
assert bh.summarize(ms)["median_ms"] < 2, "client floor"
