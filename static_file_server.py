"""
Static File Server for QUASAR QA³

This module provides a solution for serving static files required by the PWA,
such as service worker, manifest, and icons. It runs alongside the Streamlit app.
"""

import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('static-file-server')

class StaticFileHandler(SimpleHTTPRequestHandler):
    """Custom request handler for static files"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.getcwd(), **kwargs)
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info("%s - %s", self.address_string(), format % args)
    
    def end_headers(self):
        """Add CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        
        # Special headers for service worker
        if self.path.endswith('sw.js'):
            self.send_header('Content-Type', 'application/javascript')
            self.send_header('Service-Worker-Allowed', '/')
        
        # Headers for manifest
        elif self.path.endswith('manifest.json'):
            self.send_header('Content-Type', 'application/manifest+json')
        
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.end_headers()

def run_server(port=8000, bind="0.0.0.0"):
    """Run the static file server on the specified port"""
    server_address = (bind, port)
    httpd = HTTPServer(server_address, StaticFileHandler)
    logger.info(f"Starting static file server on http://{bind}:{port}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Static file server stopped")
    finally:
        httpd.server_close()

def start_server_thread(port=8000):
    """Start the server in a background thread"""
    server_thread = threading.Thread(target=run_server, args=(port,), daemon=True)
    server_thread.start()
    logger.info(f"Static file server thread started on port {port}")
    return server_thread

if __name__ == "__main__":
    run_server()