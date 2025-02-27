"""
Static File Server for QUASAR QA³

This module provides a solution for serving static files required by the PWA,
such as service worker, manifest, and icons. It runs alongside the Streamlit app.
"""

import os
import threading
import queue
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

def run_server(port=8765, bind="0.0.0.0"):
    """Run the static file server on the specified port"""
    # Try to find an available port, starting with the specified one
    max_attempts = 5
    httpd = None
    actual_port = None
    
    for attempt in range(max_attempts):
        try:
            server_address = (bind, port)
            httpd = HTTPServer(server_address, StaticFileHandler)
            actual_port = port
            logger.info(f"Starting static file server on http://{bind}:{port}")
            break
        except OSError as e:
            error_code = getattr(e, 'errno', None)
            if error_code in (98, 10048):  # Address already in use (Linux/Windows)
                logger.warning(f"Port {port} is already in use, trying port {port + 1}")
                port += 1
                if attempt == max_attempts - 1:
                    logger.error(f"Could not find an available port after {max_attempts} attempts")
                    raise
            else:
                logger.error(f"Error starting server: {str(e)}")
                raise
    
    if httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Static file server stopped")
        finally:
            httpd.server_close()
    else:
        logger.error("Failed to initialize HTTP server")
        return None
    
    return actual_port

def start_server_thread(port=8765):
    """Start the server in a background thread"""
    # Use a queue to get the actual port used by the server
    port_queue = queue.Queue()
    
    def run_with_queue():
        try:
            actual_port = run_server(port)
            port_queue.put(actual_port)
        except Exception as e:
            logger.error(f"Error starting static file server: {e}")
            port_queue.put(None)
    
    server_thread = threading.Thread(target=run_with_queue, daemon=True)
    server_thread.start()
    
    # Wait for a short time to see if the server starts successfully
    try:
        actual_port = port_queue.get(timeout=3)
        if actual_port:
            logger.info(f"Static file server thread started successfully on port {actual_port}")
            return server_thread, actual_port
        else:
            logger.error("Failed to start static file server")
            return None, None
    except queue.Empty:
        logger.warning(f"Static file server thread started but no port confirmation received")
        return server_thread, None

if __name__ == "__main__":
    run_server()