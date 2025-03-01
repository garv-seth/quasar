"""
PWA Integration Module for QA³ (Quantum-Accelerated AI Agent)

This module provides Progressive Web App (PWA) capabilities for the QA³ agent,
enabling offline functionality, installation as a standalone app, and
enhanced user experience.
"""

import os
import json
import logging
import threading
import http.server
import socketserver
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Try to import Streamlit
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logging.warning("Streamlit not available. PWA UI integration will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pwa-integration")

# Global variables
static_server_thread = None
static_server_port = None

class StaticFileHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for static files with proper MIME types"""
    
    def __init__(self, *args, **kwargs):
        self.static_dir = kwargs.pop('static_dir', './static')
        super().__init__(*args, **kwargs)
    
    def translate_path(self, path):
        """Translate URL path to file system path in static directory"""
        # Remove query parameters
        path = path.split('?', 1)[0]
        path = path.split('#', 1)[0]
        
        # Handle root path
        if path == '/':
            return os.path.join(self.static_dir, 'index.html')
        
        # Map URL path to static directory
        return os.path.join(self.static_dir, path.lstrip('/'))
    
    def log_message(self, format, *args):
        """Suppress logs for static file server"""
        pass
    
    def end_headers(self):
        """Add CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_static_server(port=8765) -> Tuple[threading.Thread, int]:
    """Start the static file server for PWA files if not already running
    
    Args:
        port: Port to start the server on
        
    Returns:
        Tuple of (server_thread, actual_port)
    """
    global static_server_thread, static_server_port
    
    # If server is already running, return the existing thread and port
    if static_server_thread is not None and static_server_thread.is_alive():
        return static_server_thread, static_server_port
    
    # Ensure static directory exists
    static_dir = './static'
    os.makedirs(static_dir, exist_ok=True)
    
    # Create handler with static directory
    handler = lambda *args, **kwargs: StaticFileHandler(*args, static_dir=static_dir, **kwargs)
    
    # Try to start server on requested port
    for attempt_port in range(port, port + 10):
        try:
            httpd = socketserver.TCPServer(("0.0.0.0", attempt_port), handler)
            break
        except OSError:
            # Port in use, try next port
            if attempt_port == port + 9:
                logger.error(f"Failed to find an available port after 10 attempts")
                return None, None
            continue
    
    actual_port = httpd.server_address[1]
    
    # Start server in a thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    # Store thread and port in global variables
    static_server_thread = server_thread
    static_server_port = actual_port
    
    logger.info(f"Static file server started on port {actual_port}")
    return server_thread, actual_port

def generate_manifest():
    """Generate the PWA manifest file"""
    manifest = {
        "name": "QA³: Quantum-Accelerated AI Agent",
        "short_name": "QA³",
        "description": "Quantum-powered AI assistant with deep search and web browsing capabilities",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#121212",
        "theme_color": "#7e57c2",
        "icons": [
            {
                "src": "./icon-192.png",
                "sizes": "192x192",
                "type": "image/png",
                "purpose": "any maskable"
            },
            {
                "src": "./icon-512.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "any maskable"
            }
        ]
    }
    
    # Write manifest to static directory
    static_dir = './static'
    os.makedirs(static_dir, exist_ok=True)
    
    with open(os.path.join(static_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Also write to root for direct access
    with open('manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info("Generated PWA manifest file")

def generate_service_worker():
    """Generate the service worker for offline functionality"""
    sw_content = """
// QA³ (Quantum-Accelerated AI Agent) Service Worker
const CACHE_NAME = 'qa3-cache-v1';
const OFFLINE_URL = 'offline.html';

// Files to cache
const urlsToCache = [
  '/',
  '/icon-192.png',
  '/icon-512.png',
  '/manifest.json',
  '/offline.html',
  '/pwa.js'
];

// Install service worker
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('Service worker installed. Caching assets...');
        return cache.addAll(urlsToCache);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate service worker
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((name) => {
          if (name !== CACHE_NAME) {
            console.log('Clearing old cache:', name);
            return caches.delete(name);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch event handler
self.addEventListener('fetch', (event) => {
  // For API requests, try network first, then fall back to cache
  if (event.request.url.includes('/api/')) {
    event.respondWith(
      fetch(event.request)
        .catch(() => {
          return caches.match(event.request);
        })
    );
    return;
  }
  
  // For other requests, try cache first, then fall back to network
  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        // If found in cache, return cached response
        if (response) {
          return response;
        }
        
        // If not in cache, fetch from network
        return fetch(event.request)
          .then((response) => {
            // Don't cache if not valid response
            if (!response || response.status !== 200 || response.type !== 'basic') {
              return response;
            }
            
            // Clone response to store in cache
            const responseToCache = response.clone();
            
            caches.open(CACHE_NAME)
              .then((cache) => {
                cache.put(event.request, responseToCache);
              });
            
            return response;
          })
          .catch(() => {
            // If both cache and network fail, show offline page
            if (event.request.mode === 'navigate') {
              return caches.match(OFFLINE_URL);
            }
          });
      })
  );
});

// Handle push notifications
self.addEventListener('push', (event) => {
  const data = event.data.json();
  const options = {
    body: data.body,
    icon: '/icon-192.png',
    badge: '/icon-192.png',
    data: {
      url: data.url || '/'
    }
  };
  
  event.waitUntil(
    self.registration.showNotification(data.title, options)
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  event.waitUntil(
    clients.openWindow(event.notification.data.url)
  );
});
"""
    
    # Write service worker to static directory
    static_dir = './static'
    os.makedirs(static_dir, exist_ok=True)
    
    with open(os.path.join(static_dir, 'sw.js'), 'w') as f:
        f.write(sw_content)
    
    # Also write to root for direct access
    with open('sw.js', 'w') as f:
        f.write(sw_content)
    
    logger.info("Generated service worker file")

def generate_offline_html():
    """Generate the offline fallback page"""
    offline_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Offline - QA³</title>
    <link rel="manifest" href="./manifest.json">
    <link rel="icon" href="./icon-192.png">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            text-align: center;
            background-color: #121212;
            color: #ffffff;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        .container {
            max-width: 600px;
            margin: 0 auto;
        }
        
        h1 {
            color: #7e57c2;
        }
        
        .logo {
            max-width: 150px;
            margin-bottom: 20px;
        }
        
        .button {
            background-color: #7e57c2;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            margin-top: 20px;
            cursor: pointer;
            font-size: 16px;
            text-decoration: none;
            display: inline-block;
        }
        
        .quantum-animation {
            position: relative;
            width: 100px;
            height: 100px;
            margin: 20px auto;
        }
        
        .quantum-dot {
            position: absolute;
            width: 20px;
            height: 20px;
            background-color: #7e57c2;
            border-radius: 50%;
            top: 40px;
            left: 40px;
            animation: quantum-spin 3s infinite linear;
        }
        
        @keyframes quantum-spin {
            0% { transform: translateX(0) translateY(0); }
            25% { transform: translateX(30px) translateY(-30px); opacity: 0.7; }
            50% { transform: translateX(0) translateY(0); opacity: 1; }
            75% { transform: translateX(-30px) translateY(30px); opacity: 0.7; }
            100% { transform: translateX(0) translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <img src="./icon-192.png" alt="QA³ Logo" class="logo">
        <h1>You're currently offline</h1>
        <p>The QA³ quantum agent requires an internet connection to process tasks and provide up-to-date information.</p>
        
        <div class="quantum-animation">
            <div class="quantum-dot"></div>
        </div>
        
        <p>Some previously performed searches and tasks may be available in your task history.</p>
        
        <a href="/" class="button">Try Again</a>
    </div>
    
    <script>
        // Check for connectivity and reload if back online
        window.addEventListener('online', () => {
            window.location.reload();
        });
    </script>
</body>
</html>
"""
    
    # Write offline page to static directory
    static_dir = './static'
    os.makedirs(static_dir, exist_ok=True)
    
    with open(os.path.join(static_dir, 'offline.html'), 'w') as f:
        f.write(offline_html)
    
    # Also write to root for direct access
    with open('offline.html', 'w') as f:
        f.write(offline_html)
    
    logger.info("Generated offline HTML page")

def generate_pwa_js():
    """Generate the PWA JavaScript helper"""
    pwa_js = """
/**
 * QA³ (Quantum-Accelerated AI Agent) PWA Helper
 * 
 * This script handles PWA integration with Streamlit, including:
 * - Service worker registration
 * - PWA installation prompts
 * - Offline database for task and search history
 * - Notification handling
 */

// PWA status
let pwaStatus = {
    isInstalled: false,
    canInstall: false,
    supportsNotifications: 'Notification' in window,
    notificationsEnabled: false,
    deferredPrompt: null,
    isStandalone: window.matchMedia('(display-mode: standalone)').matches
};

// Check if running in PWA mode
if (window.matchMedia('(display-mode: standalone)').matches || 
    window.navigator.standalone === true) {
    pwaStatus.isInstalled = true;
    console.log('Running in PWA mode');
}

// Register service worker
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('./sw.js')
            .then((registration) => {
                console.log('PWA: Service Worker registered with scope:', registration.scope);
                
                // Check notification permissions
                if (pwaStatus.supportsNotifications) {
                    pwaStatus.notificationsEnabled = Notification.permission === 'granted';
                }
                
                // Send status to Streamlit
                sendPwaStatusToStreamlit();
            })
            .catch((error) => {
                console.error('PWA: Service Worker registration failed:', error);
            });
    });
}

// Handle installation prompt
window.addEventListener('beforeinstallprompt', (event) => {
    // Prevent default browser install prompt
    event.preventDefault();
    
    // Store event for later use
    pwaStatus.deferredPrompt = event;
    pwaStatus.canInstall = true;
    
    // Send status to Streamlit
    sendPwaStatusToStreamlit();
});

// Check if app was installed
window.addEventListener('appinstalled', () => {
    pwaStatus.isInstalled = true;
    pwaStatus.canInstall = false;
    pwaStatus.deferredPrompt = null;
    
    console.log('PWA: App was installed');
    
    // Send status to Streamlit
    sendPwaStatusToStreamlit();
});

// Initialize IndexedDB for offline storage
let db;
initializeDatabase();

function initializeDatabase() {
    const request = indexedDB.open('QA3Database', 1);
    
    request.onerror = (event) => {
        console.error('PWA: Database error:', event.target.error);
    };
    
    request.onupgradeneeded = (event) => {
        const db = event.target.result;
        
        // Create task history store
        if (!db.objectStoreNames.contains('tasks')) {
            const taskStore = db.createObjectStore('tasks', { keyPath: 'id' });
            taskStore.createIndex('timestamp', 'timestamp', { unique: false });
        }
        
        // Create search history store
        if (!db.objectStoreNames.contains('searches')) {
            const searchStore = db.createObjectStore('searches', { keyPath: 'id' });
            searchStore.createIndex('timestamp', 'timestamp', { unique: false });
        }
        
        // Create settings store
        if (!db.objectStoreNames.contains('settings')) {
            const settingsStore = db.createObjectStore('settings', { keyPath: 'key' });
        }
    };
    
    request.onsuccess = (event) => {
        db = event.target.result;
        console.log('PWA: Database initialized');
    };
}

// Store task in offline database
async function storeTask(task) {
    if (!db) return false;
    
    return new Promise((resolve, reject) => {
        const transaction = db.transaction(['tasks'], 'readwrite');
        const taskStore = transaction.objectStore('tasks');
        
        // Add task to store
        const request = taskStore.put(task);
        
        request.onsuccess = () => resolve(true);
        request.onerror = () => reject(request.error);
    });
}

// Store generic data
async function storeData(key, data) {
    if (!db) return false;
    
    return new Promise((resolve, reject) => {
        const transaction = db.transaction(['settings'], 'readwrite');
        const settingsStore = transaction.objectStore('settings');
        
        // Add data to store
        const request = settingsStore.put({ key, value: data });
        
        request.onsuccess = () => resolve(true);
        request.onerror = () => reject(request.error);
    });
}

// Get stored data
async function getData(key) {
    if (!db) return null;
    
    return new Promise((resolve, reject) => {
        const transaction = db.transaction(['settings'], 'readonly');
        const settingsStore = transaction.objectStore('settings');
        
        // Get data from store
        const request = settingsStore.get(key);
        
        request.onsuccess = () => resolve(request.result ? request.result.value : null);
        request.onerror = () => reject(request.error);
    });
}

// Install the PWA
async function installPwa() {
    if (!pwaStatus.deferredPrompt) {
        console.log('PWA: Cannot install, no installation prompt available');
        return false;
    }
    
    // Show the installation prompt
    pwaStatus.deferredPrompt.prompt();
    
    // Wait for user response
    const choiceResult = await pwaStatus.deferredPrompt.userChoice;
    
    // Update status based on user choice
    if (choiceResult.outcome === 'accepted') {
        console.log('PWA: User accepted installation');
        pwaStatus.isInstalled = true;
    } else {
        console.log('PWA: User declined installation');
    }
    
    // Clear the prompt
    pwaStatus.deferredPrompt = null;
    pwaStatus.canInstall = false;
    
    // Send updated status to Streamlit
    sendPwaStatusToStreamlit();
    
    return choiceResult.outcome === 'accepted';
}

// Request notification permission
async function requestNotificationPermission() {
    if (!pwaStatus.supportsNotifications) {
        console.log('PWA: Notifications not supported');
        return false;
    }
    
    try {
        const permission = await Notification.requestPermission();
        pwaStatus.notificationsEnabled = permission === 'granted';
        
        // Send updated status to Streamlit
        sendPwaStatusToStreamlit();
        
        return pwaStatus.notificationsEnabled;
    } catch (error) {
        console.error('PWA: Error requesting notification permission:', error);
        return false;
    }
}

// Send a test notification
function sendTestNotification() {
    if (!pwaStatus.notificationsEnabled) {
        console.log('PWA: Notifications not enabled');
        return false;
    }
    
    try {
        // Check if service worker is active
        if (navigator.serviceWorker.controller) {
            // Send notification through service worker
            navigator.serviceWorker.controller.postMessage({
                type: 'SEND_NOTIFICATION',
                payload: {
                    title: 'QA³ Test Notification',
                    body: 'This is a test notification from the Quantum-Accelerated AI Agent',
                    url: window.location.href
                }
            });
        } else {
            // Fall back to regular notification
            const notification = new Notification('QA³ Test Notification', {
                body: 'This is a test notification from the Quantum-Accelerated AI Agent',
                icon: './icon-192.png'
            });
            
            notification.onclick = () => {
                window.focus();
                notification.close();
            };
        }
        
        return true;
    } catch (error) {
        console.error('PWA: Error sending test notification:', error);
        return false;
    }
}

// Send PWA status to Streamlit
function sendPwaStatusToStreamlit() {
    if (window.Streamlit) {
        window.Streamlit.setComponentValue({
            type: 'pwa_status',
            data: pwaStatus
        });
    }
}

// Expose functions to Streamlit
window.qa3Pwa = {
    installPwa,
    requestNotificationPermission,
    sendTestNotification,
    storeTask,
    storeData,
    getData,
    getPwaStatus: () => pwaStatus
};

// Send status when Streamlit is ready
if (window.Streamlit) {
    window.Streamlit.onReady(() => {
        sendPwaStatusToStreamlit();
    });
}
"""
    
    # Write PWA JS to static directory
    static_dir = './static'
    os.makedirs(static_dir, exist_ok=True)
    
    with open(os.path.join(static_dir, 'pwa.js'), 'w') as f:
        f.write(pwa_js)
    
    # Also write to root for direct access
    with open('pwa.js', 'w') as f:
        f.write(pwa_js)
    
    logger.info("Generated PWA JavaScript helper")

def create_pwa_files():
    """Create necessary PWA files in the static directory"""
    # Generate all required PWA files
    generate_manifest()
    generate_service_worker()
    generate_offline_html()
    generate_pwa_js()
    
    logger.info("Created all PWA files")

def initialize_pwa():
    """Initialize PWA components for Streamlit"""
    if not STREAMLIT_AVAILABLE:
        logger.warning("Streamlit not available, skipping PWA UI initialization")
        return
    
    # Create PWA files
    create_pwa_files()
    
    # Start static file server
    thread, port = start_static_server()
    
    # Add JavaScript to Streamlit
    pwa_script = f"""
    <script src="http://localhost:{port}/pwa.js"></script>
    <link rel="manifest" href="http://localhost:{port}/manifest.json">
    <script>
        // Register PWA components
        if ('serviceWorker' in navigator) {{
            window.addEventListener('load', () => {{
                navigator.serviceWorker.register('http://localhost:{port}/sw.js')
                    .then(registration => console.log('Service worker registered'))
                    .catch(error => console.error('Service worker registration failed:', error));
            }});
        }}
    </script>
    """
    
    # Inject script into Streamlit
    st.components.v1.html(pwa_script, height=0)
    
    logger.info("PWA integration initialized for Streamlit")

def get_pwa_status():
    """Get the current PWA status"""
    if not STREAMLIT_AVAILABLE:
        return {
            "available": False,
            "error": "Streamlit not available"
        }
    
    # Get PWA status from session state if available
    if "pwa_status" in st.session_state:
        return st.session_state.pwa_status
    
    # Default status
    return {
        "available": True,
        "isInstalled": False,
        "canInstall": False,
        "supportsNotifications": False,
        "notificationsEnabled": False
    }

def check_pwa_mode():
    """Check if the app is running in PWA mode"""
    status = get_pwa_status()
    return status.get("isInstalled", False)

def display_pwa_install_button():
    """Display a button to install the PWA"""
    if not STREAMLIT_AVAILABLE:
        return
    
    status = get_pwa_status()
    
    if status.get("canInstall", False) and not status.get("isInstalled", False):
        if st.button("📱 Install as App"):
            # Call JavaScript to trigger installation
            js_code = """
            <script>
                if (window.qa3Pwa && window.qa3Pwa.installPwa) {
                    window.qa3Pwa.installPwa();
                }
            </script>
            """
            st.components.v1.html(js_code, height=0)
            st.success("Installation prompt shown. Please follow the instructions in your browser.")

def display_notification_button():
    """Display a button to enable/disable notifications"""
    if not STREAMLIT_AVAILABLE:
        return
    
    status = get_pwa_status()
    
    if status.get("supportsNotifications", False):
        if status.get("notificationsEnabled", False):
            if st.button("🔔 Send Test Notification"):
                # Call JavaScript to send test notification
                js_code = """
                <script>
                    if (window.qa3Pwa && window.qa3Pwa.sendTestNotification) {
                        window.qa3Pwa.sendTestNotification();
                    }
                </script>
                """
                st.components.v1.html(js_code, height=0)
        else:
            if st.button("🔔 Enable Notifications"):
                # Call JavaScript to request notification permission
                js_code = """
                <script>
                    if (window.qa3Pwa && window.qa3Pwa.requestNotificationPermission) {
                        window.qa3Pwa.requestNotificationPermission();
                    }
                </script>
                """
                st.components.v1.html(js_code, height=0)

def add_offline_task(task_data):
    """Add a task to the offline queue"""
    if not STREAMLIT_AVAILABLE:
        return False
    
    # Call JavaScript to store task
    js_code = f"""
    <script>
        if (window.qa3Pwa && window.qa3Pwa.storeTask) {{
            window.qa3Pwa.storeTask({json.dumps(task_data)});
        }}
    </script>
    """
    st.components.v1.html(js_code, height=0)
    return True

def get_pwa_controls():
    """Get PWA control elements for the sidebar"""
    if not STREAMLIT_AVAILABLE:
        return
    
    status = get_pwa_status()
    
    st.divider()
    st.subheader("PWA Controls")
    
    # Installation status
    if status.get("isInstalled", False):
        st.success("⚡ Running as installed app")
    else:
        if status.get("canInstall", False):
            display_pwa_install_button()
        else:
            st.info("📱 Install app for offline use")
    
    # Notification controls
    if status.get("supportsNotifications", False):
        if status.get("notificationsEnabled", False):
            st.success("🔔 Notifications enabled")
            if st.button("Test Notification"):
                # Call JavaScript to send test notification
                js_code = """
                <script>
                    if (window.qa3Pwa && window.qa3Pwa.sendTestNotification) {
                        window.qa3Pwa.sendTestNotification();
                    }
                </script>
                """
                st.components.v1.html(js_code, height=0)
        else:
            st.warning("🔕 Notifications disabled")
            if st.button("Enable Notifications"):
                # Call JavaScript to request notification permission
                js_code = """
                <script>
                    if (window.qa3Pwa && window.qa3Pwa.requestNotificationPermission) {
                        window.qa3Pwa.requestNotificationPermission();
                    }
                </script>
                """
                st.components.v1.html(js_code, height=0)

def add_browser_integration():
    """Add visible browser integration for the PWA"""
    if not STREAMLIT_AVAILABLE:
        return
    
    # Add JavaScript to handle browser status
    js_code = """
    <script>
        // Send browser status to Streamlit
        function sendBrowserStatus() {
            const status = {
                userAgent: navigator.userAgent,
                language: navigator.language,
                online: navigator.onLine,
                cookies: navigator.cookieEnabled,
                platform: navigator.platform,
                doNotTrack: navigator.doNotTrack,
                screenWidth: window.screen.width,
                screenHeight: window.screen.height,
                colorDepth: window.screen.colorDepth,
                devicePixelRatio: window.devicePixelRatio,
                memoryInfo: performance.memory ? {
                    jsHeapSizeLimit: performance.memory.jsHeapSizeLimit,
                    totalJSHeapSize: performance.memory.totalJSHeapSize,
                    usedJSHeapSize: performance.memory.usedJSHeapSize
                } : null
            };
            
            if (window.Streamlit) {
                window.Streamlit.setComponentValue({
                    type: 'browser_status',
                    data: status
                });
            }
        }
        
        // Send status when Streamlit is ready
        if (window.Streamlit) {
            window.Streamlit.onReady(() => {
                sendBrowserStatus();
            });
        }
        
        // Update status when online/offline changes
        window.addEventListener('online', sendBrowserStatus);
        window.addEventListener('offline', sendBrowserStatus);
    </script>
    """
    
    st.components.v1.html(js_code, height=0)

def display_pwa_card():
    """Display a card explaining PWA features"""
    if not STREAMLIT_AVAILABLE:
        return
    
    with st.expander("What is PWA (Progressive Web App)?"):
        st.markdown("""
        **Progressive Web App (PWA)** features of QA³ allow you to:
        
        - 📱 **Install as an app** on your device without going through an app store
        - 🔌 **Work offline** with limited functionality when you don't have internet
        - 🔔 **Receive notifications** about completed tasks or search results
        - 💾 **Save disk space** compared to traditional desktop apps
        - 🔄 **Always stay updated** with the latest version
        
        To install, look for the install button in the sidebar or use your browser's install option.
        """)

def display_notification_demo():
    """Display a demo of the notification system"""
    if not STREAMLIT_AVAILABLE:
        return
    
    status = get_pwa_status()
    
    st.subheader("Notification Demo")
    
    if not status.get("supportsNotifications", False):
        st.warning("Your browser does not support notifications.")
        return
    
    if not status.get("notificationsEnabled", False):
        st.warning("Notifications are not enabled. Enable them in the sidebar.")
        return
    
    notification_type = st.selectbox(
        "Notification Type",
        ["Task Completion", "Search Results", "Agent Update"]
    )
    
    if st.button("Send Demo Notification"):
        title = f"QA³: {notification_type}"
        body = ""
        
        if notification_type == "Task Completion":
            body = "Your task 'Analyze quantum computing trends' has been completed."
        elif notification_type == "Search Results":
            body = "Your search for 'quantum circuit optimization' found 15 new results."
        else:
            body = "QA³ Agent has been updated to version 1.0.1 with improved search capabilities."
        
        # Call JavaScript to send notification
        js_code = f"""
        <script>
            if (window.qa3Pwa && window.qa3Pwa.sendTestNotification) {{
                const notification = {{
                    title: "{title}",
                    body: "{body}",
                    url: window.location.href
                }};
                
                if (navigator.serviceWorker.controller) {{
                    navigator.serviceWorker.controller.postMessage({{
                        type: 'SEND_NOTIFICATION',
                        payload: notification
                    }});
                }} else {{
                    new Notification(notification.title, {{
                        body: notification.body,
                        icon: './icon-192.png'
                    }});
                }}
            }}
        </script>
        """
        
        st.components.v1.html(js_code, height=0)
        st.success(f"Sent {notification_type.lower()} notification: {body}")