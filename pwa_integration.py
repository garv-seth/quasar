"""
PWA Integration Module for QUASAR QA³

This module integrates Progressive Web App (PWA) capabilities with Streamlit,
enabling offline access, notifications, and installations.
"""

import os
import json
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Use import with try/except to handle LSP issues
try:
    import streamlit as st
    from streamlit.components.v1 import html
except ImportError:
    pass  # This will be handled at runtime

# Import static file server
from static_file_server import start_server_thread

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pwa-integration')

def start_static_server(port=8765) -> Tuple[threading.Thread, int]:
    """Start the static file server for PWA files if not already running
    
    Args:
        port: Port to start the server on
        
    Returns:
        Tuple of (server_thread, actual_port)
    """
    if 'static_server' not in st.session_state:
        logger.info(f"Starting static file server on port {port}")
        server_thread, actual_port = start_server_thread(port)
        
        if server_thread and actual_port:
            st.session_state.static_server = {
                'thread': server_thread,
                'port': actual_port
            }
            logger.info(f"Static file server started on port {actual_port}")
            return server_thread, actual_port
        else:
            logger.error("Failed to start static file server")
            return None, None
    else:
        return st.session_state.static_server['thread'], st.session_state.static_server['port']

def initialize_pwa():
    """Initialize PWA components for Streamlit"""
    # Start the static file server if not already running
    server_thread, actual_port = start_static_server()
    
    # Use the actual port in the URLs
    base_url = f"http://localhost:{actual_port}"
    
    # Add service worker registration and PWA scripts to the head
    manifest_link = f'<link rel="manifest" href="{base_url}/manifest.json">'
    theme_color = '<meta name="theme-color" content="#7B2FFF">'
    apple_meta = '<meta name="apple-mobile-web-app-capable" content="yes">'
    apple_icon = f'<link rel="apple-touch-icon" href="{base_url}/icon-192.png">'
    pwa_script = f'<script src="{base_url}/pwa.js" defer></script>'
    
    # Combine all head elements
    head_html = f"""
    {manifest_link}
    {theme_color}
    {apple_meta}
    {apple_icon}
    {pwa_script}
    """
    
    # Inject head elements into Streamlit
    st.markdown(head_html, unsafe_allow_html=True)
    
    # Create a container for PWA status updates from JavaScript
    if 'pwa_status' not in st.session_state:
        st.session_state.pwa_status = {
            'serviceWorkerSupported': False,
            'serviceWorkerRegistered': False,
            'serviceWorkerActive': False,
            'databaseSupported': False,
            'databaseInitialized': False,
            'pushManagerSupported': False,
            'notificationsSupported': False,
            'notificationsEnabled': False,
            'installPromptAvailable': False,
            'installed': False,
            'pwaMode': False,
            'online': True
        }

def get_pwa_status():
    """Get the current PWA status"""
    if 'pwa_status' not in st.session_state:
        initialize_pwa()
    
    return st.session_state.pwa_status

def check_pwa_mode():
    """Check if the app is running in PWA mode"""
    status = get_pwa_status()
    return status.get('pwaMode', False)

def display_pwa_install_button():
    """Display a button to install the PWA"""
    status = get_pwa_status()
    
    if status.get('installed', False):
        st.success("✅ QA³ is installed as a PWA")
        return
    
    if status.get('installPromptAvailable', False):
        install_button_html = """
        <button 
            onclick="window.installQuasarPwa()" 
            style="background-color: #7B2FFF; color: white; border: none; 
                  padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer;">
            Install QA³ as App
        </button>
        <script>
            // Update button when install state changes
            window.addEventListener('appinstalled', function() {
                document.querySelector('button').style.display = 'none';
            });
        </script>
        """
        st.markdown(install_button_html, unsafe_allow_html=True)
    else:
        st.info("⚠️ Install QA³ by using your browser's 'Add to Home Screen' or 'Install' option in the address bar")

def display_notification_button():
    """Display a button to enable/disable notifications"""
    status = get_pwa_status()
    
    if not status.get('notificationsSupported', False):
        st.warning("⚠️ Notifications are not supported in your browser")
        return
    
    button_text = "Disable Notifications" if status.get('notificationsEnabled', False) else "Enable Notifications"
    notification_button_html = f"""
    <button 
        id="notification-toggle"
        onclick="window.toggleQuasarNotifications()" 
        style="background-color: #7B2FFF; color: white; border: none; 
              padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer;">
        {button_text}
    </button>
    <button 
        onclick="window.sendQuasarTestNotification()" 
        style="background-color: #555555; color: white; border: none; 
              margin-left: 0.5rem; padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer;">
        Test Notification
    </button>
    """
    st.markdown(notification_button_html, unsafe_allow_html=True)

def add_offline_task(task_data):
    """Add a task to the offline queue"""
    # Create a component that will call the storeTask JavaScript function
    store_task_html = f"""
    <div id="store-task-container"></div>
    <script>
        (function() {{
            const taskData = {json.dumps(task_data)};
            if (window.quasarPwa) {{
                window.quasarPwa.storeTask(taskData)
                    .then(result => {{
                        console.log('Task stored:', result);
                    }})
                    .catch(error => {{
                        console.error('Failed to store task:', error);
                    }});
            }} else {{
                console.error('QUASAR PWA system not initialized');
            }}
        }})();
    </script>
    """
    st.markdown(store_task_html, unsafe_allow_html=True)

def get_pwa_controls():
    """Get PWA control elements for the sidebar"""
    status = get_pwa_status()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if status.get('installed', False):
            st.markdown("✅ **PWA Installed**")
        elif status.get('installPromptAvailable', False):
            display_pwa_install_button()
        else:
            st.markdown("⚙️ **PWA Available**")
    
    with col2:
        if status.get('serviceWorkerActive', False):
            if status.get('notificationsSupported', False):
                display_notification_button()
            else:
                st.markdown("⚙️ **Offline Ready**")
        else:
            st.markdown("⏳ **Initializing...**")
    
    return status

def create_pwa_files():
    """Create necessary PWA files in the public directory"""
    # This function would be called during deployment
    # For this demo, we've already created the files manually
    pass

def add_browser_integration():
    """Add visible browser integration for the PWA"""
    # Inject a component that displays browser controls in the PWA context
    browser_html = """
    <div id="browser-controls" class="pwa-browser" style="display: none;">
        <div class="browser-top">
            <button onclick="window.history.back()" class="browser-btn">←</button>
            <button onclick="window.history.forward()" class="browser-btn">→</button>
            <button onclick="window.location.reload()" class="browser-btn">↻</button>
            <div class="url-bar">
                <span id="browser-url"></span>
            </div>
        </div>
    </div>
    
    <style>
        .pwa-browser {
            border: 1px solid #444;
            border-radius: 8px;
            margin-bottom: 10px;
            background: #2A2A3C;
        }
        
        .browser-top {
            display: flex;
            align-items: center;
            padding: 8px;
            background: #333344;
            border-radius: 8px 8px 0 0;
        }
        
        .browser-btn {
            background: #555;
            border: none;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            margin-right: 5px;
            cursor: pointer;
        }
        
        .url-bar {
            flex-grow: 1;
            background: #222;
            padding: 5px 10px;
            border-radius: 15px;
            color: #aaa;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
    </style>
    
    <script>
        // Show browser controls only in PWA mode
        window.addEventListener('load', function() {
            setTimeout(function() {
                if (isPwaMode()) {
                    document.getElementById('browser-controls').style.display = 'block';
                    document.getElementById('browser-url').textContent = window.location.href;
                }
            }, 1000);
        });
    </script>
    """
    
    st.markdown(browser_html, unsafe_allow_html=True)

def display_pwa_card():
    """Display a card explaining PWA features"""
    with st.expander("About PWA Features", expanded=False):
        st.markdown("""
        ### Progressive Web App Features
        
        QA³ is available as a Progressive Web App, which means you can:
        
        - **Install it on your device** - Use it like a native app from your home screen
        - **Work offline** - Access core functionality even without an internet connection
        - **Receive notifications** - Get alerts when your quantum tasks complete
        - **Automatic updates** - Always have the latest version
        
        Try installing QA³ as an app for the best experience!
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            display_pwa_install_button()
        
        with col2:
            display_notification_button()

def display_notification_demo():
    """Display a demo of the notification system"""
    with st.expander("Notification Demo", expanded=False):
        st.markdown("""
        ### Notification System
        
        QA³ can send you notifications when:
        
        - Your quantum tasks complete
        - Optimization solutions are found
        - Factorization tasks finish
        - Search results are ready
        
        Try a test notification below:
        """)
        
        notification_demo_html = """
        <button 
            onclick="window.sendQuasarTestNotification()" 
            style="background-color: #7B2FFF; color: white; border: none; 
                  padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer;">
            Send Test Notification
        </button>
        """
        
        st.markdown(notification_demo_html, unsafe_allow_html=True)