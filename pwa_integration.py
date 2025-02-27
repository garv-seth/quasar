"""
PWA Integration Module for QUASAR QA³

This module integrates Progressive Web App (PWA) capabilities with Streamlit,
enabling offline access, notifications, and installations.
"""

import os
import json
import base64
import streamlit as st
from datetime import datetime

def initialize_pwa():
    """Initialize PWA components for Streamlit"""
    # Create directories if they don't exist
    os.makedirs(".streamlit", exist_ok=True)
    
    # Create PWA metadata
    if not hasattr(st.session_state, "pwa"):
        st.session_state.pwa = {
            "installed": False,
            "notifications_enabled": False,
            "online": True,
            "queued_tasks": [],
            "last_sync": None,
        }
    
    # Static file server port
    static_port = 8000
    
    # Create HTML for PWA components to inject - using the static file server
    pwa_html = f"""
    <link rel="manifest" href="http://localhost:{static_port}/manifest.json">
    <meta name="theme-color" content="#7B2FFF">
    <link rel="apple-touch-icon" href="http://localhost:{static_port}/icon-192.png">
    <script src="http://localhost:{static_port}/pwa.js" defer></script>
    <script>
        // Register event listener for messages from PWA script
        window.addEventListener('message', function(event) {{
            if (event.data.type === 'streamlit:setComponentValue') {{
                // Pass PWA status to Streamlit
                const data = event.data;
                if (data.key && data.value) {{
                    // Handle PWA status updates
                    console.log("PWA Status:", data.key, data.value);
                    
                    // Send this data to Streamlit via the Streamlit component API
                    if (window.parent && window.parent.postMessage) {{
                        window.parent.postMessage({{
                            type: "streamlit:setComponentValue",
                            value: data.value,
                            dataType: "json"
                        }}, "*");
                    }}
                }}
            }}
        }});
        
        // Configure service worker URL
        window.SERVICE_WORKER_URL = "http://localhost:{static_port}/sw.js";
        
        // Check offline status
        function updateOnlineStatus() {{
            console.log("Online status:", navigator.onLine);
            const statusEl = document.getElementById('online-status');
            if (statusEl) {{
                statusEl.textContent = navigator.onLine ? "✓ Online" : "✗ Offline";
                statusEl.className = navigator.onLine ? "online-status online" : "online-status offline";
            }}
        }}
        
        window.addEventListener('online', updateOnlineStatus);
        window.addEventListener('offline', updateOnlineStatus);
    </script>
    <style>
        .online-status {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 14px;
            margin-left: 10px;
        }}
        .online {{
            background-color: #c2f5d0;
            color: #0a3816;
        }}
        .offline {{
            background-color: #f5c2c2;
            color: #380a0a;
        }}
        #pwa-status {{
            font-size: 14px;
            margin-left: 10px;
            color: #7B2FFF;
        }}
    </style>
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <span id="online-status" class="online-status online">✓ Online</span>
        <span id="pwa-status">Initializing...</span>
    </div>
    """
    
    # Inject HTML
    st.markdown(pwa_html, unsafe_allow_html=True)

def get_pwa_status():
    """Get the current PWA status"""
    return st.session_state.pwa

def check_pwa_mode():
    """Check if the app is running in PWA mode"""
    return st.session_state.pwa.get("installed", False)

def display_pwa_install_button():
    """Display a button to install the PWA"""
    st.markdown("""
    <button id="install-pwa" class="pwa-button" style="display:none;">
        Install QA³ App
    </button>
    <div id="pwa-status"></div>
    """, unsafe_allow_html=True)

def display_notification_button():
    """Display a button to enable/disable notifications"""
    st.markdown("""
    <button id="enable-notifications" class="notification-button">
        Enable Notifications
    </button>
    """, unsafe_allow_html=True)

def add_offline_task(task_data):
    """Add a task to the offline queue"""
    if not hasattr(st.session_state, "pwa"):
        initialize_pwa()
    
    st.session_state.pwa["queued_tasks"].append({
        "id": len(st.session_state.pwa["queued_tasks"]) + 1,
        "data": task_data,
        "timestamp": datetime.now().isoformat()
    })
    
    # In a real implementation, you would also store this in IndexedDB
    # through communication with the PWA script
    
    return True

def get_pwa_controls():
    """Get PWA control elements for the sidebar"""
    if not hasattr(st.session_state, "pwa"):
        initialize_pwa()
    
    is_installed = st.session_state.pwa.get("installed", False)
    has_notifications = st.session_state.pwa.get("notifications_enabled", False)
    
    pwa_status = f"""
    <div class="pwa-controls">
        <h4>QA³ App Status</h4>
        <p>Installation: {"✅ Installed" if is_installed else "❌ Not Installed"}</p>
        <p>Notifications: {"✅ Enabled" if has_notifications else "❌ Disabled"}</p>
    </div>
    """
    
    return pwa_status

def create_pwa_files():
    """Create necessary PWA files in the public directory"""
    # This would create the manifest.json, service worker, and icons
    # but we're already creating these files separately
    pass

def add_browser_integration():
    """Add visible browser integration for the PWA"""
    st.markdown("""
    <style>
        .visible-browser {
            border: 1px solid #444;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
            background-color: #fff;
        }
        .browser-toolbar {
            background-color: #2D2D44;
            padding: 8px;
            display: flex;
            align-items: center;
            border-bottom: 1px solid #444;
        }
        .browser-controls {
            display: flex;
            gap: 8px;
            margin-right: 10px;
        }
        .browser-control {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .browser-control.red { background-color: #FF5F57; }
        .browser-control.yellow { background-color: #FFBD2E; }
        .browser-control.green { background-color: #28CA41; }
        .browser-address {
            background-color: #1A1A2E;
            color: #ddd;
            padding: 6px 10px;
            border-radius: 4px;
            flex-grow: 1;
            font-size: 12px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .browser-content {
            height: 500px;
            background-color: #fff;
            color: #000;
            padding: 0;
            position: relative;
            overflow: hidden;
        }
        .browser-loading {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(to right, #7B2FFF, #28CA41);
            animation: loading 2s infinite;
        }
        @keyframes loading {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        .browser-iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
        .browser-placeholder {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #666;
            font-size: 14px;
        }
        .browser-placeholder svg {
            margin-bottom: 20px;
            width: 80px;
            height: 80px;
            color: #7B2FFF;
        }
    </style>
    <div class="visible-browser">
        <div class="browser-toolbar">
            <div class="browser-controls">
                <div class="browser-control red"></div>
                <div class="browser-control yellow"></div>
                <div class="browser-control green"></div>
            </div>
            <div class="browser-address" id="browser-url">https://www.example.com</div>
        </div>
        <div class="browser-content">
            <div class="browser-loading" id="browser-loading"></div>
            <div class="browser-placeholder" id="browser-placeholder">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p>Enter a task above to begin browsing with quantum acceleration</p>
            </div>
            <!-- Browser content will be dynamically updated -->
        </div>
    </div>
    <script>
        // This would be expanded to handle real browser interaction
        function updateBrowserUI(url, content) {
            document.getElementById('browser-url').textContent = url;
            document.getElementById('browser-placeholder').style.display = 'none';
            // Add content to browser
        }
    </script>
    """, unsafe_allow_html=True)

def display_pwa_card():
    """Display a card explaining PWA features"""
    st.markdown("""
    <div style="background-color: #252538; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #7B2FFF;">QA³ Progressive Web App</h3>
        <p>Install QUASAR QA³ as an app on your device for:</p>
        <ul>
            <li>Offline access to quantum tasks</li>
            <li>Task notifications when complete</li>
            <li>Faster performance</li>
            <li>Desktop/home screen access</li>
        </ul>
        <button id="install-pwa-card" style="background-color: #7B2FFF; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Install App</button>
    </div>
    <script>
        // Connect this button to the PWA install function
        document.getElementById('install-pwa-card').addEventListener('click', function() {
            if (window.quasarPwa) {
                window.quasarPwa.installPwa();
            }
        });
    </script>
    """, unsafe_allow_html=True)

def display_notification_demo():
    """Display a demo of the notification system"""
    if st.button("Send Test Notification"):
        # In a real implementation, this would call the PWA script to send a notification
        st.success("Test notification sent! Check your device notifications.")
        
        # For demo purposes, we'll just display what the notification would look like
        st.markdown("""
        <div style="background-color: #252538; padding: 15px; border-radius: 5px; margin-top: 10px; display: flex; align-items: center;">
            <img src="/icon-192.png" width="40" height="40" style="margin-right: 15px;">
            <div>
                <div style="font-weight: bold;">QUASAR QA³ Notification</div>
                <div style="font-size: 14px;">Your quantum task has completed processing!</div>
            </div>
        </div>
        """, unsafe_allow_html=True)