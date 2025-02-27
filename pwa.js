/**
 * QUASAR QA³: PWA Implementation
 * Provides Progressive Web App functionality for the Quantum-Accelerated AI Agent
 */

class QUASARPwa {
  constructor() {
    this.deferredPrompt = null;
    this.isInstalled = false;
    this.notificationsEnabled = false;
    this.db = null;
    this.initialized = false;

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => this.init());
    } else {
      this.init();
    }
  }

  async init() {
    console.log('[PWA] Initializing QUASAR QA³ PWA...');
    
    try {
      await this.initializeServiceWorker();
      await this.initializeDatabase();
      this.checkInstallState();
      this.updateNotificationUI();
      this.registerEventListeners();
      
      this.initialized = true;
      console.log('[PWA] Initialization complete');
      
      // Update Streamlit with the PWA status
      this.sendPwaStatusToStreamlit();
    } catch (error) {
      console.error('[PWA] Initialization failed:', error);
    }
  }

  async initializeServiceWorker() {
    if ('serviceWorker' in navigator) {
      try {
        // If SERVICE_WORKER_URL is defined (from the streamlit component), use it
        const swUrl = window.SERVICE_WORKER_URL || '/sw.js';
        
        const registration = await navigator.serviceWorker.register(swUrl, {
          scope: '/'
        });
        console.log('[PWA] Service Worker registered with scope:', registration.scope);
        
        // Set up background sync for offline tasks if supported
        if ('sync' in registration) {
          console.log('[PWA] Background Sync supported');
        }
        
        // Set up push notifications if supported
        if ('pushManager' in registration) {
          console.log('[PWA] Push API supported');
          this.notificationsSupported = true;
        }
      } catch (error) {
        console.error('[PWA] Service Worker registration failed:', error);
      }
    } else {
      console.warn('[PWA] Service Workers not supported');
    }
  }

  async initializeDatabase() {
    if (!('indexedDB' in window)) {
      console.warn('[PWA] IndexedDB not supported');
      return;
    }
    
    try {
      this.db = await this.openDatabase();
      console.log('[PWA] Database initialized');
    } catch (error) {
      console.error('[PWA] Database initialization failed:', error);
    }
  }
  
  openDatabase() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('quasar-qa3-db', 1);
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        
        // Create object stores
        if (!db.objectStoreNames.contains('tasks')) {
          const taskStore = db.createObjectStore('tasks', { keyPath: 'id', autoIncrement: true });
          taskStore.createIndex('status', 'status', { unique: false });
          taskStore.createIndex('timestamp', 'timestamp', { unique: false });
        }
        
        if (!db.objectStoreNames.contains('settings')) {
          db.createObjectStore('settings', { keyPath: 'key' });
        }
        
        if (!db.objectStoreNames.contains('cache')) {
          db.createObjectStore('cache', { keyPath: 'key' });
        }
      };
      
      request.onsuccess = (event) => {
        resolve(event.target.result);
      };
      
      request.onerror = (event) => {
        console.error('[PWA] Database error:', event.target.error);
        reject(event.target.error);
      };
    });
  }
  
  async storeTask(task) {
    if (!this.db) {
      console.error('[PWA] Database not initialized');
      return false;
    }
    
    try {
      const transaction = this.db.transaction(['tasks'], 'readwrite');
      const store = transaction.objectStore('tasks');
      
      // Add timestamp if not present
      if (!task.timestamp) {
        task.timestamp = new Date().toISOString();
      }
      
      const request = store.add(task);
      
      return new Promise((resolve, reject) => {
        request.onsuccess = () => {
          console.log('[PWA] Task stored successfully');
          
          // Request background sync if available
          if ('serviceWorker' in navigator && 'SyncManager' in window) {
            navigator.serviceWorker.ready.then(registration => {
              registration.sync.register('sync-quantum-tasks');
            });
          }
          
          resolve(true);
        };
        
        request.onerror = () => {
          console.error('[PWA] Failed to store task');
          reject(new Error('Failed to store task'));
        };
      });
    } catch (error) {
      console.error('[PWA] Error storing task:', error);
      return false;
    }
  }
  
  async storeData(key, data) {
    if (!this.db) return false;
    
    try {
      const transaction = this.db.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      
      const item = {
        key,
        data,
        timestamp: new Date().toISOString()
      };
      
      const request = store.put(item);
      
      return new Promise((resolve, reject) => {
        request.onsuccess = () => resolve(true);
        request.onerror = () => reject(new Error('Failed to store data'));
      });
    } catch (error) {
      console.error('[PWA] Error storing data:', error);
      return false;
    }
  }
  
  async getData(key) {
    if (!this.db) return null;
    
    try {
      const transaction = this.db.transaction(['cache'], 'readonly');
      const store = transaction.objectStore('cache');
      
      const request = store.get(key);
      
      return new Promise((resolve, reject) => {
        request.onsuccess = () => {
          resolve(request.result ? request.result.data : null);
        };
        
        request.onerror = () => {
          reject(new Error('Failed to retrieve data'));
        };
      });
    } catch (error) {
      console.error('[PWA] Error retrieving data:', error);
      return null;
    }
  }
  
  async storeSetting(key, value) {
    if (!this.db) return false;
    
    try {
      const transaction = this.db.transaction(['settings'], 'readwrite');
      const store = transaction.objectStore('settings');
      
      const item = {
        key,
        value,
        timestamp: new Date().toISOString()
      };
      
      const request = store.put(item);
      
      return new Promise((resolve, reject) => {
        request.onsuccess = () => resolve(true);
        request.onerror = () => reject(new Error('Failed to store setting'));
      });
    } catch (error) {
      console.error('[PWA] Error storing setting:', error);
      return false;
    }
  }
  
  async getSetting(key, defaultValue = null) {
    if (!this.db) return defaultValue;
    
    try {
      const transaction = this.db.transaction(['settings'], 'readonly');
      const store = transaction.objectStore('settings');
      
      const request = store.get(key);
      
      return new Promise((resolve, reject) => {
        request.onsuccess = () => {
          resolve(request.result ? request.result.value : defaultValue);
        };
        
        request.onerror = () => {
          reject(new Error('Failed to retrieve setting'));
        };
      });
    } catch (error) {
      console.error('[PWA] Error retrieving setting:', error);
      return defaultValue;
    }
  }
  
  registerEventListeners() {
    // Listen for beforeinstallprompt event to handle PWA installation
    window.addEventListener('beforeinstallprompt', (event) => {
      // Prevent Chrome 67 and earlier from automatically showing the prompt
      event.preventDefault();
      // Stash the event so it can be triggered later
      this.deferredPrompt = event;
      
      // Update the UI to show the install button
      this.updatePwaStatus('installable', true);
      
      // Find and attach event listeners to all install buttons
      document.querySelectorAll('#install-pwa, #install-pwa-card').forEach(button => {
        if (button) {
          button.style.display = 'inline-block';
          button.addEventListener('click', () => this.installPwa());
        }
      });
    });
    
    // Track when the PWA is successfully installed
    window.addEventListener('appinstalled', () => {
      console.log('[PWA] Application installed');
      this.isInstalled = true;
      this.deferredPrompt = null;
      
      // Update settings
      this.storeSetting('installed', true);
      this.updatePwaStatus('installed', true);
      
      // Hide install buttons
      document.querySelectorAll('#install-pwa, #install-pwa-card').forEach(button => {
        if (button) button.style.display = 'none';
      });
    });
    
    // Set up notification toggle buttons
    document.querySelectorAll('#enable-notifications').forEach(button => {
      if (button) {
        button.addEventListener('click', () => this.toggleNotifications());
      }
    });
    
    // Listen for online/offline events
    window.addEventListener('online', () => {
      console.log('[PWA] Application is online');
      this.updatePwaStatus('online', true);
      
      // Attempt to sync pending tasks
      if ('serviceWorker' in navigator && 'SyncManager' in window) {
        navigator.serviceWorker.ready.then(registration => {
          registration.sync.register('sync-quantum-tasks');
        });
      }
    });
    
    window.addEventListener('offline', () => {
      console.log('[PWA] Application is offline');
      this.updatePwaStatus('online', false);
    });
  }
  
  checkInstallState() {
    // Check if running as a PWA
    if (window.matchMedia('(display-mode: standalone)').matches || 
        window.navigator.standalone === true) {
      this.isInstalled = true;
      this.storeSetting('installed', true);
      this.updatePwaStatus('installed', true);
      console.log('[PWA] Running in standalone/installed mode');
      
      // Hide install buttons
      document.querySelectorAll('#install-pwa, #install-pwa-card').forEach(button => {
        if (button) button.style.display = 'none';
      });
    } else {
      // Update from stored settings
      this.getSetting('installed', false).then(installed => {
        this.isInstalled = installed;
        this.updatePwaStatus('installed', installed);
      });
    }
  }

  async installPwa() {
    if (!this.deferredPrompt) {
      console.log('[PWA] No installation prompt available');
      return;
    }
    
    // Show the installation prompt
    this.deferredPrompt.prompt();
    
    // Wait for the user to respond to the prompt
    const { outcome } = await this.deferredPrompt.userChoice;
    console.log(`[PWA] User response to installation prompt: ${outcome}`);
    
    // Clear the saved prompt
    this.deferredPrompt = null;
  }
  
  async updateSubscriptionStatus() {
    if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
      console.warn('[PWA] Push notifications not supported');
      this.notificationsSupported = false;
      return;
    }
    
    try {
      const registration = await navigator.serviceWorker.ready;
      const subscription = await registration.pushManager.getSubscription();
      
      this.notificationsEnabled = !!subscription;
      this.updatePwaStatus('notifications_enabled', this.notificationsEnabled);
      
      // Update settings
      this.storeSetting('notifications_enabled', this.notificationsEnabled);
      
      this.updateNotificationUI();
    } catch (error) {
      console.error('[PWA] Error checking notification status:', error);
    }
  }
  
  updateNotificationUI() {
    // Update notification toggle buttons
    document.querySelectorAll('#enable-notifications').forEach(button => {
      if (!button) return;
      
      if (this.notificationsEnabled) {
        button.textContent = 'Disable Notifications';
        button.classList.add('enabled');
      } else {
        button.textContent = 'Enable Notifications';
        button.classList.remove('enabled');
      }
    });
  }
  
  async toggleNotifications() {
    if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
      console.warn('[PWA] Push notifications not supported');
      return;
    }
    
    try {
      // Request permission first
      const permission = await Notification.requestPermission();
      if (permission !== 'granted') {
        console.warn('[PWA] Notification permission denied');
        return;
      }
      
      const registration = await navigator.serviceWorker.ready;
      const existingSubscription = await registration.pushManager.getSubscription();
      
      if (existingSubscription) {
        // Unsubscribe
        await existingSubscription.unsubscribe();
        this.notificationsEnabled = false;
      } else {
        // Subscribe
        // This would normally involve your server creating a subscription
        // Here's a simplified version
        const subscription = await registration.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey: this.urlB64ToUint8Array(
            // This is a placeholder public key - in a real implementation,
            // this would come from your server
            'BEl62iUYgUivxIkv69yViEuiBIa-Ib9-SkvMeAtA3LFgDzkrxZJjSgSnfckjBJuBkr3qBUYIHBQFLXYp5Nksh8U'
          )
        });
        
        this.notificationsEnabled = true;
        console.log('[PWA] Push notification subscription:', JSON.stringify(subscription));
        
        // In a real implementation, you would send the subscription to your server
        // Here we'll simulate success
        await this.sendTestNotification();
      }
      
      // Update settings and UI
      this.storeSetting('notifications_enabled', this.notificationsEnabled);
      this.updatePwaStatus('notifications_enabled', this.notificationsEnabled);
      this.updateNotificationUI();
    } catch (error) {
      console.error('[PWA] Error toggling notifications:', error);
    }
  }
  
  async sendTestNotification() {
    // This simulates receiving a push notification for testing purposes
    if (!('serviceWorker' in navigator)) {
      console.warn('[PWA] Service workers not supported');
      return;
    }
    
    try {
      const registration = await navigator.serviceWorker.ready;
      
      const title = 'QUASAR QA³ Notification';
      const options = {
        body: 'Notifications are now enabled for quantum task updates!',
        icon: '/icon-192.png',
        badge: '/icon-192.png',
        tag: 'test-notification',
        actions: [
          { action: 'view', title: 'View App' }
        ],
        data: {
          url: '/'
        }
      };
      
      await registration.showNotification(title, options);
    } catch (error) {
      console.error('[PWA] Error sending test notification:', error);
    }
  }
  
  sendPwaStatusToStreamlit() {
    // Send current status to Streamlit component
    // This would normally be used by a Streamlit component to update the UI
    const status = {
      installed: this.isInstalled,
      notifications_enabled: this.notificationsEnabled,
      online: navigator.onLine,
      supported: {
        serviceWorker: 'serviceWorker' in navigator,
        pushManager: 'PushManager' in window,
        indexedDB: 'indexedDB' in window,
        sync: 'SyncManager' in window,
        notifications: 'Notification' in window
      },
      initialized: this.initialized
    };
    
    // Update status display element if present
    const statusEl = document.getElementById('pwa-status');
    if (statusEl) {
      statusEl.textContent = this.isInstalled 
        ? '✓ Installed' 
        : (this.deferredPrompt ? '⭐ Available to install' : '✗ Not installable');
    }
    
    try {
      if (window.parent) {
        // Send to Streamlit component
        window.parent.postMessage({
          type: 'streamlit:setComponentValue',
          key: 'pwa_status',
          value: status,
          dataType: 'json'
        }, '*');
      }
      
      console.log('[PWA] Status sent to Streamlit:', status);
    } catch (error) {
      console.error('[PWA] Error sending status to Streamlit:', error);
    }
  }
  
  updatePwaStatus(key, value) {
    // Update a single status key and send to Streamlit
    this.sendToStreamlit(key, value);
  }
  
  sendToStreamlit(key, value) {
    try {
      if (window.parent) {
        // Send to Streamlit component
        window.parent.postMessage({
          type: 'streamlit:setComponentValue',
          key: key,
          value: value
        }, '*');
      }
    } catch (error) {
      console.error('[PWA] Error sending to Streamlit:', error);
    }
  }
  
  // Utility function to convert base64 to Uint8Array for web push
  urlB64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding)
      .replace(/-/g, '+')
      .replace(/_/g, '/');
    
    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);
    
    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }
    
    return outputArray;
  }
}

// Initialize the PWA when the script loads
const quasarPwa = new QUASARPwa();

// Make the instance available globally
window.quasarPwa = quasarPwa;

// Helper function to check if we're in PWA mode
function isPwaMode() {
  return window.matchMedia('(display-mode: standalone)').matches || 
         window.navigator.standalone === true;
}