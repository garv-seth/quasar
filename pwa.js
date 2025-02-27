// QUASAR Labs QA³ PWA Integration Script

// PWA Configuration
const PWA_CONFIG = {
  serviceWorkerUrl: '/sw.js',
  dbName: 'QUASAR-QA3-DB',
  dbVersion: 1,
  vapidPublicKey: 'BEl62iUYgUivxIkv69yViEuiBIa-Ib9-SkvMeAtA3LFgDzkrxZJjSgSnfckjBJuBkr3qBUYIHBQFLXYp5Nksh8U'
};

// Initialize PWA functionality
class QUASARPwa {
  constructor() {
    this.swRegistration = null;
    this.isSubscribed = false;
    this.db = null;
    
    this.init();
  }
  
  async init() {
    console.log('Initializing QUASAR QA³ PWA functionality...');
    
    // Check if PWA is supported
    if ('serviceWorker' in navigator) {
      this.initializeServiceWorker();
      this.initializeDatabase();
      this.checkInstallState();
      
      // Set up install button if available
      const installButton = document.getElementById('install-pwa');
      if (installButton) {
        installButton.addEventListener('click', () => this.installPwa());
      }
      
      // Set up notification button if available
      const notifyButton = document.getElementById('enable-notifications');
      if (notifyButton) {
        notifyButton.addEventListener('click', () => this.toggleNotifications());
      }
    } else {
      console.warn('Service workers are not supported in this browser');
      this.updatePwaStatus('Service workers not supported');
    }
    
    // Send PWA status to Streamlit
    this.sendPwaStatusToStreamlit();
  }
  
  // Initialize service worker
  async initializeServiceWorker() {
    try {
      this.swRegistration = await navigator.serviceWorker.register(PWA_CONFIG.serviceWorkerUrl);
      console.log('Service Worker registered successfully:', this.swRegistration);
      
      // Check notification subscription status
      this.updateSubscriptionStatus();
      
      // Listen for service worker state changes
      navigator.serviceWorker.addEventListener('controllerchange', () => {
        console.log('Service Worker controller changed');
      });
      
      // Send ready message to Streamlit once SW is active
      if (navigator.serviceWorker.controller) {
        this.updatePwaStatus('Service Worker active');
      }
    } catch (error) {
      console.error('Service Worker registration failed:', error);
      this.updatePwaStatus('Service Worker registration failed');
    }
  }
  
  // Initialize IndexedDB
  async initializeDatabase() {
    try {
      this.db = await this.openDatabase();
      console.log('Database initialized successfully');
    } catch (error) {
      console.error('Database initialization failed:', error);
    }
  }
  
  // Open the database
  openDatabase() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(PWA_CONFIG.dbName, PWA_CONFIG.dbVersion);
      
      request.onupgradeneeded = event => {
        const db = event.target.result;
        
        // Create task queue store for offline operations
        if (!db.objectStoreNames.contains('taskQueue')) {
          db.createObjectStore('taskQueue', { keyPath: 'id', autoIncrement: true });
        }
        
        // Create data cache store
        if (!db.objectStoreNames.contains('dataCache')) {
          db.createObjectStore('dataCache', { keyPath: 'key' });
        }
        
        // Create settings store
        if (!db.objectStoreNames.contains('settings')) {
          db.createObjectStore('settings', { keyPath: 'key' });
        }
      };
      
      request.onsuccess = event => {
        resolve(event.target.result);
      };
      
      request.onerror = event => {
        console.error('IndexedDB error:', event.target.error);
        reject(event.target.error);
      };
    });
  }
  
  // Store a task for offline processing
  async storeTask(task) {
    if (!this.db) await this.initializeDatabase();
    
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('taskQueue', 'readwrite');
      const store = tx.objectStore('taskQueue');
      
      const request = store.add({
        task: task,
        timestamp: new Date().toISOString()
      });
      
      request.onsuccess = () => {
        resolve(true);
        
        // Request background sync if supported
        if ('serviceWorker' in navigator && 'SyncManager' in window) {
          navigator.serviceWorker.ready.then(registration => {
            registration.sync.register('quantum-task-sync')
              .catch(err => console.error('Background sync registration failed:', err));
          });
        }
      };
      
      request.onerror = () => {
        console.error('Error storing task:', request.error);
        reject(request.error);
      };
    });
  }
  
  // Store data in cache
  async storeData(key, data) {
    if (!this.db) await this.initializeDatabase();
    
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('dataCache', 'readwrite');
      const store = tx.objectStore('dataCache');
      
      const request = store.put({
        key: key,
        data: data,
        timestamp: new Date().toISOString()
      });
      
      request.onsuccess = () => resolve(true);
      request.onerror = () => reject(request.error);
    });
  }
  
  // Retrieve data from cache
  async getData(key) {
    if (!this.db) await this.initializeDatabase();
    
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('dataCache', 'readonly');
      const store = tx.objectStore('dataCache');
      
      const request = store.get(key);
      
      request.onsuccess = () => resolve(request.result ? request.result.data : null);
      request.onerror = () => reject(request.error);
    });
  }
  
  // Store a setting
  async storeSetting(key, value) {
    if (!this.db) await this.initializeDatabase();
    
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('settings', 'readwrite');
      const store = tx.objectStore('settings');
      
      const request = store.put({
        key: key,
        value: value,
        timestamp: new Date().toISOString()
      });
      
      request.onsuccess = () => resolve(true);
      request.onerror = () => reject(request.error);
    });
  }
  
  // Get a setting
  async getSetting(key, defaultValue = null) {
    if (!this.db) await this.initializeDatabase();
    
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('settings', 'readonly');
      const store = tx.objectStore('settings');
      
      const request = store.get(key);
      
      request.onsuccess = () => resolve(request.result ? request.result.value : defaultValue);
      request.onerror = () => reject(request.error);
    });
  }
  
  // Check PWA install state
  checkInstallState() {
    // Check if using standalone mode (installed PWA)
    const isStandalone = window.matchMedia('(display-mode: standalone)').matches ||
                         window.navigator.standalone ||
                         document.referrer.includes('android-app://');
    
    if (isStandalone) {
      console.log('PWA is running in standalone mode');
      this.updatePwaStatus('PWA installed');
      
      // Store the information in Streamlit session
      this.sendToStreamlit('pwa_installed', true);
    } else {
      console.log('PWA is running in browser mode');
      
      // Check if the app can be installed
      window.addEventListener('beforeinstallprompt', (e) => {
        // Prevent Chrome 67+ from automatically showing the prompt
        e.preventDefault();
        // Stash the event so it can be triggered later
        this.deferredPrompt = e;
        
        // Show install button if available
        const installButton = document.getElementById('install-pwa');
        if (installButton) {
          installButton.style.display = 'block';
        }
        
        this.updatePwaStatus('PWA installable');
      });
    }
    
    // Listen for app installation
    window.addEventListener('appinstalled', (evt) => {
      console.log('QUASAR QA³ PWA was installed');
      this.updatePwaStatus('PWA installed');
      this.sendToStreamlit('pwa_installed', true);
      
      // Hide install button if available
      const installButton = document.getElementById('install-pwa');
      if (installButton) {
        installButton.style.display = 'none';
      }
    });
  }
  
  // Try to install the PWA
  async installPwa() {
    if (!this.deferredPrompt) {
      console.log('No installation prompt available');
      return;
    }
    
    // Show the installation prompt
    this.deferredPrompt.prompt();
    
    // Wait for the user to respond
    const choiceResult = await this.deferredPrompt.userChoice;
    
    if (choiceResult.outcome === 'accepted') {
      console.log('User accepted the PWA installation');
    } else {
      console.log('User declined the PWA installation');
    }
    
    // Clear the deferred prompt
    this.deferredPrompt = null;
  }
  
  // Update notification subscription status
  async updateSubscriptionStatus() {
    if (!this.swRegistration) return;
    
    try {
      const subscription = await this.swRegistration.pushManager.getSubscription();
      this.isSubscribed = !(subscription === null);
      
      console.log('User is ' + (this.isSubscribed ? 'subscribed' : 'not subscribed') + ' to notifications');
      
      // Update UI
      this.updateNotificationUI();
      
      // Save to settings
      if (this.isSubscribed) {
        this.storeSetting('notificationsEnabled', true);
        this.sendToStreamlit('notifications_enabled', true);
      } else {
        this.storeSetting('notificationsEnabled', false);
        this.sendToStreamlit('notifications_enabled', false);
      }
    } catch (err) {
      console.error('Error checking subscription:', err);
    }
  }
  
  // Update notification button state
  updateNotificationUI() {
    const notifyButton = document.getElementById('enable-notifications');
    if (!notifyButton) return;
    
    if (Notification.permission === 'denied') {
      notifyButton.textContent = 'Notifications Blocked';
      notifyButton.disabled = true;
      return;
    }
    
    if (this.isSubscribed) {
      notifyButton.textContent = 'Disable Notifications';
    } else {
      notifyButton.textContent = 'Enable Notifications';
    }
    
    notifyButton.disabled = false;
  }
  
  // Toggle notifications subscription
  async toggleNotifications() {
    if (!this.swRegistration) {
      console.warn('Service Worker not registered');
      return;
    }
    
    if (this.isSubscribed) {
      try {
        const subscription = await this.swRegistration.pushManager.getSubscription();
        
        if (subscription) {
          await subscription.unsubscribe();
          this.isSubscribed = false;
          this.storeSetting('notificationsEnabled', false);
          this.sendToStreamlit('notifications_enabled', false);
          console.log('User unsubscribed from notifications');
        }
      } catch (err) {
        console.error('Error unsubscribing:', err);
      }
    } else {
      try {
        const applicationServerKey = this.urlB64ToUint8Array(PWA_CONFIG.vapidPublicKey);
        const subscription = await this.swRegistration.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey: applicationServerKey
        });
        
        console.log('User subscribed to notifications:', subscription);
        
        // Send the subscription to the server
        // await this.sendSubscriptionToServer(subscription);
        
        this.isSubscribed = true;
        this.storeSetting('notificationsEnabled', true);
        this.sendToStreamlit('notifications_enabled', true);
      } catch (err) {
        console.error('Failed to subscribe to notifications:', err);
        if (Notification.permission === 'denied') {
          console.warn('Notifications are blocked by the browser');
        }
      }
    }
    
    // Update UI
    this.updateNotificationUI();
  }
  
  // Send a notification
  async sendNotification(title, options = {}) {
    if (!this.swRegistration) {
      console.warn('Service Worker not registered');
      return false;
    }
    
    // Check if subscribed
    const subscribed = await this.getSetting('notificationsEnabled', false);
    if (!subscribed) {
      console.warn('User not subscribed to notifications');
      return false;
    }
    
    // Default options
    const notificationOptions = {
      body: options.body || 'New update from your quantum agent',
      icon: options.icon || '/icon-192.png',
      badge: '/badge.png',
      vibrate: options.vibrate || [100, 50, 100],
      data: {
        url: options.url || '/',
        timestamp: new Date().getTime()
      },
      actions: options.actions || [
        { action: 'view', title: 'View' },
        { action: 'dismiss', title: 'Dismiss' }
      ]
    };
    
    try {
      await this.swRegistration.showNotification(title, notificationOptions);
      return true;
    } catch (err) {
      console.error('Failed to show notification:', err);
      return false;
    }
  }
  
  // Send PWA status to Streamlit
  sendPwaStatusToStreamlit() {
    // Check if Streamlit is available
    if (typeof window.parent.postMessage !== 'function') return;
    
    // Gather status
    const pwaStatus = {
      serviceWorkerSupported: 'serviceWorker' in navigator,
      serviceWorkerRegistered: !!this.swRegistration,
      notificationsSupported: 'Notification' in window,
      notificationsPermission: Notification.permission,
      isStandalone: window.matchMedia('(display-mode: standalone)').matches ||
                   window.navigator.standalone ||
                   document.referrer.includes('android-app://'),
      isOnline: navigator.onLine
    };
    
    // Send to Streamlit
    this.sendToStreamlit('pwa_status', pwaStatus);
  }
  
  // Update the PWA status display
  updatePwaStatus(status) {
    const statusElement = document.getElementById('pwa-status');
    if (statusElement) {
      statusElement.textContent = status;
    }
    
    // Also send to Streamlit
    this.sendToStreamlit('pwa_status_message', status);
  }
  
  // Send data to Streamlit
  sendToStreamlit(key, value) {
    try {
      // Check if Streamlit is available
      if (typeof window.parent.postMessage !== 'function') return;
      
      // Send message to Streamlit
      window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        key: key,
        value: value
      }, '*');
    } catch (err) {
      console.error('Error sending data to Streamlit:', err);
    }
  }
  
  // Convert base64 to Uint8Array (for VAPID keys)
  urlB64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding)
      .replace(/\-/g, '+')
      .replace(/_/g, '/');
    
    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);
    
    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }
    
    return outputArray;
  }
}

// Initialize the PWA functionality
window.quasarPwa = new QUASARPwa();

// Helper function to check if app is running in PWA mode
function isPwaMode() {
  return window.matchMedia('(display-mode: standalone)').matches ||
         window.navigator.standalone ||
         document.referrer.includes('android-app://');
}

// Add event listener for online/offline status
window.addEventListener('online', () => {
  console.log('Application is online');
  document.body.classList.remove('offline');
  
  // Notify Streamlit
  if (window.quasarPwa) {
    window.quasarPwa.sendToStreamlit('is_online', true);
  }
  
  // Try to sync pending tasks
  if ('serviceWorker' in navigator && 'SyncManager' in window) {
    navigator.serviceWorker.ready.then(registration => {
      registration.sync.register('quantum-task-sync')
        .catch(err => console.error('Background sync registration failed:', err));
    });
  }
});

window.addEventListener('offline', () => {
  console.log('Application is offline');
  document.body.classList.add('offline');
  
  // Notify Streamlit
  if (window.quasarPwa) {
    window.quasarPwa.sendToStreamlit('is_online', false);
  }
});