/**
 * QUASAR QA³: PWA Implementation
 * Provides Progressive Web App functionality for the Quantum-Accelerated AI Agent
 */

class QUASARPwa {
  constructor() {
    this.db = null;
    this.swRegistration = null;
    this.installPrompt = null;
    this.status = {
      serviceWorkerSupported: 'serviceWorker' in navigator,
      serviceWorkerRegistered: false,
      serviceWorkerActive: false,
      databaseSupported: 'indexedDB' in window,
      databaseInitialized: false,
      pushManagerSupported: 'PushManager' in window,
      notificationsSupported: 'Notification' in window,
      notificationsEnabled: false,
      installPromptAvailable: false,
      installed: false,
      pwaMode: window.matchMedia('(display-mode: standalone)').matches,
      online: navigator.onLine
    };
    
    // Bind methods to this
    this.init = this.init.bind(this);
    this.initializeServiceWorker = this.initializeServiceWorker.bind(this);
    this.initializeDatabase = this.initializeDatabase.bind(this);
    this.registerEventListeners = this.registerEventListeners.bind(this);
    this.storeTask = this.storeTask.bind(this);
    this.installPwa = this.installPwa.bind(this);
    this.toggleNotifications = this.toggleNotifications.bind(this);
    this.sendTestNotification = this.sendTestNotification.bind(this);
    this.updatePwaStatus = this.updatePwaStatus.bind(this);
    this.sendToStreamlit = this.sendToStreamlit.bind(this);
  }
  
  async init() {
    console.log('Initializing QUASAR QA³ PWA...');
    
    // Update installed state
    this.status.installed = 
      window.matchMedia('(display-mode: standalone)').matches || 
      window.navigator.standalone ||
      document.referrer.includes('android-app://');
    
    // Initialize service worker
    if (this.status.serviceWorkerSupported) {
      await this.initializeServiceWorker();
    }
    
    // Initialize database
    if (this.status.databaseSupported) {
      await this.initializeDatabase();
    }
    
    // Register event listeners
    this.registerEventListeners();
    
    // Check installation state
    this.checkInstallState();
    
    // Update notification permissions
    if (this.status.notificationsSupported) {
      this.updateSubscriptionStatus();
    }
    
    // Send initial status to Streamlit
    this.sendPwaStatusToStreamlit();
    
    console.log('QUASAR QA³ PWA initialized', this.status);
    return this.status;
  }
  
  async initializeServiceWorker() {
    try {
      this.swRegistration = await navigator.serviceWorker.register('/sw.js');
      this.updatePwaStatus('serviceWorkerRegistered', true);
      
      if (this.swRegistration.active) {
        this.updatePwaStatus('serviceWorkerActive', true);
      }
      
      this.swRegistration.addEventListener('updatefound', () => {
        const newWorker = this.swRegistration.installing;
        
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'activated') {
            this.updatePwaStatus('serviceWorkerActive', true);
          }
        });
      });
      
      console.log('Service Worker registered successfully');
    } catch (error) {
      console.error('Service Worker registration failed:', error);
    }
  }
  
  async initializeDatabase() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('QUASAR_QA3_DB', 1);
      
      request.onerror = (event) => {
        console.error('IndexedDB error:', event.target.errorCode);
        reject(event.target.errorCode);
      };
      
      request.onsuccess = (event) => {
        this.db = event.target.result;
        this.updatePwaStatus('databaseInitialized', true);
        console.log('Database initialized successfully');
        resolve(this.db);
      };
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        
        // Create object stores
        if (!db.objectStoreNames.contains('tasks')) {
          const taskStore = db.createObjectStore('tasks', { keyPath: 'id', autoIncrement: true });
          taskStore.createIndex('status', 'status', { unique: false });
          taskStore.createIndex('timestamp', 'timestamp', { unique: false });
        }
        
        if (!db.objectStoreNames.contains('circuits')) {
          const circuitStore = db.createObjectStore('circuits', { keyPath: 'id', autoIncrement: true });
          circuitStore.createIndex('name', 'name', { unique: false });
          circuitStore.createIndex('timestamp', 'timestamp', { unique: false });
        }
        
        if (!db.objectStoreNames.contains('settings')) {
          db.createObjectStore('settings', { keyPath: 'key' });
        }
      };
    });
  }
  
  openDatabase() {
    return new Promise((resolve, reject) => {
      if (this.db) {
        resolve(this.db);
        return;
      }
      
      const request = indexedDB.open('QUASAR_QA3_DB', 1);
      
      request.onerror = (event) => {
        console.error('IndexedDB error:', event.target.errorCode);
        reject(event.target.errorCode);
      };
      
      request.onsuccess = (event) => {
        this.db = event.target.result;
        resolve(this.db);
      };
    });
  }
  
  async storeTask(task) {
    try {
      const db = await this.openDatabase();
      
      return new Promise((resolve, reject) => {
        const transaction = db.transaction(['tasks'], 'readwrite');
        const store = transaction.objectStore('tasks');
        
        // Add timestamp if not present
        if (!task.timestamp) {
          task.timestamp = Date.now();
        }
        
        // Add status if not present
        if (!task.status) {
          task.status = 'pending';
        }
        
        const request = store.add(task);
        
        request.onsuccess = (event) => {
          console.log('Task stored successfully with ID:', event.target.result);
          resolve({ 
            success: true, 
            id: event.target.result,
            message: 'Task stored successfully'
          });
        };
        
        request.onerror = (event) => {
          console.error('Error storing task:', event.target.error);
          reject({
            success: false,
            error: event.target.error.message,
            message: 'Failed to store task'
          });
        };
      });
    } catch (error) {
      console.error('Database error:', error);
      throw error;
    }
  }
  
  async storeData(key, data) {
    try {
      const db = await this.openDatabase();
      
      return new Promise((resolve, reject) => {
        const transaction = db.transaction(['settings'], 'readwrite');
        const store = transaction.objectStore('settings');
        
        const request = store.put({ key, value: data, timestamp: Date.now() });
        
        request.onsuccess = () => {
          resolve({ success: true, message: 'Data stored successfully' });
        };
        
        request.onerror = (event) => {
          reject({ 
            success: false, 
            error: event.target.error.message,
            message: 'Failed to store data'
          });
        };
      });
    } catch (error) {
      console.error('Database error:', error);
      throw error;
    }
  }
  
  async getData(key) {
    try {
      const db = await this.openDatabase();
      
      return new Promise((resolve, reject) => {
        const transaction = db.transaction(['settings'], 'readonly');
        const store = transaction.objectStore('settings');
        
        const request = store.get(key);
        
        request.onsuccess = (event) => {
          if (event.target.result) {
            resolve(event.target.result.value);
          } else {
            resolve(null);
          }
        };
        
        request.onerror = (event) => {
          reject(event.target.error);
        };
      });
    } catch (error) {
      console.error('Database error:', error);
      throw error;
    }
  }
  
  async storeSetting(key, value) {
    return this.storeData(key, value);
  }
  
  async getSetting(key, defaultValue = null) {
    const value = await this.getData(key);
    return value !== null ? value : defaultValue;
  }
  
  registerEventListeners() {
    // Listen for the beforeinstallprompt event
    window.addEventListener('beforeinstallprompt', (event) => {
      // Prevent Chrome 67 and earlier from automatically showing the prompt
      event.preventDefault();
      
      // Stash the event so it can be triggered later
      this.installPrompt = event;
      this.updatePwaStatus('installPromptAvailable', true);
      
      console.log('Install prompt available');
    });
    
    // Listen for the appinstalled event
    window.addEventListener('appinstalled', (event) => {
      this.updatePwaStatus('installed', true);
      this.updatePwaStatus('installPromptAvailable', false);
      this.installPrompt = null;
      
      console.log('QUASAR QA³ PWA installed');
    });
    
    // Listen for online/offline events
    window.addEventListener('online', () => {
      this.updatePwaStatus('online', true);
      console.log('App is online');
    });
    
    window.addEventListener('offline', () => {
      this.updatePwaStatus('online', false);
      console.log('App is offline');
    });
    
    // Add display mode media query listener
    window.matchMedia('(display-mode: standalone)').addEventListener('change', (event) => {
      this.updatePwaStatus('pwaMode', event.matches);
      console.log('PWA mode:', event.matches);
    });
  }
  
  checkInstallState() {
    const isStandalone = window.matchMedia('(display-mode: standalone)').matches;
    const isIosStandalone = window.navigator.standalone;
    const isInstalled = isStandalone || isIosStandalone;
    
    this.updatePwaStatus('installed', isInstalled);
    this.updatePwaStatus('pwaMode', isInstalled);
  }
  
  async installPwa() {
    if (!this.installPrompt) {
      console.warn('Install prompt not available');
      return { success: false, message: 'Install prompt not available' };
    }
    
    // Show the install prompt
    this.installPrompt.prompt();
    
    // Wait for the user to respond to the prompt
    const choiceResult = await this.installPrompt.userChoice;
    
    // Clear the install prompt
    this.installPrompt = null;
    this.updatePwaStatus('installPromptAvailable', false);
    
    if (choiceResult.outcome === 'accepted') {
      console.log('User accepted the install prompt');
      return { success: true, message: 'Installation accepted' };
    } else {
      console.log('User dismissed the install prompt');
      return { success: false, message: 'Installation declined' };
    }
  }
  
  async updateSubscriptionStatus() {
    if (!this.status.notificationsSupported) {
      return;
    }
    
    // Check if we have permission
    const permission = Notification.permission;
    const notificationsEnabled = permission === 'granted';
    
    this.updatePwaStatus('notificationsEnabled', notificationsEnabled);
    this.updateNotificationUI();
  }
  
  updateNotificationUI() {
    // Update notification toggle button text
    const notificationToggle = document.getElementById('notification-toggle');
    if (notificationToggle) {
      notificationToggle.innerText = this.status.notificationsEnabled 
        ? 'Disable Notifications'
        : 'Enable Notifications';
    }
  }
  
  async toggleNotifications() {
    if (!this.status.notificationsSupported) {
      console.warn('Notifications not supported');
      return { success: false, message: 'Notifications not supported' };
    }
    
    if (this.status.notificationsEnabled) {
      // Can't actually revoke permission once granted, just update UI
      console.log('Notification permissions cannot be revoked once granted');
      
      return { 
        success: false, 
        message: 'Notification permissions cannot be revoked. Clear site data in browser settings to revoke.'
      };
    } else {
      // Request permission
      try {
        const permission = await Notification.requestPermission();
        const notificationsEnabled = permission === 'granted';
        
        this.updatePwaStatus('notificationsEnabled', notificationsEnabled);
        this.updateNotificationUI();
        
        if (notificationsEnabled) {
          // Send a test notification
          this.sendTestNotification();
          return { success: true, message: 'Notifications enabled' };
        } else {
          return { success: false, message: 'Notification permission denied' };
        }
      } catch (error) {
        console.error('Error requesting notification permission:', error);
        return { success: false, message: 'Error requesting permission', error };
      }
    }
  }
  
  async sendTestNotification() {
    if (!this.status.notificationsSupported || !this.status.notificationsEnabled) {
      console.warn('Notifications not supported or not enabled');
      return { success: false, message: 'Notifications not enabled' };
    }
    
    try {
      // Send via service worker if available
      if (this.swRegistration) {
        await this.swRegistration.showNotification('QUASAR QA³ Notification', {
          body: 'Notification system is working!',
          icon: '/icon-192.png',
          badge: '/icon-192.png',
          vibrate: [100, 50, 100],
          data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
          },
          actions: [
            {
              action: 'explore',
              title: 'View Quantum App',
              icon: '/icon-192.png'
            }
          ]
        });
      } else {
        // Fall back to regular notification
        new Notification('QUASAR QA³ Notification', {
          body: 'Notification system is working!',
          icon: '/icon-192.png'
        });
      }
      
      return { success: true, message: 'Test notification sent' };
    } catch (error) {
      console.error('Error sending notification:', error);
      return { success: false, message: 'Error sending notification', error };
    }
  }
  
  sendPwaStatusToStreamlit() {
    Object.entries(this.status).forEach(([key, value]) => {
      this.sendToStreamlit(key, value);
    });
  }
  
  updatePwaStatus(key, value) {
    if (this.status[key] !== value) {
      this.status[key] = value;
      this.sendToStreamlit(key, value);
    }
  }
  
  sendToStreamlit(key, value) {
    // Use Streamlit's component communication mechanism if available
    if (window.Streamlit) {
      const data = {
        pwaStatus: { [key]: value }
      };
      window.Streamlit.setComponentValue(data);
    }
    
    // Also dispatch a custom event
    window.parent.postMessage({
      type: 'pwaStatus',
      key,
      value
    }, '*');
  }
  
  // Utility function for web push
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

// Function to check if we're in PWA mode
function isPwaMode() {
  return window.matchMedia('(display-mode: standalone)').matches || 
         window.navigator.standalone ||
         document.referrer.includes('android-app://');
}

// Initialize the PWA system
window.quasarPwa = new QUASARPwa();
window.quasarPwa.init();

// Expose some methods globally
window.installQuasarPwa = window.quasarPwa.installPwa;
window.toggleQuasarNotifications = window.quasarPwa.toggleNotifications;
window.sendQuasarTestNotification = window.quasarPwa.sendTestNotification;