/**
 * QUASAR QA³: PWA Implementation
 * Provides Progressive Web App functionality for the Quantum-Accelerated AI Agent
 */

class QUASARPwa {
  constructor() {
    this.db = null;
    this.swRegistration = null;
    this.isInstalled = false;
    this.notificationsEnabled = false;
    this.installPrompt = null;
    this.statusUpdates = {};
    
    // Check if running in a PWA context
    this.isPwa = window.matchMedia('(display-mode: standalone)').matches || 
                window.navigator.standalone || 
                document.referrer.includes('android-app://');
    
    // Initialize when the page loads
    window.addEventListener('load', () => this.init());
  }
  
  async init() {
    console.log('Initializing QUASAR PWA system');
    
    // Register service worker if supported
    if ('serviceWorker' in navigator) {
      await this.initializeServiceWorker();
    } else {
      console.warn('Service Worker is not supported in this browser');
      this.updatePwaStatus('serviceWorkerSupported', false);
    }
    
    // Initialize IndexedDB for offline storage
    if ('indexedDB' in window) {
      await this.initializeDatabase();
    } else {
      console.warn('IndexedDB is not supported in this browser');
      this.updatePwaStatus('databaseSupported', false);
    }
    
    // Register event listeners for install prompt and other events
    this.registerEventListeners();
    
    // Check if already installed as PWA
    this.checkInstallState();
    
    // Update notification subscription status
    await this.updateSubscriptionStatus();
    
    // Send initial status to Streamlit
    this.sendPwaStatusToStreamlit();
  }
  
  async initializeServiceWorker() {
    try {
      this.swRegistration = await navigator.serviceWorker.register('/sw.js');
      console.log('Service Worker registered successfully:', this.swRegistration);
      this.updatePwaStatus('serviceWorkerRegistered', true);
      
      // Check if service worker is active
      if (this.swRegistration.active) {
        console.log('Service Worker is active');
        this.updatePwaStatus('serviceWorkerActive', true);
      } else {
        this.swRegistration.addEventListener('updatefound', () => {
          const newWorker = this.swRegistration.installing;
          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'activated') {
              console.log('Service Worker activated');
              this.updatePwaStatus('serviceWorkerActive', true);
              this.sendPwaStatusToStreamlit();
            }
          });
        });
      }
      
      // Check if pushManager is available (needed for notifications)
      if ('pushManager' in this.swRegistration) {
        this.updatePwaStatus('pushManagerSupported', true);
      } else {
        console.warn('Push Manager is not supported');
        this.updatePwaStatus('pushManagerSupported', false);
      }
    } catch (error) {
      console.error('Service Worker registration failed:', error);
      this.updatePwaStatus('serviceWorkerRegistered', false);
      this.updatePwaStatus('serviceWorkerActive', false);
    }
  }
  
  async initializeDatabase() {
    try {
      this.db = await this.openDatabase();
      console.log('IndexedDB initialized successfully');
      this.updatePwaStatus('databaseInitialized', true);
    } catch (error) {
      console.error('Failed to initialize IndexedDB:', error);
      this.updatePwaStatus('databaseInitialized', false);
    }
  }
  
  openDatabase() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('QUASAR_QA3_DB', 1);
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        
        // Create object stores if they don't exist
        if (!db.objectStoreNames.contains('offlineRequests')) {
          db.createObjectStore('offlineRequests', { keyPath: 'timestamp' });
        }
        
        if (!db.objectStoreNames.contains('quantumTasks')) {
          db.createObjectStore('quantumTasks', { keyPath: 'id' });
        }
        
        if (!db.objectStoreNames.contains('settings')) {
          db.createObjectStore('settings', { keyPath: 'key' });
        }
      };
      
      request.onsuccess = (event) => {
        resolve(event.target.result);
      };
      
      request.onerror = (event) => {
        console.error('IndexedDB error:', event.target.error);
        reject(event.target.error);
      };
    });
  }
  
  async storeTask(task) {
    if (!this.db) {
      console.error('Database not initialized');
      return false;
    }
    
    try {
      const tx = this.db.transaction('quantumTasks', 'readwrite');
      const store = tx.objectStore('quantumTasks');
      
      // Add the task
      await store.put(task);
      
      // Complete the transaction
      await new Promise((resolve, reject) => {
        tx.oncomplete = resolve;
        tx.onerror = reject;
      });
      
      console.log('Task stored successfully:', task.id);
      return true;
    } catch (error) {
      console.error('Failed to store task:', error);
      return false;
    }
  }
  
  async storeData(key, data) {
    if (!this.db) {
      console.error('Database not initialized');
      return false;
    }
    
    try {
      const tx = this.db.transaction('settings', 'readwrite');
      const store = tx.objectStore('settings');
      
      // Add the data
      await store.put({ key, value: data, timestamp: Date.now() });
      
      // Complete the transaction
      await new Promise((resolve, reject) => {
        tx.oncomplete = resolve;
        tx.onerror = reject;
      });
      
      console.log(`Data stored successfully for key: ${key}`);
      return true;
    } catch (error) {
      console.error(`Failed to store data for key ${key}:`, error);
      return false;
    }
  }
  
  async getData(key) {
    if (!this.db) {
      console.error('Database not initialized');
      return null;
    }
    
    try {
      const tx = this.db.transaction('settings', 'readonly');
      const store = tx.objectStore('settings');
      
      // Get the data
      const data = await new Promise((resolve, reject) => {
        const request = store.get(key);
        request.onsuccess = () => resolve(request.result);
        request.onerror = reject;
      });
      
      return data ? data.value : null;
    } catch (error) {
      console.error(`Failed to get data for key ${key}:`, error);
      return null;
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
    // Listen for beforeinstallprompt event to capture install prompt
    window.addEventListener('beforeinstallprompt', (event) => {
      // Prevent the default prompt
      event.preventDefault();
      
      // Save the event for later use
      this.installPrompt = event;
      
      // Update status
      this.updatePwaStatus('installPromptAvailable', true);
      this.sendPwaStatusToStreamlit();
      
      console.log('Install prompt is available');
    });
    
    // Listen for appinstalled event to know when PWA is installed
    window.addEventListener('appinstalled', () => {
      this.isInstalled = true;
      this.installPrompt = null;
      
      this.updatePwaStatus('installed', true);
      this.updatePwaStatus('installPromptAvailable', false);
      this.sendPwaStatusToStreamlit();
      
      console.log('PWA was installed');
    });
    
    // Listen for online/offline events
    window.addEventListener('online', () => {
      this.updatePwaStatus('online', true);
      this.sendPwaStatusToStreamlit();
      console.log('App is online');
      
      // Try to sync pending tasks if service worker is active
      if (this.swRegistration && this.swRegistration.active) {
        this.swRegistration.sync.register('sync-quantum-tasks')
          .catch(error => console.error('Failed to register sync:', error));
      }
    });
    
    window.addEventListener('offline', () => {
      this.updatePwaStatus('online', false);
      this.sendPwaStatusToStreamlit();
      console.log('App is offline');
    });
    
    // Listen for messages from service worker
    navigator.serviceWorker.addEventListener('message', (event) => {
      console.log('Message from Service Worker:', event.data);
      
      // Handle different message types
      if (event.data.type === 'syncComplete') {
        this.sendToStreamlit('syncComplete', event.data.tasks);
      }
    });
    
    // Listen for display-mode changes
    window.matchMedia('(display-mode: standalone)').addEventListener('change', (event) => {
      this.isPwa = event.matches;
      this.updatePwaStatus('pwaMode', this.isPwa);
      this.sendPwaStatusToStreamlit();
    });
  }
  
  checkInstallState() {
    // Check if the app is already installed as a PWA
    this.isPwa = window.matchMedia('(display-mode: standalone)').matches || 
               window.navigator.standalone || 
               document.referrer.includes('android-app://');
    
    if (this.isPwa) {
      this.isInstalled = true;
      this.updatePwaStatus('installed', true);
      this.updatePwaStatus('pwaMode', true);
    } else {
      this.updatePwaStatus('pwaMode', false);
    }
  }
  
  async installPwa() {
    // Check if install prompt is available
    if (!this.installPrompt) {
      console.warn('Install prompt not available');
      return false;
    }
    
    try {
      // Show the install prompt
      this.installPrompt.prompt();
      
      // Wait for the user to respond to the prompt
      const choiceResult = await this.installPrompt.userChoice;
      
      if (choiceResult.outcome === 'accepted') {
        console.log('User accepted the PWA installation');
        this.isInstalled = true;
        this.installPrompt = null;
        
        this.updatePwaStatus('installed', true);
        this.updatePwaStatus('installPromptAvailable', false);
        
        return true;
      } else {
        console.log('User declined the PWA installation');
        return false;
      }
    } catch (error) {
      console.error('Error during PWA installation:', error);
      return false;
    } finally {
      this.sendPwaStatusToStreamlit();
    }
  }
  
  async updateSubscriptionStatus() {
    if (!this.swRegistration || !('pushManager' in this.swRegistration)) {
      this.updatePwaStatus('notificationsSupported', false);
      return;
    }
    
    try {
      const subscription = await this.swRegistration.pushManager.getSubscription();
      this.notificationsEnabled = !!subscription;
      
      this.updatePwaStatus('notificationsSupported', true);
      this.updatePwaStatus('notificationsEnabled', this.notificationsEnabled);
      
      this.updateNotificationUI();
    } catch (error) {
      console.error('Error checking notification status:', error);
      this.updatePwaStatus('notificationsSupported', false);
    }
  }
  
  updateNotificationUI() {
    // Find notification toggle button if it exists
    const notificationToggle = document.getElementById('notification-toggle');
    if (notificationToggle) {
      notificationToggle.textContent = this.notificationsEnabled ? 
        'Disable Notifications' : 'Enable Notifications';
    }
  }
  
  async toggleNotifications() {
    if (!this.swRegistration || !('pushManager' in this.swRegistration)) {
      console.warn('Push notifications are not supported');
      return false;
    }
    
    try {
      if (this.notificationsEnabled) {
        // Unsubscribe from push notifications
        const subscription = await this.swRegistration.pushManager.getSubscription();
        if (subscription) {
          await subscription.unsubscribe();
          this.notificationsEnabled = false;
          console.log('Notifications disabled');
        }
      } else {
        // Request permission for notifications
        const permission = await Notification.requestPermission();
        if (permission !== 'granted') {
          console.warn('Notification permission denied');
          return false;
        }
        
        // Get the server's public key (would normally come from your server)
        // For demo purposes, we'll use a dummy key
        const applicationServerKey = this.urlB64ToUint8Array(
          'BEl62iUYgUivxIkv69yViEuiBIa-Ib9-SkvMeAtA3LFgDzkrxZJjSgSnfckjBJuBkr3qBUYIHBQFLXYp5Nksh8U'
        );
        
        // Subscribe to push notifications
        const subscription = await this.swRegistration.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey: applicationServerKey
        });
        
        this.notificationsEnabled = true;
        console.log('Notifications enabled');
        
        // In a real app, you would send the subscription to your server
        console.log('Notification subscription:', JSON.stringify(subscription));
      }
      
      this.updatePwaStatus('notificationsEnabled', this.notificationsEnabled);
      this.updateNotificationUI();
      this.sendPwaStatusToStreamlit();
      
      return true;
    } catch (error) {
      console.error('Error toggling notifications:', error);
      return false;
    }
  }
  
  async sendTestNotification() {
    if (!('Notification' in window)) {
      console.warn('Notifications not supported');
      return false;
    }
    
    if (Notification.permission !== 'granted') {
      const permission = await Notification.requestPermission();
      if (permission !== 'granted') {
        console.warn('Notification permission denied');
        return false;
      }
    }
    
    // Create and show a notification
    const notification = new Notification('QUASAR QA³', {
      body: 'This is a test notification from the Quantum-Accelerated AI Agent',
      icon: '/icon-192.png',
      badge: '/icon-192.png'
    });
    
    notification.onclick = () => {
      console.log('Notification clicked');
      window.focus();
      notification.close();
    };
    
    return true;
  }
  
  sendPwaStatusToStreamlit() {
    // Send the entire status object to Streamlit
    this.sendToStreamlit('pwaStatus', this.statusUpdates);
  }
  
  updatePwaStatus(key, value) {
    this.statusUpdates[key] = value;
  }
  
  sendToStreamlit(key, value) {
    // Check if Streamlit is available
    if (window.Streamlit && window.Streamlit.setComponentValue) {
      window.Streamlit.setComponentValue({
        [key]: value
      });
    } else {
      console.log('Streamlit not available, status update:', key, value);
    }
  }
  
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

// Initialize the QUASAR PWA system
const quasarPwa = new QUASARPwa();

// Export for use in the application
window.quasarPwa = quasarPwa;

// Function to check if we're in PWA mode
function isPwaMode() {
  return window.matchMedia('(display-mode: standalone)').matches || 
         window.navigator.standalone || 
         document.referrer.includes('android-app://');
}

// Add a global method to install the PWA
window.installQuasarPwa = () => quasarPwa.installPwa();

// Add a global method to toggle notifications
window.toggleQuasarNotifications = () => quasarPwa.toggleNotifications();

// Add a global method to send a test notification
window.sendQuasarTestNotification = () => quasarPwa.sendTestNotification();