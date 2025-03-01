// PWA Registration Script for QA³
// This script handles the registration and management of the PWA functionality

// Check if service workers are supported
if ('serviceWorker' in navigator) {
  window.addEventListener('load', function() {
    navigator.serviceWorker.register('/sw.js')
      .then(function(registration) {
        console.log('Service Worker registered successfully with scope: ', registration.scope);
        initializeDatabase();
        checkPwaStatus();
      })
      .catch(function(error) {
        console.error('Service Worker registration failed: ', error);
      });
  });
}

// Initialize the IndexedDB database for offline storage
function initializeDatabase() {
  const request = indexedDB.open('qa3-offline-db', 1);
  
  request.onerror = function(event) {
    console.error('Error opening IndexedDB database:', event.target.errorCode);
  };
  
  request.onupgradeneeded = function(event) {
    const db = event.target.result;
    
    // Create object stores if they don't exist
    if (!db.objectStoreNames.contains('searches')) {
      db.createObjectStore('searches', { keyPath: 'id', autoIncrement: true });
    }
    
    if (!db.objectStoreNames.contains('tasks')) {
      db.createObjectStore('tasks', { keyPath: 'id', autoIncrement: true });
    }
    
    if (!db.objectStoreNames.contains('storage')) {
      db.createObjectStore('storage', { keyPath: 'key' });
    }
    
    console.log('IndexedDB database initialized');
  };
  
  request.onsuccess = function(event) {
    console.log('IndexedDB database opened successfully');
    
    // Store a connection flag for offline detection
    storeData('connection_state', {
      lastOnline: new Date().toISOString(),
      status: 'online'
    });
  };
}

// Store a task for offline processing
async function storeTask(task) {
  try {
    const db = await openDatabase();
    const transaction = db.transaction(['tasks'], 'readwrite');
    const store = transaction.objectStore('tasks');
    
    // Add timestamp if not present
    if (!task.timestamp) {
      task.timestamp = new Date().toISOString();
    }
    
    const request = store.add(task);
    
    return new Promise((resolve, reject) => {
      request.onsuccess = function(event) {
        console.log('Task stored successfully for offline processing');
        resolve({ success: true, id: event.target.result });
      };
      
      request.onerror = function(event) {
        console.error('Error storing task:', event.target.error);
        reject({ success: false, error: event.target.error });
      };
    });
  } catch (error) {
    console.error('Error in storeTask:', error);
    return { success: false, error };
  }
}

// Store data in the storage object store
async function storeData(key, data) {
  try {
    const db = await openDatabase();
    const transaction = db.transaction(['storage'], 'readwrite');
    const store = transaction.objectStore('storage');
    
    const request = store.put({ key, value: data, timestamp: new Date().toISOString() });
    
    return new Promise((resolve, reject) => {
      request.onsuccess = function() {
        resolve({ success: true });
      };
      
      request.onerror = function(event) {
        reject({ success: false, error: event.target.error });
      };
    });
  } catch (error) {
    console.error('Error in storeData:', error);
    return { success: false, error };
  }
}

// Get data from the storage object store
async function getData(key) {
  try {
    const db = await openDatabase();
    const transaction = db.transaction(['storage'], 'readonly');
    const store = transaction.objectStore('storage');
    
    const request = store.get(key);
    
    return new Promise((resolve, reject) => {
      request.onsuccess = function() {
        if (request.result) {
          resolve({ success: true, data: request.result.value });
        } else {
          resolve({ success: false, error: 'Key not found' });
        }
      };
      
      request.onerror = function(event) {
        reject({ success: false, error: event.target.error });
      };
    });
  } catch (error) {
    console.error('Error in getData:', error);
    return { success: false, error };
  }
}

// Helper function to open IndexedDB
function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('qa3-offline-db', 1);
    
    request.onerror = function(event) {
      reject(event.target.error);
    };
    
    request.onsuccess = function(event) {
      resolve(event.target.result);
    };
  });
}

// Install PWA when the user clicks the install button
async function installPwa() {
  const installButton = document.getElementById('pwa-install-button');
  if (!installButton) return;
  
  let deferredPrompt;
  
  window.addEventListener('beforeinstallprompt', (e) => {
    // Prevent the mini-infobar from appearing on mobile
    e.preventDefault();
    // Stash the event so it can be triggered later
    deferredPrompt = e;
    // Update UI to notify the user they can install the PWA
    installButton.style.display = 'block';
    
    // Send status to Streamlit
    sendPwaStatusToStreamlit({ canInstall: true });
  });
  
  installButton.addEventListener('click', async () => {
    if (!deferredPrompt) {
      console.log('No installation prompt available');
      return;
    }
    
    // Show the install prompt
    deferredPrompt.prompt();
    
    // Wait for the user to respond to the prompt
    const { outcome } = await deferredPrompt.userChoice;
    console.log(`User ${outcome} the installation`);
    
    // We've used the prompt, and can't use it again, so clear it
    deferredPrompt = null;
    
    // Hide the install button
    installButton.style.display = 'none';
    
    // Send status to Streamlit
    sendPwaStatusToStreamlit({ canInstall: false, installed: outcome === 'accepted' });
  });
  
  // Listen for app installed event
  window.addEventListener('appinstalled', (event) => {
    console.log('PWA was installed');
    // Send status to Streamlit
    sendPwaStatusToStreamlit({ installed: true });
  });
}

// Request notification permission
async function requestNotificationPermission() {
  if (!('Notification' in window)) {
    console.log('This browser does not support notifications');
    return { success: false, error: 'Notifications not supported' };
  }
  
  try {
    const permission = await Notification.requestPermission();
    
    if (permission === 'granted') {
      console.log('Notification permission granted');
      return { success: true, permission };
    } else {
      console.log('Notification permission denied');
      return { success: false, permission };
    }
  } catch (error) {
    console.error('Error requesting notification permission:', error);
    return { success: false, error };
  }
}

// Send a test notification
function sendTestNotification() {
  if (!('Notification' in window)) {
    console.log('This browser does not support notifications');
    return;
  }
  
  if (Notification.permission === 'granted') {
    const notification = new Notification('QA³ Notification Test', {
      body: 'Your notification system is working correctly.',
      icon: '/icon-192.png'
    });
    
    notification.onclick = function() {
      window.focus();
      notification.close();
    };
  } else {
    console.log('Notification permission not granted');
    requestNotificationPermission();
  }
}

// Send PWA status information to Streamlit
function sendPwaStatusToStreamlit(status) {
  // This function uses Streamlit's component communication
  if (window.Streamlit) {
    window.Streamlit.setComponentValue({
      pwaStatus: {
        ...status,
        isStandalone: window.matchMedia('(display-mode: standalone)').matches,
        lastUpdated: new Date().toISOString()
      }
    });
  }
}

// Check if the app is running in standalone mode (installed PWA)
function checkPwaStatus() {
  const isStandalone = window.matchMedia('(display-mode: standalone)').matches || 
                      (window.navigator.standalone === true);
  
  if (isStandalone) {
    console.log('App is running as installed PWA');
    
    // Store PWA status in IndexedDB
    storeData('pwa_status', { 
      isInstalled: true, 
      installedAt: new Date().toISOString(),
      displayMode: 'standalone'
    });
    
    // Add PWA class to body for CSS targeting
    document.body.classList.add('pwa-mode');
  } else {
    console.log('App is running in browser');
  }
  
  // Listen for display mode changes
  window.matchMedia('(display-mode: standalone)').addEventListener('change', (event) => {
    if (event.matches) {
      console.log('App display mode changed to standalone');
      document.body.classList.add('pwa-mode');
    } else {
      console.log('App display mode changed to browser');
      document.body.classList.remove('pwa-mode');
    }
    
    // Send status to Streamlit
    sendPwaStatusToStreamlit({ isStandalone: event.matches });
  });
  
  // Send initial status to Streamlit
  sendPwaStatusToStreamlit({ 
    isStandalone,
    canInstall: false // Will be updated by beforeinstallprompt event
  });
}

// Monitor online/offline status
window.addEventListener('online', () => {
  console.log('App is online');
  storeData('connection_state', {
    lastOnline: new Date().toISOString(),
    status: 'online'
  });
  
  // Attempt to sync pending tasks
  if ('serviceWorker' in navigator && 'SyncManager' in window) {
    navigator.serviceWorker.ready
      .then(registration => {
        registration.sync.register('offline-task');
        registration.sync.register('offline-search');
      })
      .catch(error => {
        console.error('Error registering sync:', error);
      });
  }
});

window.addEventListener('offline', () => {
  console.log('App is offline');
  storeData('connection_state', {
    lastOffline: new Date().toISOString(),
    status: 'offline'
  });
});

// Initialize when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  installPwa();
  
  // Initialize notification permission button
  const notificationButton = document.getElementById('notification-permission-button');
  if (notificationButton) {
    notificationButton.addEventListener('click', async () => {
      const result = await requestNotificationPermission();
      if (result.success) {
        sendTestNotification();
      }
    });
  }
});