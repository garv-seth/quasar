
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
