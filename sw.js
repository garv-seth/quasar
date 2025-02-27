/**
 * QUASAR QA³: Service Worker
 * Provides offline capabilities and resource caching for the PWA
 */

const CACHE_NAME = 'quasar-qa3-v1';
const OFFLINE_URL = 'offline.html';

// Resources that should be pre-cached (important app files)
const PRE_CACHE_RESOURCES = [
  '/',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png',
  '/screenshot1.png',
  '/offline.html',
  '/pwa.js'
];

// Resources that should use a cache-first strategy
const CACHE_FIRST_RESOURCES = [
  '/icon-192.png',
  '/icon-512.png',
  '/screenshot1.png',
  '/manifest.json'
];

// Installation event - cache critical assets
self.addEventListener('install', event => {
  console.log('[Service Worker] Installation started');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('[Service Worker] Pre-caching offline resources');
        return cache.addAll(PRE_CACHE_RESOURCES);
      })
      .then(() => {
        console.log('[Service Worker] Installation completed');
        return self.skipWaiting();
      })
  );
});

// Activation event - clean up old caches
self.addEventListener('activate', event => {
  console.log('[Service Worker] Activation started');
  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames
            .filter(cacheName => cacheName !== CACHE_NAME)
            .map(cacheName => {
              console.log('[Service Worker] Removing old cache:', cacheName);
              return caches.delete(cacheName);
            })
        );
      })
      .then(() => {
        console.log('[Service Worker] Activation completed');
        return self.clients.claim();
      })
  );
});

// Fetch event - handle resource requests
self.addEventListener('fetch', event => {
  // Skip non-GET requests
  if (event.request.method !== 'GET') return;
  
  // Handle API calls separately
  if (event.request.url.includes('/api/')) {
    return handleApiRequest(event);
  }
  
  // Determine the caching strategy based on the request
  if (CACHE_FIRST_RESOURCES.some(resource => event.request.url.endsWith(resource))) {
    // Cache-first for static assets like images and the manifest
    event.respondWith(cacheFirstStrategy(event.request));
  } else {
    // Network-first for everything else (dynamic content)
    event.respondWith(networkFirstStrategy(event.request));
  }
});

// Cache-first strategy for static resources
async function cacheFirstStrategy(request) {
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }
  
  try {
    const networkResponse = await fetch(request);
    if (networkResponse && networkResponse.status === 200) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.error('[Service Worker] Cache-first fetch failed:', error);
    return new Response(
      'Network error. Please check your connection.',
      { status: 408, headers: { 'Content-Type': 'text/plain' } }
    );
  }
}

// Network-first strategy for dynamic content
async function networkFirstStrategy(request) {
  try {
    // Try network first
    const networkResponse = await fetch(request);
    if (networkResponse && networkResponse.status === 200) {
      // Cache the successful response
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
      return networkResponse;
    }
    throw new Error('Network response was not valid');
  } catch (error) {
    console.log('[Service Worker] Network request failed, using cache:', error);
    
    // If network fails, try from cache
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // If not in cache and it's a page navigation, show the offline page
    if (request.mode === 'navigate') {
      const cache = await caches.open(CACHE_NAME);
      return cache.match(OFFLINE_URL);
    }
    
    // Otherwise return an error
    return new Response(
      'Network error and resource not in cache.',
      { status: 404, headers: { 'Content-Type': 'text/plain' } }
    );
  }
}

// Handle API requests, with background sync for offline usage
function handleApiRequest(event) {
  // Attempt online operation first
  event.respondWith(
    fetch(event.request.clone())
      .catch(error => {
        // If offline, store the request for later sync
        return storeRequestForSync(event.request)
          .then(() => {
            return new Response(
              JSON.stringify({ 
                offline: true, 
                message: 'Your request was saved and will be processed when you are back online.' 
              }),
              { 
                status: 503, 
                headers: { 'Content-Type': 'application/json' } 
              }
            );
          });
      })
  );
}

// Store a request for later sync
async function storeRequestForSync(request) {
  // Clone the request to read its body
  const requestClone = request.clone();
  let requestData;
  
  try {
    // Get the request data
    const body = await requestClone.text();
    requestData = {
      url: request.url,
      method: request.method,
      headers: Array.from(request.headers.entries()),
      body: body,
      timestamp: Date.now()
    };
    
    // Open the database
    const db = await openDatabase();
    
    // Add the request to the offline queue
    const tx = db.transaction('offlineRequests', 'readwrite');
    const store = tx.objectStore('offlineRequests');
    await store.add(requestData);
    await tx.complete;
    
    console.log('[Service Worker] Request stored for later sync:', requestData.url);
    
    // Register a sync if supported
    if ('sync' in self.registration) {
      await self.registration.sync.register('sync-quantum-tasks');
    }
    
    return true;
  } catch (error) {
    console.error('[Service Worker] Failed to store request:', error);
    return false;
  }
}

// Open the IndexedDB database
function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('QUASAR_QA3_DB', 1);
    
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('offlineRequests')) {
        db.createObjectStore('offlineRequests', { keyPath: 'timestamp' });
      }
      if (!db.objectStoreNames.contains('quantumTasks')) {
        db.createObjectStore('quantumTasks', { keyPath: 'id' });
      }
    };
    
    request.onsuccess = (event) => {
      resolve(event.target.result);
    };
    
    request.onerror = (event) => {
      console.error('[Service Worker] Database error:', event.target.error);
      reject(event.target.error);
    };
  });
}

// Handle sync event
self.addEventListener('sync', event => {
  if (event.tag === 'sync-quantum-tasks') {
    event.waitUntil(syncQuantumTasks());
  }
});

// Process stored requests when online
async function syncQuantumTasks() {
  console.log('[Service Worker] Syncing quantum tasks');
  try {
    const db = await openDatabase();
    const tasks = await getOfflineTasks(db);
    
    console.log(`[Service Worker] Found ${tasks.length} tasks to sync`);
    
    // Process each stored request
    for (const task of tasks) {
      try {
        const response = await fetch(task.url, {
          method: task.method,
          headers: task.headers.reduce((obj, [key, value]) => {
            obj[key] = value;
            return obj;
          }, {}),
          body: task.method !== 'GET' ? task.body : undefined
        });
        
        if (response.ok) {
          // Request succeeded, remove from store
          const tx = db.transaction('offlineRequests', 'readwrite');
          const store = tx.objectStore('offlineRequests');
          await store.delete(task.timestamp);
          await tx.complete;
          console.log('[Service Worker] Successfully synced task:', task.url);
          
          // Show notification if supported
          if ('Notification' in self) {
            self.registration.showNotification('QUASAR QA³', {
              body: 'Your quantum task has been processed successfully.',
              icon: '/icon-192.png',
              badge: '/icon-192.png'
            });
          }
        }
      } catch (error) {
        console.error('[Service Worker] Failed to sync task:', task.url, error);
      }
    }
    
    return true;
  } catch (error) {
    console.error('[Service Worker] Failed to sync quantum tasks:', error);
    return false;
  }
}

// Get stored offline tasks
async function getOfflineTasks(db) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('offlineRequests', 'readonly');
    const store = tx.objectStore('offlineRequests');
    const request = store.getAll();
    
    request.onsuccess = () => {
      resolve(request.result);
    };
    
    request.onerror = (event) => {
      reject(event.target.error);
    };
  });
}

// Handle push notifications
self.addEventListener('push', event => {
  if (!event.data) {
    console.warn('[Service Worker] Push received but no data');
    return;
  }
  
  let notification;
  try {
    notification = event.data.json();
  } catch (e) {
    notification = {
      title: 'QUASAR QA³',
      body: event.data.text(),
      icon: '/icon-192.png'
    };
  }
  
  event.waitUntil(
    self.registration.showNotification(notification.title, {
      body: notification.body,
      icon: notification.icon || '/icon-192.png',
      badge: '/icon-192.png',
      data: notification.data
    })
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', event => {
  event.notification.close();
  
  let url = '/';
  if (event.notification.data && event.notification.data.url) {
    url = event.notification.data.url;
  }
  
  event.waitUntil(
    clients.matchAll({ type: 'window' })
      .then(windowClients => {
        // Check if there is already a window open
        for (const client of windowClients) {
          if (client.url === url && 'focus' in client) {
            return client.focus();
          }
        }
        
        // If no window is open, open a new one
        if (clients.openWindow) {
          return clients.openWindow(url);
        }
      })
  );
});

console.log('[Service Worker] Script loaded');