/**
 * QA³ Service Worker
 * Provides offline capabilities, background sync, and push notifications
 */

const CACHE_NAME = 'qa3-cache-v1';
const OFFLINE_URL = '/offline.html';
const ASSETS_TO_CACHE = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png',
  '/screenshot1.png',
  '/offline.html',
  '/pwa.js',
  '/browser_implementation.js',
  '/static/style.css',
  '/static/main.js'
];

// Install event - cache assets
self.addEventListener('install', event => {
  console.log('[ServiceWorker] Installing Service Worker...');
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('[ServiceWorker] Caching app shell and assets');
        return cache.addAll(ASSETS_TO_CACHE);
      })
      .then(() => {
        console.log('[ServiceWorker] Install completed');
        return self.skipWaiting();
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('[ServiceWorker] Activating Service Worker...');
  
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.filter(cacheName => {
          return cacheName !== CACHE_NAME;
        }).map(cacheName => {
          console.log('[ServiceWorker] Removing old cache:', cacheName);
          return caches.delete(cacheName);
        })
      );
    }).then(() => {
      console.log('[ServiceWorker] Activation completed');
      return self.clients.claim();
    })
  );
});

// Fetch event - serve from cache or fetch from network
self.addEventListener('fetch', event => {
  console.log('[ServiceWorker] Fetch:', event.request.url);
  
  // Skip cross-origin requests
  if (!event.request.url.startsWith(self.location.origin)) {
    return;
  }
  
  // API requests - network first, then cache
  if (event.request.url.includes('/api/')) {
    handleApiRequest(event);
    return;
  }
  
  // HTML requests - network first, then cache, then offline page
  if (event.request.mode === 'navigate' || 
      (event.request.method === 'GET' && 
       event.request.headers.get('accept').includes('text/html'))) {
    handleNavigationRequest(event);
    return;
  }
  
  // Assets - cache first, then network
  handleAssetRequest(event);
});

// Handle API requests
function handleApiRequest(event) {
  event.respondWith(
    fetch(event.request)
      .then(response => {
        // Clone the response for caching
        const responseToCache = response.clone();
        
        caches.open(CACHE_NAME).then(cache => {
          // Don't cache errors or non-successful responses
          if (response.ok) {
            cache.put(event.request, responseToCache);
          }
        });
        
        return response;
      })
      .catch(() => {
        // Try to get from cache if network fails
        return caches.match(event.request);
      })
  );
}

// Handle navigation requests
function handleNavigationRequest(event) {
  event.respondWith(
    fetch(event.request)
      .then(response => {
        // Clone the response for caching
        const responseToCache = response.clone();
        
        caches.open(CACHE_NAME).then(cache => {
          cache.put(event.request, responseToCache);
        });
        
        return response;
      })
      .catch(() => {
        return caches.match(event.request)
          .then(cachedResponse => {
            // Return cached response if available
            if (cachedResponse) {
              return cachedResponse;
            }
            
            // Return offline page if nothing else available
            return caches.match(OFFLINE_URL);
          });
      })
  );
}

// Handle asset requests
function handleAssetRequest(event) {
  event.respondWith(
    caches.match(event.request)
      .then(cachedResponse => {
        // Return cached response if available
        if (cachedResponse) {
          return cachedResponse;
        }
        
        // Otherwise fetch from network
        return fetch(event.request)
          .then(response => {
            // Clone the response for caching
            const responseToCache = response.clone();
            
            caches.open(CACHE_NAME).then(cache => {
              // Cache successful responses
              if (response.ok) {
                cache.put(event.request, responseToCache);
              }
            });
            
            return response;
          });
      })
  );
}

// Background sync for offline tasks
self.addEventListener('sync', event => {
  console.log('[ServiceWorker] Background Sync:', event.tag);
  
  if (event.tag === 'sync-tasks') {
    event.waitUntil(syncOfflineTasks());
  } else if (event.tag === 'sync-searches') {
    event.waitUntil(syncOfflineSearches());
  }
});

// Push notification handler
self.addEventListener('push', event => {
  console.log('[ServiceWorker] Push received:', event);
  
  const data = event.data ? event.data.json() : {};
  const title = data.title || 'QA³ Notification';
  const options = {
    body: data.body || 'Something happened in QA³',
    icon: '/icon-192.png',
    badge: '/icon-192.png',
    data: data.data || {}
  };
  
  event.waitUntil(
    self.registration.showNotification(title, options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', event => {
  console.log('[ServiceWorker] Notification click:', event);
  
  event.notification.close();
  
  const urlToOpen = event.notification.data.url || '/';
  
  event.waitUntil(
    clients.matchAll({
      type: 'window'
    })
    .then(windowClients => {
      // Check if there is already a window/tab open with the target URL
      for (let i = 0; i < windowClients.length; i++) {
        const client = windowClients[i];
        if (client.url === urlToOpen && 'focus' in client) {
          return client.focus();
        }
      }
      
      // If no open window/tab, open a new one
      if (clients.openWindow) {
        return clients.openWindow(urlToOpen);
      }
    })
  );
});

// Sync offline tasks
async function syncOfflineTasks() {
  try {
    // Open IndexedDB database
    const db = await openDatabase();
    const tx = db.transaction('offlineTasks', 'readwrite');
    const store = tx.objectStore('offlineTasks');
    
    // Get all stored tasks
    const tasks = await store.getAll();
    
    // Process tasks
    for (const task of tasks) {
      try {
        // Send task to server
        const response = await fetch('/api/tasks', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(task)
        });
        
        if (response.ok) {
          // Delete task from IndexedDB if successfully sent
          await store.delete(task.id);
        }
      } catch (err) {
        console.error('[ServiceWorker] Error syncing task:', err);
      }
    }
    
    await tx.complete;
    console.log('[ServiceWorker] Tasks sync completed');
  } catch (err) {
    console.error('[ServiceWorker] Error in syncOfflineTasks:', err);
  }
}

// Sync offline searches
async function syncOfflineSearches() {
  try {
    // Open IndexedDB database
    const db = await openDatabase();
    const tx = db.transaction('offlineSearches', 'readwrite');
    const store = tx.objectStore('offlineSearches');
    
    // Get all stored searches
    const searches = await store.getAll();
    
    // Process searches
    for (const search of searches) {
      try {
        // Send search to server
        const response = await fetch('/api/searches', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(search)
        });
        
        if (response.ok) {
          // Delete search from IndexedDB if successfully sent
          await store.delete(search.id);
        }
      } catch (err) {
        console.error('[ServiceWorker] Error syncing search:', err);
      }
    }
    
    await tx.complete;
    console.log('[ServiceWorker] Searches sync completed');
  } catch (err) {
    console.error('[ServiceWorker] Error in syncOfflineSearches:', err);
  }
}

// Open IndexedDB database
function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('qa3-offline-db', 1);
    
    request.onerror = event => {
      reject(new Error('Failed to open database'));
    };
    
    request.onsuccess = event => {
      resolve(event.target.result);
    };
    
    request.onupgradeneeded = event => {
      const db = event.target.result;
      
      // Create stores if they don't exist
      if (!db.objectStoreNames.contains('offlineTasks')) {
        db.createObjectStore('offlineTasks', { keyPath: 'id' });
      }
      
      if (!db.objectStoreNames.contains('offlineSearches')) {
        db.createObjectStore('offlineSearches', { keyPath: 'id' });
      }
    };
  });
}

// Periodic background sync (if supported)
self.addEventListener('periodicsync', event => {
  console.log('[ServiceWorker] Periodic Sync:', event.tag);
  
  if (event.tag === 'update-content') {
    event.waitUntil(updateContent());
  }
});

// Update content in the background
async function updateContent() {
  try {
    // Fetch latest data
    const response = await fetch('/api/content/latest');
    
    if (response.ok) {
      const data = await response.json();
      
      // Store in cache
      const cache = await caches.open(CACHE_NAME);
      
      // Store the data
      await cache.put('/api/content/latest', new Response(JSON.stringify(data)));
      
      // Notify clients
      const clients = await self.clients.matchAll();
      clients.forEach(client => {
        client.postMessage({
          type: 'CONTENT_UPDATED',
          data: data
        });
      });
    }
  } catch (err) {
    console.error('[ServiceWorker] Error updating content:', err);
  }
}