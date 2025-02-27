// QUASAR QA³ Service Worker

// Cache version - update when files change
const CACHE_VERSION = 'quasar-qa3-v1';
const DYNAMIC_CACHE = 'quasar-dynamic-v1';

// Files to cache
const STATIC_CACHE_URLS = [
  '/',
  '/offline.html',
  '/icon-192.png',
  '/icon-512.png',
  '/pwa.js',
  '/manifest.json'
];

// IndexedDB helpers
const idb = {
  get dbPromise() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('QUASAR-QA3-DB', 1);
      
      request.onupgradeneeded = event => {
        const db = event.target.result;
        
        // Create stores if they don't exist
        if (!db.objectStoreNames.contains('taskQueue')) {
          db.createObjectStore('taskQueue', { keyPath: 'id', autoIncrement: true });
        }
        
        if (!db.objectStoreNames.contains('dataCache')) {
          db.createObjectStore('dataCache', { keyPath: 'key' });
        }
        
        if (!db.objectStoreNames.contains('settings')) {
          db.createObjectStore('settings', { keyPath: 'key' });
        }
      };
      
      request.onsuccess = event => resolve(event.target.result);
      request.onerror = event => reject(event.target.error);
    });
  },
  
  async add(storeName, item) {
    const db = await this.dbPromise;
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      const request = store.add(item);
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      tx.oncomplete = () => db.close();
    });
  },
  
  async get(storeName, key) {
    const db = await this.dbPromise;
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readonly');
      const store = tx.objectStore(storeName);
      const request = store.get(key);
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      tx.oncomplete = () => db.close();
    });
  },
  
  async getAll(storeName) {
    const db = await this.dbPromise;
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readonly');
      const store = tx.objectStore(storeName);
      const request = store.getAll();
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      tx.oncomplete = () => db.close();
    });
  },
  
  async delete(storeName, key) {
    const db = await this.dbPromise;
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      const request = store.delete(key);
      
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
      tx.oncomplete = () => db.close();
    });
  },
  
  async clear(storeName) {
    const db = await this.dbPromise;
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      const request = store.clear();
      
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
      tx.oncomplete = () => db.close();
    });
  }
};

// Install service worker
self.addEventListener('install', event => {
  console.log('[Service Worker] Installing Service Worker...', event);
  
  // Skip waiting to make sure the new service worker activates immediately
  self.skipWaiting();
  
  // Cache static assets
  event.waitUntil(
    caches.open(CACHE_VERSION)
      .then(cache => {
        console.log('[Service Worker] Precaching App Shell');
        return cache.addAll(STATIC_CACHE_URLS);
      })
      .catch(error => {
        console.error('[Service Worker] Precaching failed:', error);
      })
  );
});

// Activate service worker
self.addEventListener('activate', event => {
  console.log('[Service Worker] Activating Service Worker...', event);
  
  // Clean up old caches
  event.waitUntil(
    caches.keys()
      .then(keyList => {
        return Promise.all(keyList.map(key => {
          if (key !== CACHE_VERSION && key !== DYNAMIC_CACHE) {
            console.log('[Service Worker] Removing old cache:', key);
            return caches.delete(key);
          }
        }));
      })
  );
  
  // Take control of all clients immediately
  return self.clients.claim();
});

// Intercept fetch requests
self.addEventListener('fetch', event => {
  // Skip cross-origin requests
  if (!event.request.url.startsWith(self.location.origin)) {
    return;
  }
  
  // Skip non-GET requests
  if (event.request.method !== 'GET') {
    return;
  }
  
  // Handle API requests differently
  if (event.request.url.includes('/api/')) {
    handleApiRequest(event);
    return;
  }
  
  // For normal navigation requests, use stale-while-revalidate strategy
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Return cached response if available
        if (response) {
          // Fetch updated resource in the background
          fetch(event.request)
            .then(networkResponse => {
              caches.open(CACHE_VERSION)
                .then(cache => {
                  cache.put(event.request, networkResponse.clone());
                });
            })
            .catch(error => {
              console.log('[Service Worker] Network request failed:', error);
            });
          
          return response;
        }
        
        // If not in cache, fetch from network
        return fetch(event.request)
          .then(networkResponse => {
            // Cache the response for future use
            caches.open(DYNAMIC_CACHE)
              .then(cache => {
                cache.put(event.request, networkResponse.clone());
              });
            
            return networkResponse;
          })
          .catch(error => {
            console.log('[Service Worker] Network request failed:', error);
            
            // For HTML navigation requests, return the offline page
            if (event.request.headers.get('accept').includes('text/html')) {
              return caches.match('/offline.html');
            }
            
            // Otherwise, just fail
            return new Response('Network error', {
              status: 408,
              headers: { 'Content-Type': 'text/plain' }
            });
          });
      })
  );
});

// Handle API requests
function handleApiRequest(event) {
  event.respondWith(
    fetch(event.request)
      .then(response => response)
      .catch(error => {
        console.log('[Service Worker] API request failed:', error);
        
        // Store offline request for later synchronization
        return serializeRequest(event.request)
          .then(serializedRequest => {
            return idb.add('taskQueue', {
              request: serializedRequest,
              timestamp: new Date().toISOString()
            }).then(() => {
              // Register for background sync if supported
              if ('SyncManager' in self) {
                return self.registration.sync.register('quantum-task-sync');
              }
            }).then(() => {
              // Return a basic response indicating offline storage
              return new Response(JSON.stringify({
                success: false,
                offline: true,
                message: 'Request stored for processing when online'
              }), {
                headers: { 'Content-Type': 'application/json' }
              });
            });
          });
      })
  );
}

// Serialize a request for storage
async function serializeRequest(request) {
  const headers = {};
  for (const [key, value] of request.headers.entries()) {
    headers[key] = value;
  }
  
  let body = null;
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    body = await request.clone().text();
  }
  
  return {
    url: request.url,
    method: request.method,
    headers: headers,
    body: body,
    mode: request.mode,
    credentials: request.credentials,
    cache: request.cache,
    redirect: request.redirect,
    referrer: request.referrer
  };
}

// Process background sync
self.addEventListener('sync', event => {
  console.log('[Service Worker] Background Sync:', event);
  
  if (event.tag === 'quantum-task-sync') {
    event.waitUntil(
      idb.getAll('taskQueue')
        .then(tasks => {
          return Promise.all(tasks.map(task => {
            // Deserialize and send the request
            const request = deserializeRequest(task.request);
            return fetch(request)
              .then(response => {
                if (response.ok) {
                  return idb.delete('taskQueue', task.id);
                }
                throw new Error('Failed to sync task');
              })
              .catch(error => {
                console.error('[Service Worker] Sync failed:', error);
                // Keep the task in the queue for next sync
              });
          }));
        })
    );
  }
});

// Deserialize a request from storage
function deserializeRequest(serialized) {
  return new Request(serialized.url, {
    method: serialized.method,
    headers: serialized.headers,
    body: serialized.body,
    mode: serialized.mode,
    credentials: serialized.credentials,
    cache: serialized.cache,
    redirect: serialized.redirect,
    referrer: serialized.referrer
  });
}

// Handle push notifications
self.addEventListener('push', event => {
  console.log('[Service Worker] Push Received:', event);
  
  let notificationData = {
    title: 'QUASAR QA³ Update',
    body: 'Something new has happened!',
    icon: '/icon-192.png',
    badge: '/badge.png',
    actions: [
      { action: 'view', title: 'View' },
      { action: 'dismiss', title: 'Dismiss' }
    ]
  };
  
  if (event.data) {
    try {
      const data = event.data.json();
      notificationData = { ...notificationData, ...data };
    } catch (e) {
      notificationData.body = event.data.text();
    }
  }
  
  event.waitUntil(
    self.registration.showNotification(notificationData.title, {
      body: notificationData.body,
      icon: notificationData.icon,
      badge: notificationData.badge,
      vibrate: [100, 50, 100],
      data: {
        url: notificationData.url || '/',
        timestamp: new Date().getTime()
      },
      actions: notificationData.actions
    })
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', event => {
  console.log('[Service Worker] Notification click:', event);
  
  event.notification.close();
  
  if (event.action === 'dismiss') {
    return;
  }
  
  const url = event.notification.data.url || '/';
  
  event.waitUntil(
    clients.matchAll({ type: 'window' })
      .then(windowClients => {
        // Check if already open and focus
        for (const client of windowClients) {
          if (client.url === url && 'focus' in client) {
            return client.focus();
          }
        }
        
        // Otherwise open a new window
        if (clients.openWindow) {
          return clients.openWindow(url);
        }
      })
  );
});