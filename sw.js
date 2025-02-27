// QUASAR Labs QA³ Service Worker
const CACHE_NAME = 'quasar-qa3-cache-v1';
const urlsToCache = [
  '/',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png',
  '/_stcore/streamlit.js',
  '/_stcore/static/main.css'
];

// Install event - cache core assets
self.addEventListener('install', event => {
  console.log('[Service Worker] Installing...');
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('[Service Worker] Caching app shell');
        return cache.addAll(urlsToCache);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('[Service Worker] Activating...');
  
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            console.log('[Service Worker] Removing old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch event - serve from cache, fall back to network
self.addEventListener('fetch', event => {
  // Skip non-GET requests
  if (event.request.method !== 'GET') return;
  
  // Skip cross-origin requests
  if (!event.request.url.startsWith(self.location.origin)) return;
  
  // For Streamlit's WebSocket connections, always go to network
  if (event.request.url.includes('_stcore/stream')) return;
  
  event.respondWith(
    caches.match(event.request)
      .then(cachedResponse => {
        if (cachedResponse) {
          // Return cached response
          return cachedResponse;
        }
        
        // Not in cache, fetch from network
        return fetch(event.request)
          .then(response => {
            // Check if valid response
            if (!response || response.status !== 200 || response.type !== 'basic') {
              return response;
            }
            
            // Clone the response
            const responseToCache = response.clone();
            
            // Cache the fetched resource for future
            caches.open(CACHE_NAME)
              .then(cache => {
                cache.put(event.request, responseToCache);
              });
            
            return response;
          })
          .catch(error => {
            // Network request failed, show offline page
            console.error('[Service Worker] Fetch failed:', error);
            
            // For navigation requests, return the offline page
            if (event.request.mode === 'navigate') {
              return caches.match('/offline.html')
                .then(response => {
                  return response || new Response(
                    '<html><body><h1>QUASAR QA³ is offline</h1><p>Please check your internet connection.</p></body></html>', 
                    { headers: { 'Content-Type': 'text/html' } }
                  );
                });
            }
            
            return new Response('Network error', { status: 503, statusText: 'Service Unavailable' });
          });
      })
  );
});

// Push notification event handler
self.addEventListener('push', event => {
  console.log('[Service Worker] Push received:', event);
  
  let data = {};
  try {
    data = event.data.json();
  } catch (e) {
    data = {
      title: 'QUASAR QA³ Notification',
      body: event.data ? event.data.text() : 'New notification from QA³',
      icon: '/icon-192.png'
    };
  }
  
  const title = data.title || 'QUASAR QA³ Notification';
  const options = {
    body: data.body || 'New update from your quantum agent',
    icon: data.icon || '/icon-192.png',
    badge: '/badge.png',
    data: {
      url: data.url || '/'
    },
    actions: data.actions || [
      { action: 'view', title: 'View' },
      { action: 'dismiss', title: 'Dismiss' }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification(title, options)
  );
});

// Notification click event handler
self.addEventListener('notificationclick', event => {
  console.log('[Service Worker] Notification click:', event);
  event.notification.close();
  
  if (event.action === 'dismiss') return;
  
  const urlToOpen = event.notification.data && event.notification.data.url 
    ? event.notification.data.url 
    : '/';
  
  event.waitUntil(
    clients.matchAll({ type: 'window' })
      .then(windowClients => {
        // Check if there is already a window open
        for (let client of windowClients) {
          if (client.url === urlToOpen && 'focus' in client) {
            return client.focus();
          }
        }
        // If no open window, open a new one
        if (clients.openWindow) {
          return clients.openWindow(urlToOpen);
        }
      })
  );
});

// Background sync for offline task queuing
self.addEventListener('sync', event => {
  console.log('[Service Worker] Background sync:', event);
  
  if (event.tag === 'quantum-task-sync') {
    event.waitUntil(
      self.db.getAll('taskQueue')
        .then(tasks => {
          return Promise.all(
            tasks.map(task => {
              return fetch('/api/task', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(task)
              })
              .then(response => {
                if (response.ok) {
                  return self.db.delete('taskQueue', task.id);
                }
              })
              .catch(err => {
                console.error('[Service Worker] Failed to sync task:', err);
              });
            })
          );
        })
    );
  }
});

// Initialize IndexedDB for the service worker
self.db = {
  _dbPromise: null,
  
  get dbPromise() {
    if (!this._dbPromise) {
      this._dbPromise = new Promise((resolve, reject) => {
        const request = indexedDB.open('QUASAR-QA3-DB', 1);
        
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
        };
        
        request.onsuccess = event => {
          resolve(event.target.result);
        };
        
        request.onerror = event => {
          console.error('[Service Worker] IndexedDB error:', event.target.error);
          reject(event.target.error);
        };
      });
    }
    
    return this._dbPromise;
  },
  
  // Add item to a store
  add(storeName, item) {
    return this.dbPromise.then(db => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      store.add(item);
      return tx.complete;
    });
  },
  
  // Get item by key
  get(storeName, key) {
    return this.dbPromise.then(db => {
      const tx = db.transaction(storeName, 'readonly');
      const store = tx.objectStore(storeName);
      return store.get(key);
    }).then(result => result);
  },
  
  // Get all items from a store
  getAll(storeName) {
    return this.dbPromise.then(db => {
      const tx = db.transaction(storeName, 'readonly');
      const store = tx.objectStore(storeName);
      return store.getAll();
    }).then(result => result);
  },
  
  // Delete item by key
  delete(storeName, key) {
    return this.dbPromise.then(db => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      store.delete(key);
      return tx.complete;
    });
  },
  
  // Clear all items in a store
  clear(storeName) {
    return this.dbPromise.then(db => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      store.clear();
      return tx.complete;
    });
  }
};