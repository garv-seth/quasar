// Service Worker for QA³ PWA
const CACHE_NAME = 'qa3-cache-v1';
const urlsToCache = [
  '/',
  '/static/css/style.css',
  '/icon-192.png',
  '/icon-512.png',
  '/offline.html'
];

// Install event - cache core assets
self.addEventListener('install', event => {
  console.log('[ServiceWorker] Install');
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('[ServiceWorker] Caching app shell');
        return cache.addAll(urlsToCache);
      })
  );
  
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('[ServiceWorker] Activate');
  
  event.waitUntil(
    caches.keys().then(keyList => {
      return Promise.all(keyList.map(key => {
        if (key !== CACHE_NAME) {
          console.log('[ServiceWorker] Removing old cache', key);
          return caches.delete(key);
        }
      }));
    })
  );
  
  self.clients.claim();
});

// Fetch event - respond with cache, then network
self.addEventListener('fetch', event => {
  console.log('[ServiceWorker] Fetch', event.request.url);
  
  // Skip cross-origin requests
  if (!event.request.url.startsWith(self.location.origin)) {
    return;
  }
  
  // For API requests, try network first, then fall back to offline page
  if (event.request.url.includes('/api/')) {
    event.respondWith(
      fetch(event.request)
        .catch(() => {
          return caches.match('/offline.html');
        })
    );
    return;
  }
  
  // For all other requests, try cache first, then network with cache update
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        if (response) {
          // Return the cached response
          console.log('[ServiceWorker] Return cached response for', event.request.url);
          return response;
        }
        
        // If not in cache, fetch from network
        return fetch(event.request)
          .then(networkResponse => {
            // Check if valid response
            if (!networkResponse || networkResponse.status !== 200 || networkResponse.type !== 'basic') {
              return networkResponse;
            }
            
            // Clone the response to cache it and return it
            const responseToCache = networkResponse.clone();
            
            caches.open(CACHE_NAME)
              .then(cache => {
                cache.put(event.request, responseToCache);
              });
            
            return networkResponse;
          })
          .catch(() => {
            // If offline and resource not cached, show offline page
            if (event.request.headers.get('accept').includes('text/html')) {
              return caches.match('/offline.html');
            }
            
            // For images, return a generic placeholder
            if (event.request.headers.get('accept').includes('image/')) {
              return new Response(
                '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200"><rect width="200" height="200" fill="#f0f0f0"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#888">Image Offline</text></svg>',
                { headers: { 'Content-Type': 'image/svg+xml' } }
              );
            }
            
            // Default case
            return new Response('Offline content not available');
          });
      })
  );
});

// Background sync for offline tasks
self.addEventListener('sync', event => {
  console.log('[ServiceWorker] Sync event', event.tag);
  
  if (event.tag === 'offline-search') {
    event.waitUntil(syncOfflineSearches());
  } else if (event.tag === 'offline-task') {
    event.waitUntil(syncOfflineTasks());
  }
});

// Push notification handling
self.addEventListener('push', event => {
  console.log('[ServiceWorker] Push received', event);
  
  let title = 'QA³ Notification';
  let options = {
    body: 'Something important happened in your quantum application.',
    icon: '/icon-192.png',
    badge: '/icon-192.png',
    data: {
      url: self.location.origin
    }
  };
  
  if (event.data) {
    const data = event.data.json();
    title = data.title || title;
    options.body = data.body || options.body;
    if (data.url) options.data.url = data.url;
  }
  
  event.waitUntil(
    self.registration.showNotification(title, options)
  );
});

// Notification click handling
self.addEventListener('notificationclick', event => {
  console.log('[ServiceWorker] Notification click', event.notification.data);
  
  event.notification.close();
  
  event.waitUntil(
    clients.openWindow(event.notification.data.url || '/')
  );
});

// Helper function for background sync of offline searches
async function syncOfflineSearches() {
  try {
    // Get pending searches from IndexedDB
    const db = await openDatabase();
    const offlineSearches = await getStoredSearches(db);
    
    // Process each search
    for (const search of offlineSearches) {
      // Try to perform the search online
      try {
        const response = await fetch('/api/search', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(search)
        });
        
        if (response.ok) {
          // Search succeeded, remove from offline storage
          await deleteStoredSearch(db, search.id);
        }
      } catch (error) {
        console.error('Error syncing search:', error);
        // Keep in offline storage for next sync attempt
      }
    }
  } catch (error) {
    console.error('Error in syncOfflineSearches:', error);
  }
}

// Helper function for background sync of offline tasks
async function syncOfflineTasks() {
  try {
    // Get pending tasks from IndexedDB
    const db = await openDatabase();
    const offlineTasks = await getStoredTasks(db);
    
    // Process each task
    for (const task of offlineTasks) {
      // Try to perform the task online
      try {
        const response = await fetch('/api/task', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(task)
        });
        
        if (response.ok) {
          // Task succeeded, remove from offline storage
          await deleteStoredTask(db, task.id);
        }
      } catch (error) {
        console.error('Error syncing task:', error);
        // Keep in offline storage for next sync attempt
      }
    }
  } catch (error) {
    console.error('Error in syncOfflineTasks:', error);
  }
}

// Helper function to open IndexedDB
function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('qa3-offline-db', 1);
    
    request.onerror = event => {
      reject('Database error: ' + event.target.errorCode);
    };
    
    request.onsuccess = event => {
      resolve(event.target.result);
    };
    
    request.onupgradeneeded = event => {
      const db = event.target.result;
      
      // Create object stores if they don't exist
      if (!db.objectStoreNames.contains('searches')) {
        db.createObjectStore('searches', { keyPath: 'id' });
      }
      
      if (!db.objectStoreNames.contains('tasks')) {
        db.createObjectStore('tasks', { keyPath: 'id' });
      }
    };
  });
}

// Helper function to get stored searches
function getStoredSearches(db) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['searches'], 'readonly');
    const store = transaction.objectStore('searches');
    const request = store.getAll();
    
    request.onerror = event => {
      reject('Error fetching searches: ' + event.target.errorCode);
    };
    
    request.onsuccess = event => {
      resolve(event.target.result);
    };
  });
}

// Helper function to delete a stored search
function deleteStoredSearch(db, id) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['searches'], 'readwrite');
    const store = transaction.objectStore('searches');
    const request = store.delete(id);
    
    request.onerror = event => {
      reject('Error deleting search: ' + event.target.errorCode);
    };
    
    request.onsuccess = event => {
      resolve();
    };
  });
}

// Helper function to get stored tasks
function getStoredTasks(db) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['tasks'], 'readonly');
    const store = transaction.objectStore('tasks');
    const request = store.getAll();
    
    request.onerror = event => {
      reject('Error fetching tasks: ' + event.target.errorCode);
    };
    
    request.onsuccess = event => {
      resolve(event.target.result);
    };
  });
}

// Helper function to delete a stored task
function deleteStoredTask(db, id) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['tasks'], 'readwrite');
    const store = transaction.objectStore('tasks');
    const request = store.delete(id);
    
    request.onerror = event => {
      reject('Error deleting task: ' + event.target.errorCode);
    };
    
    request.onsuccess = event => {
      resolve();
    };
  });
}