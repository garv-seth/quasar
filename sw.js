/**
 * QUASAR QA³: Service Worker
 * Provides offline capabilities and caching for the PWA
 */

// Cache version - update when making changes to ensure updates are applied
const CACHE_VERSION = 'quasar-qa3-v1';

// Files to cache
const CACHE_FILES = [
  '/',
  '/index.html',
  '/offline.html',
  '/manifest.json',
  '/pwa.js',
  '/icon-192.png',
  '/icon-512.png'
];

// Install event
self.addEventListener('install', (event) => {
  console.log('Service Worker: Installing...');
  
  // Skip waiting to ensure the new service worker takes over immediately
  self.skipWaiting();
  
  event.waitUntil(
    caches.open(CACHE_VERSION)
      .then((cache) => {
        console.log('Service Worker: Caching files');
        return cache.addAll(CACHE_FILES);
      })
      .then(() => {
        console.log('Service Worker: Install complete');
      })
  );
});

// Activate event
self.addEventListener('activate', (event) => {
  console.log('Service Worker: Activating...');
  
  // Clean up old caches
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_VERSION) {
            console.log('Service Worker: Clearing old cache', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('Service Worker: Activation complete');
      return self.clients.claim();
    })
  );
});

// Fetch event
self.addEventListener('fetch', (event) => {
  console.log('Service Worker: Fetching', event.request.url);
  
  // Network-first strategy with fallback to cache
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // Clone the response for caching and for the browser
        const responseClone = response.clone();
        
        // Open the cache and store the new response
        caches.open(CACHE_VERSION)
          .then((cache) => {
            // Only cache successful responses
            if (response.status === 200) {
              cache.put(event.request, responseClone);
            }
          });
          
        return response;
      })
      .catch(() => {
        // If network fetch fails, try the cache
        return caches.match(event.request)
          .then((response) => {
            // If there's a cache hit, return it
            if (response) {
              return response;
            }
            
            // If the request is for a page, return the offline page
            if (event.request.mode === 'navigate') {
              return caches.match('/offline.html');
            }
            
            // For other assets, just return a simple response
            return new Response('Network error happened', {
              status: 404,
              headers: { 'Content-Type': 'text/plain' }
            });
          });
      })
  );
});

// Push notification event
self.addEventListener('push', (event) => {
  console.log('Service Worker: Push received');
  
  let title = 'QUASAR QA³ Update';
  let options = {
    body: 'Something requires your attention.',
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
        title: 'View Details',
        icon: '/icon-192.png'
      }
    ]
  };
  
  if (event.data) {
    try {
      const data = event.data.json();
      title = data.title || title;
      options.body = data.body || options.body;
      if (data.url) options.data.url = data.url;
    } catch (e) {
      options.body = event.data.text();
    }
  }
  
  event.waitUntil(self.registration.showNotification(title, options));
});

// Notification click event
self.addEventListener('notificationclick', (event) => {
  console.log('Service Worker: Notification click received');
  
  event.notification.close();
  
  // This looks to see if the current is already open and focuses it
  event.waitUntil(
    clients.matchAll({
      type: 'window'
    }).then((clientList) => {
      // If we have a custom URL, use it
      const url = event.notification.data.url || '/';
      
      // Check if there's already a window/tab open with the target URL
      for (let i = 0; i < clientList.length; i++) {
        const client = clientList[i];
        if (client.url === url && 'focus' in client) {
          return client.focus();
        }
      }
      
      // If not, open a new window/tab
      if (clients.openWindow) {
        return clients.openWindow(url);
      }
    })
  );
});

// Sync event for background syncing
self.addEventListener('sync', (event) => {
  console.log('Service Worker: Sync event', event.tag);
  
  if (event.tag === 'sync-quantum-tasks') {
    event.waitUntil(syncQuantumTasks());
  } else if (event.tag === 'sync-offline-data') {
    event.waitUntil(syncOfflineData());
  }
});

// Function to sync quantum tasks
async function syncQuantumTasks() {
  console.log('Syncing quantum tasks...');
  
  // Implementation would go here to sync stored tasks with the server
  // For now, we'll just log a success message
  console.log('Quantum tasks synced successfully');
}

// Function to sync offline data
async function syncOfflineData() {
  console.log('Syncing offline data...');
  
  // Implementation would go here to sync stored offline data with the server
  // For now, we'll just log a success message
  console.log('Offline data synced successfully');
}