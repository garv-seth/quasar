// QUASAR QA³ Service Worker

const CACHE_NAME = 'quasar-qa3-cache-v1';
const RUNTIME_CACHE = 'quasar-qa3-runtime-v1';

// Resources to pre-cache
const PRECACHE_URLS = [
  '/',
  '/offline.html',
  '/icon-192.png',
  '/icon-512.png',
  '/manifest.json',
  '/pwa.js'
];

// Lifecycle: Install
self.addEventListener('install', event => {
  console.log('[Service Worker] Installing Service Worker...', event);
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('[Service Worker] Pre-caching offline resources');
        return cache.addAll(PRECACHE_URLS);
      })
      .then(() => {
        console.log('[Service Worker] Successfully installed and cached resources');
        return self.skipWaiting();
      })
  );
});

// Lifecycle: Activate
self.addEventListener('activate', event => {
  console.log('[Service Worker] Activating Service Worker...', event);
  
  // Clean up old caches
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.filter(cacheName => {
          return cacheName !== CACHE_NAME && cacheName !== RUNTIME_CACHE;
        }).map(cacheName => {
          console.log('[Service Worker] Deleting old cache:', cacheName);
          return caches.delete(cacheName);
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch handler with network-first strategy for API requests and cache-first for assets
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  
  // For API requests, use network-first strategy
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirstStrategy(event.request));
  } 
  // For quantum calculation results, use network-first strategy
  else if (url.pathname.includes('quantum') || url.pathname.includes('calculation')) {
    event.respondWith(networkFirstStrategy(event.request));
  }
  // For main page and static assets, use cache-first strategy
  else {
    event.respondWith(cacheFirstStrategy(event.request));
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
    // Cache successful responses for future use
    if (networkResponse.ok) {
      const cache = await caches.open(RUNTIME_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.log('[Service Worker] Network request failed, serving offline page');
    // If the request is for a page (HTML), serve the offline page
    if (request.headers.get('Accept').includes('text/html')) {
      return caches.match('/offline.html');
    }
    
    // For other resources, simply return an error response
    return new Response('Network error happened', {
      status: 408,
      headers: { 'Content-Type': 'text/plain' }
    });
  }
}

// Network-first strategy for API requests
async function networkFirstStrategy(request) {
  try {
    // Try network first
    const networkResponse = await fetch(request);
    
    // Cache successful responses
    if (networkResponse.ok) {
      const cache = await caches.open(RUNTIME_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('[Service Worker] Network request failed, checking cache');
    
    // Fall back to cache
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // If not in cache and for a page, serve the offline page
    if (request.headers.get('Accept').includes('text/html')) {
      return caches.match('/offline.html');
    }
    
    // For API requests, return a JSON error
    return new Response(JSON.stringify({
      error: 'You are offline and this data is not cached.',
      offline: true,
      timestamp: new Date().toISOString()
    }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

// Handle background sync
self.addEventListener('sync', event => {
  console.log('[Service Worker] Background Sync Event:', event.tag);
  
  if (event.tag === 'sync-quantum-tasks') {
    event.waitUntil(syncQuantumTasks());
  }
});

// Sync quantum tasks that were created offline
async function syncQuantumTasks() {
  // This would ideally get from IndexedDB
  const tasksToSync = await getOfflineTasks();
  
  // Process each offline task
  const syncPromises = tasksToSync.map(async task => {
    try {
      const response = await fetch('/api/quantum/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(task)
      });
      
      if (response.ok) {
        // Task synced successfully, remove from offline queue
        await removeOfflineTask(task.id);
      }
      
      return response.ok;
    } catch (error) {
      console.error('[Service Worker] Failed to sync task:', error);
      return false;
    }
  });
  
  // Wait for all sync operations to complete
  return Promise.all(syncPromises);
}

// Simulate getting offline tasks (would use IndexedDB in a real implementation)
async function getOfflineTasks() {
  // Simulate IndexedDB read
  return [];
}

// Simulate removing an offline task (would use IndexedDB in a real implementation)
async function removeOfflineTask(taskId) {
  // Simulate IndexedDB delete
  return true;
}

// Handle push notifications
self.addEventListener('push', event => {
  console.log('[Service Worker] Push Received:', event);
  
  let notificationData = {
    title: 'QA³ Notification',
    body: 'Something happened in your QUASAR QA³ application.',
    icon: '/icon-192.png',
    badge: '/icon-192.png',
    data: {
      url: '/'
    }
  };
  
  // Try to parse the data if available
  if (event.data) {
    try {
      const data = event.data.json();
      notificationData = { ...notificationData, ...data };
    } catch (e) {
      console.error('Failed to parse push data:', e);
    }
  }
  
  event.waitUntil(
    self.registration.showNotification(notificationData.title, {
      body: notificationData.body,
      icon: notificationData.icon,
      badge: notificationData.badge,
      vibrate: [100, 50, 100],
      data: notificationData.data
    })
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', event => {
  console.log('[Service Worker] Notification click:', event);
  event.notification.close();
  
  // Open the target URL when the user clicks the notification
  if (event.notification.data && event.notification.data.url) {
    event.waitUntil(
      clients.matchAll({ type: 'window' }).then(windowClients => {
        // Check if there's already a window open that we can focus
        for (let client of windowClients) {
          if (client.url === event.notification.data.url && 'focus' in client) {
            return client.focus();
          }
        }
        // Open a new window otherwise
        if (clients.openWindow) {
          return clients.openWindow(event.notification.data.url);
        }
      })
    );
  }
});