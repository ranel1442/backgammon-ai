const CACHE_NAME = 'backgammon-ai-v6';
const WEIGHTS_PATH = '/weights/backgammon_nn.json';
const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/favicon.svg?v=3',
  '/favicon.ico?v=3',
  '/icon-192.png?v=3',
  '/manifest.webmanifest',
  WEIGHTS_PATH,
];

function isNetworkFirstAsset(request, url) {
  if (request.mode === 'navigate') return true;
  const p = url.pathname;
  return (
    p === '/' ||
    p.endsWith('.html') ||
    p.endsWith('.js') ||
    p.endsWith('.mjs') ||
    p.endsWith('.css') ||
    p.startsWith('/assets/')
  );
}

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) =>
        Promise.allSettled(PRECACHE_URLS.map((url) => cache.add(url)))
      )
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  // Network-First strategy for /weights/backgammon_nn.json so newly trained weights appear immediately
  if (url.pathname.endsWith(WEIGHTS_PATH)) {
    event.respondWith(
      fetch(request, { cache: 'no-store' })
        .then((networkResponse) => {
          if (networkResponse && networkResponse.status === 200) {
            const clone = networkResponse.clone();
            caches
              .open(CACHE_NAME)
              .then((cache) => cache.put(WEIGHTS_PATH, clone));
          }
          return networkResponse;
        })
        .catch(() =>
          caches
            .match(WEIGHTS_PATH)
            .then((cached) => cached || caches.match(request))
        )
    );
    return;
  }

  // Network-First strategy for HTML & JS/CSS bundles so new Vercel deployments always load immediately
  if (isNetworkFirstAsset(request, url)) {
    event.respondWith(
      fetch(request, { cache: 'no-store' })
        .then((networkResponse) => {
          if (networkResponse && networkResponse.status === 200) {
            const clone = networkResponse.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
          }
          return networkResponse;
        })
        .catch(() =>
          caches
            .match(request)
            .then((cached) =>
              cached || (request.mode === 'navigate' ? caches.match('/index.html') : undefined)
            )
        )
    );
    return;
  }

  // Cache-First with background revalidation for static icons/images
  event.respondWith(
    caches.match(request).then((cachedResponse) => {
      const fetchPromise = fetch(request)
        .then((networkResponse) => {
          if (networkResponse && networkResponse.status === 200) {
            const clone = networkResponse.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
          }
          return networkResponse;
        })
        .catch(() => cachedResponse);

      return cachedResponse || fetchPromise;
    })
  );
});
