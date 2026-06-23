// api.js
// ─────────────────────────────────────────────────────────────
// Talks to the Pi-side HTTP server (device/backend/server.py).
// Replaces the old ThingSpeak cloud round-trip.
//
// Set PI_BASE_URL to your Pi's LAN IP. For development we keep it
// here; production should surface it in a Settings screen.

import RNFS from 'react-native-fs';

export const PI_BASE_URL = 'http://eyezer.local:5000';   // ← change me

const SESSION_DIR = `${RNFS.DownloadDirectoryPath}/eyezer`;

// ── pipeline stages exposed by /status ───────────────────────
export const STAGE = {
  IDLE:               'idle',
  STARTING:           'starting',
  RECORDING:          'recording',
  SEGMENTING:         'segmenting',
  INFERENCE:          'inference',
  DONE:               'done',
  ERROR:              'error',
};

// ── REST calls ───────────────────────────────────────────────

export async function startSession(payload) {
  const r = await fetch(`${PI_BASE_URL}/session`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    throw new Error(`startSession failed: ${r.status}`);
  }
  return r.json();
}

export async function getStatus() {
  // Transient LAN errors are retried by the experiment screen.
  const r = await fetch(`${PI_BASE_URL}/status`);
  if (!r.ok) throw new Error(`status ${r.status}`);
  return r.json();
}

export async function getResults() {
  const r = await fetch(`${PI_BASE_URL}/results`);
  if (!r.ok) throw new Error(`results ${r.status}`);
  return r.json();
}

// ── Local cache (phone Downloads/eyezer/) ────────────────────

export async function cacheSession(meta, summary) {
  await RNFS.mkdir(SESSION_DIR).catch(() => {});
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const safeName = (meta.name || 'anon').replace(/[^a-z0-9]/gi, '_');
  const path = `${SESSION_DIR}/${stamp}_${safeName}.json`;
  await RNFS.writeFile(path, JSON.stringify({meta, summary}, null, 2), 'utf8');
  return path;
}
