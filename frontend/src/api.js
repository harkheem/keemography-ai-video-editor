// src/api.js
// All HTTP calls to the FastAPI backend.

const BASE = '/api'  // proxied in dev; same-origin in prod
// In dev the Vite proxy buffers large bodies — bypass it for uploads.
// In production the frontend IS the backend (same origin), so use relative URL.
const DIRECT = import.meta.env.DEV ? 'http://localhost:8000/api' : '/api'

// ---------------------------------------------------------------------------
// Per-browser session token. Sent with every job-related request so the
// backend can bind jobs to whoever created them — a leaked job_id alone
// isn't enough to view/download/delete someone else's job. Persisted in
// sessionStorage (per-tab, cleared on close); no accounts or login involved.
// ---------------------------------------------------------------------------
function getSessionId() {
  let id = sessionStorage.getItem('keemo_session_id')
  if (!id) {
    id = crypto.randomUUID()
    sessionStorage.setItem('keemo_session_id', id)
  }
  return id
}

// ---------------------------------------------------------------------------
// Upload video clips — chunked (50 MB per chunk) to survive proxy timeouts.
// Each chunk is sent as its own request so no single request exceeds ~10–15 s.
// ---------------------------------------------------------------------------
const CHUNK_SIZE = 50 * 1024 * 1024  // 50 MB

function xhrChunk(url, form, onChunkProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', url)
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onChunkProgress) onChunkProgress(e.loaded, e.total)
    })
    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText)) }
        catch { reject(new Error('Invalid JSON response')) }
      } else {
        let detail = `${xhr.status} ${xhr.statusText}`
        try { detail = JSON.parse(xhr.responseText).detail || detail } catch {}
        reject(new Error(`Upload failed: ${detail}`))
      }
    })
    xhr.addEventListener('error', () => reject(new Error('Upload failed: network error')))
    xhr.addEventListener('abort', () => reject(new Error('Upload cancelled')))
    xhr.send(form)
  })
}

export async function uploadClips(files, onProgress) {
  const fileArr = Array.from(files)
  const totalBytes = fileArr.reduce((s, f) => s + f.size, 0)
  let uploadedBytes = 0
  const clips = []

  for (const file of fileArr) {
    const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_SIZE))
    const uploadId = crypto.randomUUID()

    for (let i = 0; i < totalChunks; i++) {
      const start = i * CHUNK_SIZE
      const end = Math.min(start + CHUNK_SIZE, file.size)
      const blob = file.slice(start, end)

      const form = new FormData()
      form.append('upload_id', uploadId)
      form.append('chunk_index', i)
      form.append('total_chunks', totalChunks)
      form.append('filename', file.name)
      form.append('chunk', blob, file.name)

      const chunkStart = uploadedBytes
      await xhrChunk(`${DIRECT}/upload/chunk`, form, (loaded) => {
        if (onProgress) {
          const total = totalBytes
          const done = chunkStart + loaded
          onProgress(Math.min(99, Math.round((done / total) * 100)), done, total)
        }
      })
      uploadedBytes += (end - start)
    }

    // Assemble the chunks server-side
    const res = await fetch(`${DIRECT}/upload/finalize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ upload_id: uploadId }),
    })
    if (!res.ok) {
      let detail = res.statusText
      try { detail = (await res.json()).detail || detail } catch {}
      throw new Error(`Finalize failed: ${detail}`)
    }
    clips.push(await res.json())
  }

  if (onProgress) onProgress(100, totalBytes, totalBytes)
  return { clips }
}

// ---------------------------------------------------------------------------
// Upload background music
// ---------------------------------------------------------------------------
export async function uploadMusic(file, targetDurationSec) {
  const form = new FormData()
  form.append('file', file)
  // Lets the backend suggest a trim window matching the user's actual target
  // (it used to always analyze for a hardcoded 45s edit).
  if (targetDurationSec != null) form.append('target_duration_sec', String(targetDurationSec))
  const res = await fetch(`${BASE}/upload/music`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Music upload failed: ${res.statusText}`)
  return res.json() // { name, path, size, analysis }
}

// ---------------------------------------------------------------------------
// Fetch a video from a URL
// ---------------------------------------------------------------------------
export async function fetchUrl(url) {
  const res = await fetch(`${BASE}/fetch-url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.detail || `Fetch failed: ${res.statusText}`)
  }
  return res.json() // { name, path, size }
}

// ---------------------------------------------------------------------------
// Start a generation job
// ---------------------------------------------------------------------------
export async function startGenerate(params) {
  const form = new FormData()
  form.append('clip_paths', JSON.stringify(params.clipPaths))
  form.append('storyline', params.storyline)
  form.append('tone', params.tone)
  form.append('target_duration_sec', String(params.targetDurationSec))
  form.append('transition_duration', String(params.transitionDuration))
  form.append('mix_original_audio', String(params.mixOriginalAudio))
  form.append('show_opening_card', String(params.showOpeningCard))
  if (params.musicPath) form.append('music_path', params.musicPath)
  form.append('music_start_sec', String(params.musicStartSec ?? 0))
  if (params.musicEndSec != null) form.append('music_end_sec', String(params.musicEndSec))
  form.append('priority_keywords', params.priorityKeywords || '')
  form.append('exclude_keywords', params.excludeKeywords || '')
  if (params.apiKey) form.append('api_key', params.apiKey)
  form.append('session_id', getSessionId())

  const res = await fetch(`${BASE}/generate`, { method: 'POST', body: form })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.detail || `Generate failed: ${res.statusText}`)
  }
  return res.json() // { job_id }
}

// ---------------------------------------------------------------------------
// Poll job status (used when SSE is unavailable)
// ---------------------------------------------------------------------------
export async function pollStatus(jobId) {
  const res = await fetch(`${BASE}/status/${jobId}?session_id=${getSessionId()}`)
  if (!res.ok) throw new Error(`Status check failed: ${res.statusText}`)
  return res.json() // { status, progress, message, error, has_result, clip_analysis }
}

// ---------------------------------------------------------------------------
// SSE stream for live progress
// ---------------------------------------------------------------------------
export function streamEvents(jobId, onEvent, onDone, onError) {
  // EventSource can't send custom headers, so the session token travels as a
  // query param here (same mechanism api.js uses everywhere else for job auth).
  const es = new EventSource(`${BASE}/events/${jobId}?session_id=${getSessionId()}`)
  let closed = false

  const close = () => {
    closed = true
    es.close()
  }

  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data)
      if (data.done) {
        close()
        onDone(data)
      } else if (data.error && !data.progress) {
        close()
        onError(new Error(data.error))
      } else {
        onEvent(data)
      }
    } catch {
      // non-JSON frame, ignore
    }
  }
  es.onerror = () => {
    if (closed) return  // server closed stream after done — not a real error
    close()
    onError(new Error('Cannot reach backend server. Make sure it is running.'))
  }
  return close
}

// ---------------------------------------------------------------------------
// Download URL
// ---------------------------------------------------------------------------
export function downloadUrl(jobId) {
  return `${BASE}/download/${jobId}?session_id=${getSessionId()}`
}

// ---------------------------------------------------------------------------
// Delete a job and its output file
// ---------------------------------------------------------------------------
export async function deleteJob(jobId) {
  const res = await fetch(`${BASE}/job/${jobId}?session_id=${getSessionId()}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`Delete failed: ${res.statusText}`)
  return res.json() // { deleted }
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------
export async function healthCheck() {
  const res = await fetch(`${BASE}/health`)
  return res.ok ? res.json() : null
}
