import { useState, useRef, useCallback, useEffect } from 'react'
import {
  uploadClips,
  uploadMusic,
  fetchUrl,
  startGenerate,
  pollStatus,
  streamEvents,
  downloadUrl,
} from './api'

// ─── tiny helpers ───────────────────────────────────────────────────────────

function fmtBytes(b) {
  if (b < 1024) return `${b} B`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / (1024 * 1024)).toFixed(1)} MB`
}

function Label({ children }) {
  return (
    <p className="text-[0.62rem] font-bold tracking-[0.22em] uppercase text-[#e63939] mb-1">
      {children}
    </p>
  )
}

function SecTitle({ children }) {
  return (
    <p className="text-[0.97rem] font-bold text-white mb-3">{children}</p>
  )
}

function Divider() {
  return <hr className="border-none border-t border-[#181818] my-5" />
}

function Panel({ children, className = '' }) {
  return (
    <div
      className={`bg-white/[0.026] border border-white/[0.07] rounded-2xl p-5 mb-3 ${className}`}
    >
      {children}
    </div>
  )
}

function Btn({ onClick, disabled, primary, danger, children, className = '' }) {
  let base =
    'px-4 py-2 rounded-xl font-semibold text-sm border transition-all cursor-pointer select-none '
  if (primary)
    base +=
      'bg-gradient-to-br from-[#b52828] to-[#e63939] border-transparent text-white font-extrabold text-[1.05rem] shadow-[0_4px_22px_rgba(230,57,57,0.38)] hover:shadow-[0_6px_30px_rgba(230,57,57,0.58)] active:scale-95 '
  else if (danger)
    base +=
      'bg-transparent border-[#3a1010] text-[#e63939] hover:bg-[#1a0808] hover:border-[#e63939] '
  else
    base +=
      'bg-[#111] border-[#252525] text-[#ddd] hover:border-[#e63939] hover:text-white hover:bg-[#1a0808] '
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={base + (disabled ? 'opacity-40 cursor-not-allowed ' : '') + className}
    >
      {children}
    </button>
  )
}

function Input({ value, onChange, placeholder, type = 'text', className = '' }) {
  return (
    <input
      type={type}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      className={
        'w-full bg-[#0d0d0d] border border-[#242424] rounded-xl px-3 py-2 text-[0.9rem] text-[#f0f0f0] ' +
        'placeholder-[#3c3c3c] focus:outline-none focus:border-[#e63939] focus:ring-1 focus:ring-[#e63939]/20 ' +
        className
      }
    />
  )
}

function Textarea({ value, onChange, placeholder, rows = 4 }) {
  return (
    <textarea
      rows={rows}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      className={
        'w-full bg-[#0d0d0d] border border-[#242424] rounded-xl px-3 py-2 text-[0.9rem] text-[#f0f0f0] ' +
        'placeholder-[#3c3c3c] focus:outline-none focus:border-[#e63939] focus:ring-1 focus:ring-[#e63939]/20 ' +
        'resize-none'
      }
    />
  )
}

// ─── Upload drop zone ────────────────────────────────────────────────────────

function UploadZone({ clips, onAdd, onRemove }) {
  const inputRef = useRef()
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [reading, setReading] = useState(false)   // true while resolving iOS file metadata
  const [uploadPct, setUploadPct] = useState(0)
  const [uploadLoaded, setUploadLoaded] = useState(0)
  const [uploadTotal, setUploadTotal] = useState(0)
  const [error, setError] = useState('')
  // Staged files: selected by user but not yet uploaded
  const [pending, setPending] = useState([])

  const isVideo = (f) =>
    ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'video/x-matroska'].includes(f.type) ||
    /\.(mp4|mov|avi|mkv|webm|mpeg4)$/i.test(f.name)

  // Add files to the staging queue — does NOT start upload.
  // Uses setTimeout(0) so the "Reading…" state renders BEFORE we touch
  // File.size / File.name, which can block the main thread on iOS when
  // files are sourced from iCloud or the Photos library.
  const stageFiles = useCallback((files) => {
    const raw = Array.from(files)
    if (!raw.length) return
    setError('')
    setReading(true)
    if (inputRef.current) inputRef.current.value = ''

    setTimeout(() => {
      try {
        const videoFiles = raw.filter(isVideo)
        if (!videoFiles.length) {
          setError('Please select MP4, MOV, AVI, MKV, or WebM files.')
          return
        }
        setPending((prev) => {
          const existing = new Set(prev.map((f) => f.name + f.size))
          const fresh = videoFiles.filter((f) => !existing.has(f.name + f.size))
          return [...prev, ...fresh]
        })
      } finally {
        setReading(false)
      }
    }, 0)
  }, [])

  // Actually upload all staged files
  const startUpload = useCallback(async () => {
    if (!pending.length) return
    setError('')
    setUploading(true)
    setUploadPct(0)
    setUploadLoaded(0)
    setUploadTotal(0)
    try {
      const result = await uploadClips(pending, (pct, loaded, total) => {
        setUploadPct(pct)
        setUploadLoaded(loaded)
        setUploadTotal(total)
      })
      onAdd(result.clips)
      setPending([])
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
      setUploadPct(0)
    }
  }, [pending, onAdd])

  return (
    <div>
      {/* Drop / click zone */}
      <div
        onClick={() => !uploading && !reading && inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); stageFiles(e.dataTransfer.files) }}
        className={
          'border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all ' +
          (dragging ? 'border-[#e63939] bg-[#1a0808]' : 'border-[#252525] bg-[#0a0a0a] hover:border-[#e63939]/60')
        }
      >
        {/* Specific MIME types let iOS match files without triggering transcoding.
            video/* causes iOS to transcode — individual types do not. */}
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="video/mp4,video/quicktime,video/x-msvideo,video/webm,video/x-matroska,.mp4,.mov,.avi,.mkv,.webm,.mpeg4"
          onChange={(e) => stageFiles(e.target.files)}
          className="hidden"
        />
        {uploading ? (
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-[#888] mb-1">
              <span>Uploading… {uploadPct}%</span>
              <span>{uploadTotal > 0 ? `${fmtBytes(uploadLoaded)} / ${fmtBytes(uploadTotal)}` : ''}</span>
            </div>
            <div className="w-full h-2 bg-[#1a1a1a] rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-[#b52828] to-[#e63939] rounded-full transition-all duration-200"
                style={{ width: `${uploadPct}%` }}
              />
            </div>
          </div>
        ) : reading ? (
          <p className="text-[#888] text-sm animate-pulse">Reading files…</p>
        ) : (
          <>
            <p className="text-[#555] text-sm mb-1">
              {pending.length > 0
                ? <span className="text-[#aaa]">Tap to add more clips</span>
                : <><span className="text-[#e63939]">Tap to select clips</span> or drag & drop</>}
            </p>
            <p className="text-[#333] text-xs">MP4 / MOV / AVI / MKV · no size limit</p>
          </>
        )}
      </div>

      {error && <p className="text-[#e63939] text-xs mt-2">{error}</p>}

      {/* Staged (not yet uploaded) files */}
      {pending.length > 0 && !uploading && (
        <div className="mt-3">
          <p className="text-[0.72rem] text-[#555] mb-1.5 uppercase tracking-wide font-bold">
            Ready to upload · {pending.length} file{pending.length > 1 ? 's' : ''} · {fmtBytes(pending.reduce((s, f) => s + f.size, 0))}
          </p>
          <div className="space-y-1.5 mb-3">
            {pending.map((f, i) => (
              <div key={i} className="flex items-center gap-2 bg-[#0d0d0d] border border-[#252525] rounded-xl px-3 py-2">
                <span className="text-[0.72rem] bg-[#101a10] text-[#4caf50] border border-[#1a3a1a] rounded px-1.5 py-0.5 uppercase tracking-wide font-bold flex-shrink-0">
                  STAGED
                </span>
                <span className="text-[0.82rem] text-[#ccc] flex-1 truncate">{f.name}</span>
                <span className="text-[0.72rem] text-[#555] flex-shrink-0">{fmtBytes(f.size)}</span>
                <button
                  onClick={() => setPending((prev) => prev.filter((_, j) => j !== i))}
                  className="text-[#333] hover:text-[#e63939] text-xs leading-none ml-1 flex-shrink-0"
                  title="Remove"
                >✕</button>
              </div>
            ))}
          </div>
          <Btn primary onClick={startUpload} className="w-full justify-center">
            Upload {pending.length} clip{pending.length > 1 ? 's' : ''}
          </Btn>
        </div>
      )}

      {clips.length > 0 && (
        <div className="mt-3 space-y-1.5">
          {clips.map((c, i) => (
            <div
              key={i}
              className="flex items-center gap-2 bg-[#0d0d0d] border border-[#1c1c1c] rounded-xl px-3 py-2"
            >
              <span className="text-[0.72rem] bg-[#1a0808] text-[#e63939] border border-[#3a1212] rounded px-1.5 py-0.5 uppercase tracking-wide font-bold flex-shrink-0">
                CLIP
              </span>
              <span className="text-[0.82rem] text-[#ccc] flex-1 truncate">{c.name}</span>
              <span className="text-[0.72rem] text-[#555] flex-shrink-0">{fmtBytes(c.size)}</span>
              <button
                onClick={() => onRemove(i)}
                className="text-[#333] hover:text-[#e63939] text-xs leading-none ml-1 flex-shrink-0"
                title="Remove"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── URL fetcher ─────────────────────────────────────────────────────────────

function UrlFetcher({ onAdd }) {
  const [raw, setRaw] = useState('')
  const [fetching, setFetching] = useState(false)
  const [msgs, setMsgs] = useState([])

  const handleFetch = async () => {
    const urls = raw
      .split(/[\n,]+/)
      .map((u) => u.trim())
      .filter(Boolean)
    if (!urls.length) return
    setFetching(true)
    setMsgs([])
    const fetched = []
    for (const url of urls) {
      setMsgs((m) => [...m, { url, status: 'fetching' }])
      try {
        const result = await fetchUrl(url)
        fetched.push(result)
        setMsgs((m) =>
          m.map((x) => (x.url === url ? { ...x, status: 'ok', name: result.name } : x))
        )
      } catch (e) {
        setMsgs((m) =>
          m.map((x) => (x.url === url ? { ...x, status: 'error', error: e.message } : x))
        )
      }
    }
    if (fetched.length) onAdd(fetched)
    setFetching(false)
  }

  return (
    <div>
      <Textarea
        value={raw}
        onChange={(e) => setRaw(e.target.value)}
        placeholder="https://…/clip1.mp4&#10;https://…/clip2.mp4&#10;Google Drive / Dropbox links auto-converted"
        rows={3}
      />
      <div className="flex gap-2 mt-2">
        <Btn onClick={handleFetch} disabled={fetching || !raw.trim()} className="flex-1">
          {fetching ? 'Fetching…' : '⬇ Fetch URLs'}
        </Btn>
        <Btn
          onClick={() => {
            setRaw('')
            setMsgs([])
          }}
          disabled={fetching}
        >
          Clear
        </Btn>
      </div>
      {msgs.length > 0 && (
        <div className="mt-2 space-y-1">
          {msgs.map((m, i) => (
            <p
              key={i}
              className={
                'text-xs truncate ' +
                (m.status === 'ok'
                  ? 'text-green-400'
                  : m.status === 'error'
                  ? 'text-[#e63939]'
                  : 'text-[#888]')
              }
            >
              {m.status === 'ok' && '✓ '}
              {m.status === 'error' && '✗ '}
              {m.status === 'fetching' && '⏳ '}
              {m.status === 'ok' ? m.name : m.status === 'error' ? m.error : m.url.slice(0, 60)}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Music upload ─────────────────────────────────────────────────────────────

function fmtTime(sec) {
  if (sec == null || isNaN(sec)) return '0:00'
  const s = Math.round(sec)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

function MusicUpload({ musicInfo, musicTrim, onSet, onTrimChange, onClear }) {
  const inputRef = useRef()
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')

  const handleFile = async (file) => {
    if (!file) return
    if (file.size > 100 * 1024 * 1024) {
      setError('Max 100 MB')
      return
    }
    setError('')
    setUploading(true)
    try {
      const result = await uploadMusic(file)
      onSet(result)
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  const analysis = musicInfo?.analysis
  const duration = analysis?.duration || 0
  const showSliders = musicInfo && duration > 0
  const trimStart = musicTrim?.start ?? 0
  const trimEnd = musicTrim?.end ?? duration

  return (
    <div>
      <input
        ref={inputRef}
        type="file"
        accept=".mp3,.wav,.m4a,.aac,.ogg,audio/*"
        className="hidden"
        onChange={(e) => handleFile(e.target.files[0])}
      />
      {musicInfo ? (
        <div className="space-y-3">
          {/* File badge */}
          <div className="flex items-center gap-2 bg-[#0d0d0d] border border-[#1c1c1c] rounded-xl px-3 py-2">
            <span className="text-green-400 text-sm">♪</span>
            <span className="text-[0.82rem] text-[#ccc] flex-1 truncate">{musicInfo.name}</span>
            {duration > 0 && (
              <span className="text-[0.72rem] text-[#555]">{fmtTime(duration)}</span>
            )}
            <span className="text-[0.72rem] text-[#444] mx-1">{fmtBytes(musicInfo.size)}</span>
            <button onClick={onClear} className="text-[#333] hover:text-[#e63939] text-xs ml-1">✕</button>
          </div>

          {/* AI-powered trim sliders */}
          {showSliders && (
            <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-xl px-3 py-3 space-y-3">
              <div className="flex justify-between items-center">
                <p className="text-[0.75rem] text-[#888] font-medium">Trim music</p>
                <span className={`text-[0.62rem] px-2 py-0.5 rounded-full border font-semibold ${
                  musicTrim?.aiSuggested
                    ? 'bg-[#0a1f0a] border-[#1a5c1a] text-[#4caf50]'
                    : 'bg-[#1a0808] border-[#e63939] text-[#ff7070]'
                }`}>
                  {musicTrim?.aiSuggested ? 'AI Suggested' : 'Custom'}
                </span>
              </div>

              {/* Start slider */}
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-[0.65rem] text-[#555]">Start</span>
                  <span className="text-[0.65rem] text-[#aaa]">{fmtTime(trimStart)}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={Math.max(0, duration - 5)}
                  step={1}
                  value={trimStart}
                  onChange={(e) => onTrimChange({ start: Number(e.target.value), end: trimEnd })}
                  className="w-full accent-[#e63939] bg-[#252525] rounded-full h-1 cursor-pointer"
                />
              </div>

              {/* End slider */}
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-[0.65rem] text-[#555]">End</span>
                  <span className="text-[0.65rem] text-[#aaa]">{fmtTime(trimEnd)}</span>
                </div>
                <input
                  type="range"
                  min={trimStart + 5}
                  max={duration}
                  step={1}
                  value={trimEnd}
                  onChange={(e) => onTrimChange({ start: trimStart, end: Number(e.target.value) })}
                  className="w-full accent-[#e63939] bg-[#252525] rounded-full h-1 cursor-pointer"
                />
              </div>

              <p className="text-[0.62rem] text-[#444]">
                Using {fmtTime(trimEnd - trimStart)} of {fmtTime(duration)} track
              </p>
            </div>
          )}
        </div>
      ) : (
        <Btn onClick={() => !uploading && inputRef.current.click()} disabled={uploading}>
          {uploading ? 'Analyzing…' : '+ Upload MP3 / WAV (optional)'}
        </Btn>
      )}
      {error && <p className="text-[#e63939] text-xs mt-1">{error}</p>}
    </div>
  )
}

// ─── Settings panel ───────────────────────────────────────────────────────────

const TONES = ['Cinematic', 'Energetic', 'Sentimental', 'Epic', 'Calm']

function SettingsPanel({ settings, onChange }) {
  const set = (key) => (val) => onChange({ ...settings, [key]: val })

  return (
    <div className="space-y-4">
      {/* Tone */}
      <div>
        <p className="text-[0.78rem] text-[#888] font-medium mb-1.5">Tone</p>
        <div className="flex flex-wrap gap-2">
          {TONES.map((t) => (
            <button
              key={t}
              onClick={() => set('tone')(t)}
              className={
                'px-3 py-1.5 rounded-full text-xs font-bold border pill-transition ' +
                (settings.tone === t
                  ? 'bg-[#1a0808] border-[#e63939] text-[#ff7070] shadow-[0_0_14px_rgba(230,57,57,0.25)]'
                  : 'bg-[#0e0e0e] border-[#222] text-[#444] hover:border-[#444]')
              }
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Duration slider */}
      <div>
        <div className="flex justify-between mb-1">
          <p className="text-[0.78rem] text-[#888] font-medium">Target duration</p>
          <p className="text-[0.78rem] text-[#aaa]">{settings.targetDuration}s</p>
        </div>
        <input
          type="range"
          min={15}
          max={180}
          step={5}
          value={settings.targetDuration}
          onChange={(e) => set('targetDuration')(Number(e.target.value))}
          className="w-full accent-[#e63939] bg-[#252525] rounded-full h-1 cursor-pointer"
        />
        <div className="flex justify-between mt-0.5">
          <span className="text-[0.65rem] text-[#444]">15s</span>
          <span className="text-[0.65rem] text-[#444]">180s</span>
        </div>
      </div>

      {/* Transition slider */}
      <div>
        <div className="flex justify-between mb-1">
          <p className="text-[0.78rem] text-[#888] font-medium">Transition length</p>
          <p className="text-[0.78rem] text-[#aaa]">{settings.transitionDuration.toFixed(2)}s</p>
        </div>
        <input
          type="range"
          min={0.15}
          max={1.5}
          step={0.05}
          value={settings.transitionDuration}
          onChange={(e) => set('transitionDuration')(Number(e.target.value))}
          className="w-full accent-[#e63939] bg-[#252525] rounded-full h-1 cursor-pointer"
        />
        <div className="flex justify-between mt-0.5">
          <span className="text-[0.65rem] text-[#444]">0.15s snappy</span>
          <span className="text-[0.65rem] text-[#444]">1.5s smooth</span>
        </div>
      </div>

      {/* Toggles */}
      <div className="flex gap-4">
        <Toggle
          label="Mix original audio"
          value={settings.mixOriginalAudio}
          onChange={set('mixOriginalAudio')}
        />
        <Toggle
          label="Opening title card"
          value={settings.showOpeningCard}
          onChange={set('showOpeningCard')}
        />
      </div>
    </div>
  )
}

function Toggle({ label, value, onChange }) {
  return (
    <button
      onClick={() => onChange(!value)}
      className="flex items-center gap-2 text-[0.78rem] text-[#888] select-none"
    >
      <span
        className={
          'w-9 h-5 rounded-full relative transition-colors ' +
          (value ? 'bg-[#e63939]' : 'bg-[#2a2a2a]')
        }
      >
        <span
          className={
            'absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ' +
            (value ? 'translate-x-4' : 'translate-x-0.5')
          }
        />
      </span>
      {label}
    </button>
  )
}

// ─── Progress display ─────────────────────────────────────────────────────────

const STAGES = ['Upload', 'Transcribe', 'Analyze', 'Render', 'Done']

function stageName(progress) {
  if (progress < 12) return 'Upload'
  if (progress < 47) return 'Transcribe'
  if (progress < 68) return 'Analyze'
  if (progress < 100) return 'Render'
  return 'Done'
}

function ProgressPanel({ progress, message, status, startTime }) {
  const current = stageName(progress)
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    if (status !== 'running') return
    setElapsed(startTime ? Math.floor((Date.now() - startTime) / 1000) : 0)
    const id = setInterval(() => {
      setElapsed(startTime ? Math.floor((Date.now() - startTime) / 1000) : 0)
    }, 1000)
    return () => clearInterval(id)
  }, [status, startTime])

  const fmtTime = (s) => {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`
  }

  return (
    <Panel>
      <div className="flex flex-wrap gap-1.5 mb-4">
        {STAGES.map((s) => {
          const isDone = STAGES.indexOf(s) < STAGES.indexOf(current)
          const isActive = s === current
          return (
            <span
              key={s}
              className={
                'text-[0.65rem] font-bold tracking-wide uppercase px-3 py-1 rounded-full border pill-transition ' +
                (isDone
                  ? 'bg-[#0a160a] border-[#1a5c1a] text-[#5ed85e]'
                  : isActive
                  ? 'bg-[#1a0808] border-[#e63939] text-[#ff7070] shadow-[0_0_12px_rgba(230,57,57,0.2)]'
                  : 'bg-[#0e0e0e] border-[#222] text-[#444]')
              }
            >
              {s}
            </span>
          )
        })}
      </div>

      <div className="h-2 bg-[#181818] rounded-full overflow-hidden mb-2">
        {status === 'error' ? (
          <div className="h-full bg-[#e63939] rounded-full w-full opacity-40" />
        ) : (
          <div
            className={`h-full rounded-full transition-all duration-700 ${
              progress < 100 ? 'progress-shimmer' : 'bg-[#1a8c1a]'
            }`}
            style={{ width: `${progress}%` }}
          />
        )}
      </div>

      <p className="text-[0.8rem] text-[#888] mt-2">
        {status === 'error' ? (
          <span className="text-[#e63939]">✗ {message}</span>
        ) : (
          <>
            <span className="text-[#e63939] font-bold">{progress}%</span>
            <span className="mx-2 text-[#333]">·</span>
            {message}
            {status === 'running' && startTime && (
              <span className="ml-2 text-[#555]">· {fmtTime(elapsed)}</span>
            )}
          </>
        )}
      </p>
    </Panel>
  )
}

// ─── Clip analysis ────────────────────────────────────────────────────────────

const BADGE_STYLE = {
  hook:        'bg-[#2a0e00] text-[#ff8c2a] border-[#6b3000]',
  payoff:      'bg-[#18002e] text-[#c987ff] border-[#56228a]',
  turn:        'bg-[#00102e] text-[#54b4ff] border-[#1a446e]',
  development: 'bg-[#0a160a] text-[#68c868] border-[#26502a]',
  broll:       'bg-[#161616] text-[#888]    border-[#2e2e2e]',
}

function ClipAnalysis({ clips }) {
  const [open, setOpen] = useState(false)
  if (!clips || !clips.length) return null
  return (
    <Panel>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center justify-between w-full text-left"
      >
        <p className="text-[0.85rem] font-semibold text-white">
          🔍 Visual Analysis ({clips.length} clips)
        </p>
        <span className="text-[#555] text-sm">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="mt-3 space-y-1.5">
          {clips.map((c, i) => {
            const role = c.narrative_role || 'development'
            const badge = BADGE_STYLE[role] || BADGE_STYLE.broll
            return (
              <div
                key={i}
                className="flex items-center gap-2 bg-[#0d0d0d] border border-[#1c1c1c] rounded-xl px-3 py-2.5"
              >
                <span className={`text-[0.6rem] font-bold px-2 py-0.5 rounded border uppercase tracking-wide flex-shrink-0 ${badge}`}>
                  {role}
                </span>
                <span className="text-[0.62rem] font-semibold px-1.5 py-0.5 rounded border bg-[#141414] text-[#999] border-[#222] flex-shrink-0">
                  {c.emotion || 'neutral'}
                </span>
                <span className="text-[0.8rem] text-[#ddd] flex-1 truncate font-medium">{c.name}</span>
                <span className="text-[0.75rem] text-[#e63939] font-bold flex-shrink-0">
                  {(c.visual_score || 0).toFixed(2)}
                </span>
              </div>
            )
          })}
        </div>
      )}
    </Panel>
  )
}

// ─── Result panel ─────────────────────────────────────────────────────────────

function ResultPanel({ jobId }) {
  const [dlError, setDlError] = useState(null)

  const handleDownload = async () => {
    setDlError(null)
    try {
      const res = await fetch(`/api/download/${jobId}`)
      if (!res.ok) {
        const msg = await res.text()
        throw new Error(`Download failed (${res.status}): ${msg}`)
      }
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `keemography_${jobId.slice(0, 8)}.mp4`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      setDlError(e.message)
    }
  }

  return (
    <div
      className="bg-gradient-to-br from-[#0b1508] to-[#080d08] border border-[#1a3a1a] rounded-2xl p-6 mt-4
                 shadow-[0_8px_40px_rgba(0,0,0,0.5)]"
    >
      <p className="text-[#6de06d] font-extrabold tracking-wide text-[1.05rem] mb-4">
        ✅ Video Ready
      </p>
      <video
        src={`/api/download/${jobId}`}
        controls
        className="w-full rounded-xl bg-black mb-4"
        style={{ maxHeight: '340px' }}
      />
      <button
        onClick={handleDownload}
        className={
          'block text-center w-full py-2.5 rounded-xl font-bold text-[#6de06d] ' +
          'bg-gradient-to-br from-[#0a1f0a] to-[#0e2e0e] border border-[#1a5c1a] ' +
          'shadow-[0_4px_16px_rgba(20,100,20,0.25)] hover:shadow-[0_6px_24px_rgba(20,100,20,0.4)] ' +
          'transition-all cursor-pointer w-full'
        }
      >
        📥 Download MP4
      </button>
      {dlError && (
        <p className="text-red-400 text-xs mt-2 text-center">{dlError}</p>
      )}
    </div>
  )
}

// ─── Step indicator ───────────────────────────────────────────────────────────

function StepBar({ step, onBack }) {
  const steps = ['Clips', 'Settings', 'Generate']
  return (
    <div className="flex items-center justify-center gap-0 mb-8 px-4">
      {steps.map((label, i) => {
        const n = i + 1
        const done = step > n
        const active = step === n
        return (
          <div key={n} className="flex items-center">
            <button
              onClick={() => done && onBack(n)}
              disabled={!done}
              className={
                'flex items-center gap-2 px-3 py-2 rounded-xl text-[0.78rem] font-bold transition-all ' +
                (active
                  ? 'text-white bg-[#1a0808] border border-[#e63939] shadow-[0_0_14px_rgba(230,57,57,0.22)]'
                  : done
                  ? 'text-[#5ed85e] bg-[#0a160a] border border-[#1a5c1a] cursor-pointer hover:border-[#5ed85e]'
                  : 'text-[#333] bg-[#0d0d0d] border border-[#1c1c1c] cursor-default')
              }
            >
              <span className={
                'w-5 h-5 rounded-full flex items-center justify-center text-[0.65rem] font-black flex-shrink-0 ' +
                (active ? 'bg-[#e63939] text-white' : done ? 'bg-[#1a5c1a] text-[#5ed85e]' : 'bg-[#181818] text-[#333]')
              }>
                {done ? '✓' : n}
              </span>
              {label}
            </button>
            {i < steps.length - 1 && (
              <div className={
                'w-8 h-px mx-1 ' + (step > n ? 'bg-[#1a5c1a]' : 'bg-[#1c1c1c]')
              } />
            )}
          </div>
        )
      })}
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────

const DEFAULT_SETTINGS = {
  tone: 'Cinematic',
  targetDuration: 45,
  transitionDuration: 0.3,
  mixOriginalAudio: false,
  showOpeningCard: true,
}

export default function App() {
  // ── Wizard step ─────────────────────────────────────────────────────────
  const [step, setStep] = useState(1) // 1 = Clips, 2 = Settings, 3 = Generate

  // ── Form state ─────────────────────────────────────────────────────────
  const [clips, setClips] = useState([])
  const [storyline, setStoryline] = useState('')
  const [musicInfo, setMusicInfo] = useState(null)
  const [musicTrim, setMusicTrim] = useState({ start: 0, end: null, aiSuggested: true })
  const [priorityKw, setPriorityKw] = useState('')
  const [excludeKw, setExcludeKw] = useState('')
  const [settings, setSettings] = useState(DEFAULT_SETTINGS)
  const [showKeywords, setShowKeywords] = useState(false)
  const [apiKeyInput, setApiKeyInput] = useState('')
  const [showApiKey, setShowApiKey] = useState(false)

  // ── Job state ───────────────────────────────────────────────────────────
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null) // null | running | done | error
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState('')
  const [clipAnalysis, setClipAnalysis] = useState([])
  const [error, setError] = useState('')
  const [startTime, setStartTime] = useState(null)
  const cleanupRef = useRef(null)

  // ── Cleanup SSE on unmount ──────────────────────────────────────────────
  useEffect(() => () => cleanupRef.current?.(), [])

  // ── Clip management ─────────────────────────────────────────────────────
  const handleAddClips = (newClips) => setClips((prev) => [...prev, ...newClips])
  const handleRemoveClip = (i) => setClips((prev) => prev.filter((_, idx) => idx !== i))

  // ── Generate ─────────────────────────────────────────────────────────────
  const handleGenerate = async () => {
    setError('')
    setJobStatus('running')
    setProgress(0)
    setMessage('Starting…')
    setClipAnalysis([])
    setJobId(null)
    setStartTime(Date.now())

    try {
      const { job_id } = await startGenerate({
        clipPaths: clips.map((c) => c.path),
        storyline,
        tone: settings.tone,
        targetDurationSec: settings.targetDuration,
        transitionDuration: settings.transitionDuration,
        mixOriginalAudio: settings.mixOriginalAudio,
        showOpeningCard: settings.showOpeningCard,
        musicPath: musicInfo?.path || null,
        musicStartSec: musicTrim.start,
        musicEndSec: musicTrim.end,
        priorityKeywords: priorityKw,
        excludeKeywords: excludeKw,
        apiKey: apiKeyInput || null,
      })
      setJobId(job_id)

      const cleanup = streamEvents(
        job_id,
        (data) => {
          if (typeof data.progress === 'number') setProgress(data.progress)
          if (data.message) setMessage(data.message)
        },
        async (doneData) => {
          setJobStatus(doneData.status)
          if (doneData.status === 'error' && doneData.error) {
            setMessage(doneData.error)
            setError(doneData.error)
          }
          try {
            const s = await pollStatus(job_id)
            if (s.clip_analysis?.length) setClipAnalysis(s.clip_analysis)
          } catch {}
        },
        (err) => {
          setJobStatus('error')
          setMessage(err.message)
          setError(err.message)
        }
      )
      cleanupRef.current = cleanup
    } catch (e) {
      setJobStatus('error')
      setMessage(e.message)
      setError(e.message)
    }
  }

  const isRunning = jobStatus === 'running'

  return (
    <div className="min-h-screen bg-[#080808]">
      {/* ── Header ────────────────────────────────────────────────── */}
      <header className="bg-gradient-to-br from-[#0e0e0e] to-[#150808] border-b border-[#1e1e1e] text-center py-7 mb-8">
        <h1 className="text-[1.55rem] font-black tracking-[0.14em] uppercase text-white">
          Keem<span className="text-[#e63939]">ography</span>
        </h1>
        <p className="text-[0.7rem] text-[#555] tracking-[0.22em] uppercase mt-1">
          AI Video Editor
        </p>
      </header>

      {/* ── Step indicator ────────────────────────────────────────── */}
      <StepBar step={step} onBack={(n) => setStep(n)} />

      {/* ── Step content ──────────────────────────────────────────── */}
      <main className="max-w-[640px] mx-auto px-4 pb-20">

        {/* ══════════════ STEP 1 — CLIPS ══════════════ */}
        {step === 1 && (
          <div>
            <Panel>
              <Label>Select Footage</Label>
              <UploadZone clips={clips} onAdd={handleAddClips} onRemove={handleRemoveClip} />
              <div className="mt-4">
                <p className="text-[0.78rem] text-[#555] mb-1.5">Or paste direct video URLs</p>
                <UrlFetcher onAdd={handleAddClips} />
              </div>
            </Panel>

            {clips.length > 0 && (
              <div className="mt-2 mb-4 px-1">
                <p className="text-[0.72rem] text-[#555]">
                  {clips.length} clip{clips.length > 1 ? 's' : ''} ready
                </p>
              </div>
            )}

            <Btn
              primary
              onClick={() => setStep(2)}
              disabled={clips.length === 0}
              className="w-full py-3.5 mt-2"
            >
              Next: Configure →
            </Btn>
          </div>
        )}

        {/* ══════════════ STEP 2 — SETTINGS ══════════════ */}
        {step === 2 && (
          <div>
            {/* Story */}
            <Panel>
              <Label>Story</Label>
              <Textarea
                value={storyline}
                onChange={(e) => setStoryline(e.target.value)}
                placeholder="e.g. A behind-the-scenes look at a golden-hour photoshoot — raw, emotional, cinematic."
                rows={4}
              />
              <p className="text-[0.72rem] text-[#444] mt-1.5">
                The AI uses this to rank and arrange your clips.
              </p>
            </Panel>

            <Divider />

            {/* Settings */}
            <Panel>
              <Label>Settings</Label>
              <SettingsPanel settings={settings} onChange={setSettings} />
            </Panel>

            <Divider />

            {/* Music */}
            <Panel>
              <Label>
                Music{' '}
                <span className="text-[#555] font-normal text-[0.72rem] normal-case tracking-normal">
                  (optional)
                </span>
              </Label>
              <MusicUpload
                musicInfo={musicInfo}
                musicTrim={musicTrim}
                onSet={(info) => {
                  setMusicInfo(info)
                  // Pre-populate trim from AI suggestion
                  const a = info?.analysis
                  if (a && a.duration > 0) {
                    setMusicTrim({ start: a.suggested_start, end: a.suggested_end, aiSuggested: true })
                  } else {
                    setMusicTrim({ start: 0, end: null, aiSuggested: true })
                  }
                }}
                onTrimChange={(trim) => setMusicTrim({ ...trim, aiSuggested: false })}
                onClear={() => { setMusicInfo(null); setMusicTrim({ start: 0, end: null, aiSuggested: true }) }}
              />
            </Panel>

            <Divider />

            {/* Keyword filters */}
            <div className="mb-4">
              <button
                onClick={() => setShowKeywords(!showKeywords)}
                className="text-[0.78rem] text-[#666] hover:text-[#e63939] transition-colors"
              >
                🔍 Keyword filters {showKeywords ? '▲' : '▼'} (optional)
              </button>
              {showKeywords && (
                <Panel className="mt-2">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <p className="text-[0.72rem] text-[#666] mb-1">Prioritize</p>
                      <Input
                        value={priorityKw}
                        onChange={(e) => setPriorityKw(e.target.value)}
                        placeholder="dance, culture, joy"
                      />
                    </div>
                    <div>
                      <p className="text-[0.72rem] text-[#666] mb-1">Exclude</p>
                      <Input
                        value={excludeKw}
                        onChange={(e) => setExcludeKw(e.target.value)}
                        placeholder="blurry, quiet"
                      />
                    </div>
                  </div>
                </Panel>
              )}
            </div>

            {/* API key override */}
            <div className="mb-6">
              <button
                onClick={() => setShowApiKey(!showApiKey)}
                className="text-[0.72rem] text-[#444] hover:text-[#e63939] transition-colors"
              >
                🔑 Override API key {showApiKey ? '▲' : '▼'}
              </button>
              {showApiKey && (
                <div className="mt-2">
                  <Input
                    value={apiKeyInput}
                    onChange={(e) => setApiKeyInput(e.target.value)}
                    placeholder="sk-… (uses server default)"
                    type="password"
                  />
                </div>
              )}
            </div>

            {/* Nav */}
            <div className="flex gap-3">
              <Btn onClick={() => setStep(1)} className="flex-1">← Back</Btn>
              <Btn
                primary
                onClick={() => setStep(3)}
                disabled={storyline.trim().length === 0}
                className="flex-[2] py-3.5"
              >
                Next: Generate →
              </Btn>
            </div>
            {storyline.trim().length === 0 && (
              <p className="text-[0.72rem] text-[#555] mt-2 text-center">
                Add a story description to continue
              </p>
            )}
          </div>
        )}

        {/* ══════════════ STEP 3 — GENERATE ══════════════ */}
        {step === 3 && (
          <div>
            {/* Summary */}
            {!isRunning && jobStatus !== 'done' && (
              <Panel>
                <Label>Ready to generate</Label>
                <div className="space-y-1.5 text-[0.82rem] text-[#999]">
                  <p>
                    <span className="text-[#555] mr-2">Clips</span>
                    <span className="text-white font-semibold">{clips.length}</span>
                  </p>
                  <p>
                    <span className="text-[#555] mr-2">Tone</span>
                    <span className="text-white font-semibold">{settings.tone}</span>
                  </p>
                  <p>
                    <span className="text-[#555] mr-2">Duration</span>
                    <span className="text-white font-semibold">{settings.targetDuration}s</span>
                  </p>
                  <p className="text-[#555] mt-2 italic leading-snug line-clamp-2">
                    "{storyline.slice(0, 120)}{storyline.length > 120 ? '…' : ''}"
                  </p>
                </div>
              </Panel>
            )}

            {/* Generate button */}
            {!isRunning && jobStatus !== 'done' && (
              <>
                <Btn
                  primary
                  onClick={handleGenerate}
                  disabled={isRunning}
                  className="w-full py-4 mt-2 text-[1.1rem]"
                >
                  ⬡  Generate Video
                </Btn>
                <div className="flex justify-center mt-3">
                  <Btn onClick={() => setStep(2)} className="text-xs px-3 py-1.5">
                    ← Edit settings
                  </Btn>
                </div>
              </>
            )}

            {error && !isRunning && (
              <p className="text-[#e63939] text-sm mt-3 bg-[#1a0808] border border-[#3a1212] rounded-xl px-4 py-3">
                {error}
              </p>
            )}

            {/* Progress */}
            {(isRunning || jobStatus === 'error') && (
              <ProgressPanel
                progress={progress}
                message={message}
                status={jobStatus}
                startTime={startTime}
              />
            )}

            {/* Clip analysis */}
            {clipAnalysis.length > 0 && (
              <>
                <Divider />
                <ClipAnalysis clips={clipAnalysis} />
              </>
            )}

            {/* Result */}
            {jobStatus === 'done' && jobId && (
              <>
                <ResultPanel jobId={jobId} />
                <div className="flex justify-center mt-4">
                  <Btn
                    onClick={() => {
                      setStep(1)
                      setJobStatus(null)
                      setJobId(null)
                      setProgress(0)
                      setMessage('')
                      setClipAnalysis([])
                      setError('')
                      setClips([])
                    }}
                    className="text-sm px-4 py-2"
                  >
                    ↩ Start new video
                  </Btn>
                </div>
              </>
            )}
          </div>
        )}

      </main>
    </div>
  )
}
