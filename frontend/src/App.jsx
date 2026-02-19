import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { HistoryList } from "./components/HistoryList";
import { MetricCard } from "./components/MetricCard";
import { PredictionCard } from "./components/PredictionCard";
import { WaveformPulse } from "./components/WaveformPulse";
import { useWebcamCapture } from "./hooks/useWebcamCapture";
import { useWebsocket } from "./hooks/useWebsocket";

const DEFAULT_WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws";
const TARGET_FPS = 12;
const STABLE_FRAME_COUNT = 4;
const MIN_COMMIT_CONFIDENCE = 0.55;
const COMMIT_COOLDOWN_MS = 900;
const MAX_HISTORY = 30;
const MAX_LOGS = 120;

const NO_SIGN_TOKENS = new Set(["NO_SIGN", "NO SIGN", "NONE", ""]);

function isNoSign(token) {
  if (!token) return true;
  return NO_SIGN_TOKENS.has(String(token).trim().toUpperCase());
}

function formatWsStatus(status, isConnected) {
  if (isConnected) return "Connected";
  if (status === "connecting") return "Connecting";
  if (status === "error") return "Error";
  return "Disconnected";
}

function getThemeFromSystem() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function createHistoryItem(token, confidence) {
  return {
    id: `${token}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    token,
    confidence,
    timestamp: Date.now(),
  };
}

function createLogEntry(level, message) {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    timestamp: Date.now(),
    level,
    message,
  };
}

function wsToHttpBase(input) {
  try {
    const url = new URL(input);
    const protocol = url.protocol === "wss:" ? "https:" : "http:";
    return `${protocol}//${url.host}`;
  } catch {
    return "http://localhost:8000";
  }
}

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("isl-theme") || "light");
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS_URL);
  const [streaming, setStreaming] = useState(false);
  const [mirrorPreview, setMirrorPreview] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState([]);
  const [unstable, setUnstable] = useState({ token: "NO_SIGN", confidence: 0 });
  const [committed, setCommitted] = useState({ token: "NO_SIGN", confidence: 0 });
  const [stability, setStability] = useState({ progress: 0, isStabilizing: true });
  const [history, setHistory] = useState([]);
  const [sentenceTokens, setSentenceTokens] = useState([]);
  const [pulseLevel, setPulseLevel] = useState(0);
  const [metrics, setMetrics] = useState({ sendFps: 0, recvFps: 0, latencyMs: 0 });
  const [uploadFile, setUploadFile] = useState(null);
  const [sampleFps, setSampleFps] = useState(15);
  const [maxFrames, setMaxFrames] = useState(600);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadResult, setUploadResult] = useState(null);

  const sendTimesRef = useRef([]);
  const recvTimesRef = useRef([]);
  const latencyTimesRef = useRef([]);
  const sendLockRef = useRef(false);
  const wsStatusRef = useRef("");
  const stabilizerRef = useRef({
    candidateToken: "",
    candidateCount: 0,
    lastCommittedToken: "",
    lastCommitTs: 0,
  });

  const appendLog = useCallback((level, message) => {
    setLogs((prev) => [createLogEntry(level, message), ...prev].slice(0, MAX_LOGS));
  }, []);

  const {
    videoRef,
    cameraState,
    error: cameraError,
    startCamera,
    stopCamera,
    captureFrame,
  } = useWebcamCapture({ width: 640, height: 480, facingMode: "user", jpegQuality: 0.72 });

  const handleMessage = useCallback((message) => {
    if (!message) {
      return;
    }

    if (message.type === "error") {
      appendLog("error", `Backend error: ${message.message || "Unknown error"}`);
      return;
    }

    if (message.type !== "prediction") {
      appendLog("info", `WS message type: ${String(message.type || "unknown")}`);
      return;
    }

    const now = Date.now();
    const token = String(message.token ?? "NO_SIGN");
    const confidence = Number(message.confidence ?? 0);
    const liveToken = String(message.debug?.live_token ?? token);
    const liveConfidence = Number(message.debug?.live_confidence ?? confidence);

    recvTimesRef.current.push(now);
    recvTimesRef.current = recvTimesRef.current.filter((ts) => now - ts <= 1000);

    const latency = Number(message.latency_ms ?? 0);
    if (latency > 0) {
      latencyTimesRef.current.push(latency);
      latencyTimesRef.current = latencyTimesRef.current.slice(-20);
    }

    appendLog(
      "info",
      `Prediction live=${liveToken} (${(liveConfidence * 100).toFixed(0)}%) committed=${token} latency=${latency}ms`,
    );

    setUnstable({ token: liveToken, confidence: liveConfidence });

    if (isNoSign(liveToken)) {
      stabilizerRef.current.candidateToken = "";
      stabilizerRef.current.candidateCount = 0;
      setStability({ progress: 0, isStabilizing: true });
      setPulseLevel(0.1);
      return;
    }

    if (stabilizerRef.current.candidateToken === liveToken) {
      stabilizerRef.current.candidateCount += 1;
    } else {
      stabilizerRef.current.candidateToken = liveToken;
      stabilizerRef.current.candidateCount = 1;
    }

    const stabilizedByFrames = stabilizerRef.current.candidateCount >= STABLE_FRAME_COUNT;
    const stabilizedByServer = Boolean(message.is_committed) && !isNoSign(token);
    const readyToCommit =
      (stabilizedByFrames && liveConfidence >= MIN_COMMIT_CONFIDENCE) ||
      (stabilizedByServer && confidence >= MIN_COMMIT_CONFIDENCE);

    setStability({
      progress: Math.min(1, stabilizerRef.current.candidateCount / STABLE_FRAME_COUNT),
      isStabilizing: !readyToCommit,
    });
    setPulseLevel(Math.max(0.2, Math.min(1, liveConfidence)));

    if (!readyToCommit) {
      return;
    }

    const commitToken = stabilizedByServer ? token : liveToken;
    const commitConfidence = stabilizedByServer ? confidence : liveConfidence;
    const canCommitAgain =
      commitToken !== stabilizerRef.current.lastCommittedToken ||
      now - stabilizerRef.current.lastCommitTs > COMMIT_COOLDOWN_MS;

    if (!canCommitAgain) {
      return;
    }

    stabilizerRef.current.lastCommittedToken = commitToken;
    stabilizerRef.current.lastCommitTs = now;
    stabilizerRef.current.candidateCount = 0;
    stabilizerRef.current.candidateToken = "";

    setCommitted({ token: commitToken, confidence: commitConfidence });
    setHistory((prev) => [createHistoryItem(commitToken, commitConfidence), ...prev].slice(0, MAX_HISTORY));
    setSentenceTokens((prev) => [...prev, commitToken].slice(-80));
    setPulseLevel(1);
    appendLog("success", `Committed token=${commitToken} (${(commitConfidence * 100).toFixed(0)}%)`);
  }, [appendLog]);

  const { status: wsStatus, isConnected, connect, disconnect, sendBinary } = useWebsocket({
    url: wsUrl,
    onMessage: handleMessage,
  });

  const uploadEndpoint = useMemo(() => `${wsToHttpBase(wsUrl)}/predict/video`, [wsUrl]);

  useEffect(() => {
    const initialTheme = theme === "light" && !localStorage.getItem("isl-theme") ? getThemeFromSystem() : theme;
    setTheme(initialTheme);
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("isl-theme", theme);
  }, [theme]);

  useEffect(() => {
    const id = setInterval(() => {
      const now = Date.now();
      sendTimesRef.current = sendTimesRef.current.filter((ts) => now - ts <= 1000);
      recvTimesRef.current = recvTimesRef.current.filter((ts) => now - ts <= 1000);
      const latencySamples = latencyTimesRef.current;
      const latencyAvg =
        latencySamples.length > 0
          ? latencySamples.reduce((sum, value) => sum + value, 0) / latencySamples.length
          : 0;

      setMetrics({
        sendFps: sendTimesRef.current.length,
        recvFps: recvTimesRef.current.length,
        latencyMs: latencyAvg,
      });
    }, 450);

    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!streaming || !isConnected || cameraState !== "active") {
      return;
    }

    const frameIntervalMs = Math.round(1000 / TARGET_FPS);
    const timer = setInterval(async () => {
      if (sendLockRef.current) return;
      sendLockRef.current = true;

      try {
        const blob = await captureFrame();
        if (!blob) return;
        const bytes = await blob.arrayBuffer();
        const ok = sendBinary(bytes);
        if (ok) {
          const now = Date.now();
          sendTimesRef.current.push(now);
          sendTimesRef.current = sendTimesRef.current.filter((ts) => now - ts <= 1000);
        }
      } finally {
        sendLockRef.current = false;
      }
    }, frameIntervalMs);

    return () => clearInterval(timer);
  }, [cameraState, captureFrame, isConnected, sendBinary, streaming]);

  useEffect(() => {
    return () => {
      stopCamera();
      disconnect();
    };
  }, [disconnect, stopCamera]);

  useEffect(() => {
    const nextLabel = formatWsStatus(wsStatus, isConnected);
    if (wsStatusRef.current !== nextLabel) {
      wsStatusRef.current = nextLabel;
      appendLog("info", `WebSocket ${nextLabel.toLowerCase()}`);
    }
  }, [appendLog, isConnected, wsStatus]);

  const handleConnect = useCallback(() => {
    connect();
    appendLog("info", `Connecting to ${wsUrl}`);
  }, [appendLog, connect, wsUrl]);

  const handleDisconnect = useCallback(() => {
    disconnect();
    appendLog("info", "Disconnected WebSocket");
  }, [appendLog, disconnect]);

  const handleToggleCamera = useCallback(async () => {
    if (cameraState === "active") {
      stopCamera();
      appendLog("info", "Camera stopped");
      return;
    }
    await startCamera();
    appendLog("info", "Camera start requested");
  }, [appendLog, cameraState, startCamera, stopCamera]);

  const handleToggleStream = useCallback(() => {
    setStreaming((prev) => {
      const next = !prev;
      appendLog("info", next ? "Frame streaming started" : "Frame streaming stopped");
      return next;
    });
  }, [appendLog]);

  const handleUploadTest = useCallback(async () => {
    if (!uploadFile) {
      setUploadError("Select a video file first.");
      appendLog("error", "Upload test requested without file");
      return;
    }

    setUploading(true);
    setUploadError("");
    setUploadResult(null);
    appendLog("info", `Upload test started for ${uploadFile.name}`);

    try {
      const query = new URLSearchParams({
        sample_fps: String(sampleFps),
        max_frames: String(maxFrames),
      });
      const formData = new FormData();
      formData.append("file", uploadFile);

      const response = await fetch(`${uploadEndpoint}?${query.toString()}`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        const message = data?.detail || "Upload test failed.";
        throw new Error(String(message));
      }

      setUploadResult(data);
      appendLog(
        "success",
        `Upload result final=${String(data.final_token || "NO_SIGN")} conf=${Number(data.final_confidence || 0).toFixed(2)}`,
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Upload test failed.";
      setUploadError(message);
      appendLog("error", `Upload test error: ${message}`);
    } finally {
      setUploading(false);
    }
  }, [appendLog, maxFrames, sampleFps, uploadEndpoint, uploadFile]);

  const sentence = useMemo(() => sentenceTokens.join(" "), [sentenceTokens]);
  const wsLabel = formatWsStatus(wsStatus, isConnected);
  const streamReady = cameraState === "active" && isConnected;
  const waveActive = pulseLevel > 0.25 && !isNoSign(unstable.token);

  return (
    <main className="min-h-screen px-4 py-6 md:px-8">
      <div className="mx-auto max-w-7xl">
        <header className="mb-6 grid gap-4 rounded-3xl border border-white/40 bg-white/65 p-6 shadow-soft backdrop-blur-xl dark:border-white/10 dark:bg-slate-900/60">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h1 className="font-display text-3xl font-bold tracking-tight">Indian Sign Language Live Recognition</h1>
              <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                Real-time webcam streaming, stabilized token commits, and sentence building.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span
                className={`rounded-full px-3 py-1 text-xs font-semibold ${
                  isConnected
                    ? "bg-emerald-500/20 text-emerald-700 dark:text-emerald-200"
                    : wsStatus === "connecting"
                      ? "bg-amber-400/25 text-amber-700 dark:text-amber-200"
                      : "bg-slate-500/20 text-slate-700 dark:text-slate-300"
                }`}
                aria-label={`WebSocket status ${wsLabel}`}
              >
                WS: {wsLabel}
              </span>
              <button
                type="button"
                onClick={() => setShowLogs((prev) => !prev)}
                className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-semibold transition hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800"
                aria-label="Toggle live logs"
              >
                {showLogs ? "Hide Logs" : "Check Logs"}
              </button>
              <button
                type="button"
                onClick={() => setTheme((prev) => (prev === "light" ? "dark" : "light"))}
                className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-semibold transition hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800"
                aria-label="Toggle dark and light theme"
              >
                {theme === "light" ? "Dark mode" : "Light mode"}
              </button>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-[1fr_auto_auto_auto_auto_auto]">
            <input
              value={wsUrl}
              onChange={(event) => setWsUrl(event.target.value)}
              className="rounded-xl border border-slate-300 bg-white/80 px-4 py-2 text-sm outline-none ring-0 transition focus:border-teal-500 dark:border-slate-700 dark:bg-slate-900/60"
              placeholder="ws://localhost:8000/ws"
              aria-label="WebSocket URL"
            />
            <button
              type="button"
              onClick={handleConnect}
              className="rounded-xl bg-teal-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-teal-500 disabled:cursor-not-allowed disabled:opacity-60"
              disabled={isConnected}
              aria-label="Connect WebSocket"
            >
              Connect
            </button>
            <button
              type="button"
              onClick={handleDisconnect}
              className="rounded-xl bg-slate-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-600 dark:bg-slate-600 dark:hover:bg-slate-500"
              disabled={!isConnected && wsStatus !== "error"}
              aria-label="Disconnect WebSocket"
            >
              Disconnect
            </button>
            <button
              type="button"
              onClick={handleToggleCamera}
              className="rounded-xl bg-orange-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-orange-400"
              aria-label={cameraState === "active" ? "Stop camera" : "Start camera"}
            >
              {cameraState === "active" ? "Stop Camera" : "Start Camera"}
            </button>
            <button
              type="button"
              onClick={handleToggleStream}
              disabled={!streamReady}
              className="rounded-xl bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
              aria-label={streaming ? "Stop streaming frames" : "Start streaming frames"}
            >
              {streaming ? "Stop Stream" : "Start Stream"}
            </button>
            <button
              type="button"
              onClick={() => {
                setMirrorPreview((prev) => {
                  const next = !prev;
                  appendLog("info", next ? "Camera preview mirrored (visual only)" : "Camera preview normal");
                  return next;
                });
              }}
              className="rounded-xl border border-slate-300 bg-white/80 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-100 dark:border-slate-600 dark:bg-slate-900/60 dark:text-slate-200 dark:hover:bg-slate-800"
              aria-label={mirrorPreview ? "Disable mirrored camera preview" : "Enable mirrored camera preview"}
            >
              {mirrorPreview ? "Unmirror Camera" : "Mirror Camera"}
            </button>
          </div>
          {cameraError && <p className="text-sm text-rose-600 dark:text-rose-300">{cameraError}</p>}
          <p className="text-xs text-slate-500 dark:text-slate-300">
            Mirror affects preview only. Streamed frames and model input are unchanged.
          </p>
        </header>

        {showLogs && (
          <section className="glass-panel mb-6 p-5" aria-label="Live logs">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="font-display text-lg font-bold tracking-tight">Live Logs</h2>
              <button
                type="button"
                onClick={() => setLogs([])}
                className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium transition hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800"
                aria-label="Clear logs"
              >
                Clear logs
              </button>
            </div>
            <div className="max-h-64 overflow-auto rounded-2xl border border-slate-200/80 bg-white/70 p-3 dark:border-slate-700 dark:bg-slate-900/50">
              {logs.length === 0 ? (
                <p className="text-sm text-slate-600 dark:text-slate-300">No logs yet.</p>
              ) : (
                <ul className="space-y-2" aria-label="Log events list">
                  {logs.map((entry) => (
                    <li
                      key={entry.id}
                      className={`rounded-lg border px-3 py-2 text-xs ${
                        entry.level === "error"
                          ? "border-rose-300 bg-rose-50 text-rose-700 dark:border-rose-900 dark:bg-rose-900/20 dark:text-rose-200"
                          : entry.level === "success"
                            ? "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-900/20 dark:text-emerald-200"
                            : "border-slate-200 bg-white text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
                      }`}
                    >
                      <span className="mr-2 font-semibold">
                        {new Date(entry.timestamp).toLocaleTimeString()}
                      </span>
                      <span>{entry.message}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        )}

        <section className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <div className="space-y-6">
            <article className="glass-panel p-5">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="font-display text-lg font-bold tracking-tight">Live Camera</h2>
                <span className="text-xs text-slate-600 dark:text-slate-300">
                  {cameraState === "active" ? `${TARGET_FPS} FPS target` : "Camera inactive"}
                </span>
              </div>
              <div className="relative overflow-hidden rounded-2xl border border-slate-200/70 bg-slate-950 dark:border-slate-700">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`h-[320px] w-full object-cover md:h-[430px] ${mirrorPreview ? "-scale-x-100" : ""}`}
                  aria-label="Live webcam preview for sign recognition"
                />
                <div className="absolute inset-x-0 bottom-0 flex items-center justify-between bg-gradient-to-t from-black/70 to-transparent px-4 py-3 text-xs text-white">
                  <span>Live: {unstable.token}</span>
                  <span>Committed: {committed.token}</span>
                </div>
              </div>
            </article>

            <article className="glass-panel p-5">
              <h2 className="mb-3 font-display text-lg font-bold tracking-tight">Sentence Builder</h2>
              <div
                className="min-h-24 rounded-2xl border border-dashed border-slate-300/90 bg-white/70 p-4 text-base leading-relaxed dark:border-slate-700 dark:bg-slate-900/50"
                aria-live="polite"
                aria-label="Built sentence from committed tokens"
              >
                {sentence || "Committed tokens will appear here as a running sentence."}
              </div>
              <div className="mt-3 flex gap-2">
                <button
                  type="button"
                  onClick={() => setSentenceTokens([])}
                  className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium transition hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800"
                  aria-label="Clear sentence builder"
                >
                  Clear sentence
                </button>
              </div>
            </article>
          </div>

          <div className="space-y-6">
            <PredictionCard
              unstableToken={unstable.token}
              unstableConfidence={unstable.confidence}
              committedToken={committed.token}
              committedConfidence={committed.confidence}
              stabilizationProgress={stability.progress}
              isStabilizing={stability.isStabilizing}
            />

            <section className="glass-panel p-5" aria-label="Performance metrics">
              <h2 className="mb-4 font-display text-lg font-bold tracking-tight">Performance</h2>
              <div className="grid gap-3 sm:grid-cols-3">
                <MetricCard label="Send FPS" value={String(metrics.sendFps)} ariaLabel="Outbound frame rate" />
                <MetricCard label="Recv FPS" value={String(metrics.recvFps)} ariaLabel="Inbound prediction rate" />
                <MetricCard
                  label="Latency"
                  value={`${Math.round(metrics.latencyMs)} ms`}
                  ariaLabel="Average model latency in milliseconds"
                />
              </div>
            </section>

            <WaveformPulse active={waveActive} intensity={pulseLevel} />

            <HistoryList history={history} onClear={() => setHistory([])} />
          </div>
        </section>

        <section className="glass-panel mt-6 p-5" aria-label="Uploaded video testing">
          <div className="mb-4">
            <h2 className="font-display text-lg font-bold tracking-tight">Uploaded Testing</h2>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
              Test a recorded video using backend endpoint <code>{uploadEndpoint}</code>.
            </p>
          </div>

          <div className="grid gap-3 md:grid-cols-[1.5fr_0.6fr_0.6fr_auto]">
            <input
              type="file"
              accept="video/*"
              onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)}
              className="rounded-xl border border-slate-300 bg-white/80 px-4 py-2 text-sm dark:border-slate-700 dark:bg-slate-900/60"
              aria-label="Upload video file for testing"
            />
            <input
              type="number"
              min="1"
              step="1"
              value={sampleFps}
              onChange={(event) => setSampleFps(Number(event.target.value || 1))}
              className="rounded-xl border border-slate-300 bg-white/80 px-4 py-2 text-sm dark:border-slate-700 dark:bg-slate-900/60"
              aria-label="Sample FPS"
              placeholder="Sample FPS"
            />
            <input
              type="number"
              min="1"
              step="1"
              value={maxFrames}
              onChange={(event) => setMaxFrames(Number(event.target.value || 1))}
              className="rounded-xl border border-slate-300 bg-white/80 px-4 py-2 text-sm dark:border-slate-700 dark:bg-slate-900/60"
              aria-label="Maximum frames"
              placeholder="Max frames"
            />
            <button
              type="button"
              onClick={handleUploadTest}
              disabled={uploading}
              className="rounded-xl bg-teal-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-teal-500 disabled:cursor-not-allowed disabled:opacity-60"
              aria-label="Run upload test"
            >
              {uploading ? "Testing..." : "Run Upload Test"}
            </button>
          </div>

          {uploadFile && (
            <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">
              Selected: {uploadFile.name}
            </p>
          )}
          {uploadError && <p className="mt-2 text-sm text-rose-600 dark:text-rose-300">{uploadError}</p>}

          {uploadResult && (
            <div className="mt-4 rounded-2xl border border-slate-200/80 bg-white/70 p-4 dark:border-slate-700 dark:bg-slate-900/50">
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  label="Final Token"
                  value={String(uploadResult.final_token || "NO_SIGN")}
                  ariaLabel="Upload final token"
                />
                <MetricCard
                  label="Final Conf"
                  value={`${Math.round(Number(uploadResult.final_confidence || 0) * 100)}%`}
                  ariaLabel="Upload final confidence"
                />
                <MetricCard
                  label="Windows"
                  value={String(uploadResult.inferred_windows ?? 0)}
                  ariaLabel="Inferred windows count"
                />
                <MetricCard
                  label="Frames"
                  value={String(uploadResult.sampled_frames ?? 0)}
                  ariaLabel="Sampled frame count"
                />
              </div>
              {Array.isArray(uploadResult.timeline) && uploadResult.timeline.length > 0 && (
                <p className="mt-3 text-xs text-slate-600 dark:text-slate-300">
                  Timeline events: {uploadResult.timeline.length}
                </p>
              )}
            </div>
          )}
        </section>
      </div>

      <p className="sr-only" aria-live="polite">
        Current live token {unstable.token} with confidence {(unstable.confidence * 100).toFixed(0)} percent.
      </p>
    </main>
  );
}
