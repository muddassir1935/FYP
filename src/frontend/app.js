// SENTINEL Demo Console — vanilla JS, no framework.
// Talks to the FastAPI backend: status polling at 2 Hz, MJPEG stream
// for video, REST for everything else.

const $ = (id) => document.getElementById(id);

const els = {
  pill: $("state-pill"),
  elapsed: $("elapsed"),
  btnStop: $("btn-stop"),
  btnStart: $("btn-start"),
  btnShutdown: $("btn-shutdown"),
  backendBanner: $("backend-banner"),
  errorBox: $("error-box"),
  placeholder: $("video-placeholder"),
  stream: $("stream"),
  alertBanner: $("alert-banner"),
  alertText: $("alert-text"),
  recBadge: $("rec-badge"),
  btnFullscreen: $("btn-fullscreen"),
  videoPanel: $("video-panel"),
  videoSelect: $("video-select"),
  noVideos: $("no-videos"),
  webcamIndex: $("webcam-index"),
  weightsSelect: $("weights-select"),
  noWeights: $("no-weights"),
  uploadInput: $("upload-input"),
  uploadStatus: $("upload-status"),
  runForm: $("run-form"),
  runStatusLine: $("run-status-line"),
  snapshotsEmpty: $("snapshots-empty"),
  snapshotsGrid: $("snapshots-grid"),
  btnClearSnapshots: $("btn-clear-snapshots"),
  dialog: $("snapshot-dialog"),
  snapshotFull: $("snapshot-full"),
  snapshotCaption: $("snapshot-caption"),
  snapshotClose: $("snapshot-close"),
  statDisplayFps: $("stat-display-fps"),
  statDetectFps: $("stat-detect-fps"),
  statFrames: $("stat-frames"),
  statDetections: $("stat-detections"),
  statConfirmed: $("stat-confirmed"),
  statSnapshots: $("stat-snapshots"),
};

const state = {
  running: false,
  // Whether the <img> is attached to the current run's MJPEG stream.
  // Reset on every stop: the backend stream ends when a run stops,
  // and the <img> otherwise keeps showing the dead stream's last
  // frame through the next run (found live: a webcam frame stayed
  // stuck after switching to a video while the backend kept running).
  streamAttached: false,
  lastSnapshotCount: 0,
  pollFailures: 0,
  audioCtx: null,
  // A confirmation can be brief (one analyzed frame), so the banner
  // latches for a few seconds instead of flashing and vanishing.
  lastAlertAt: 0,
  // Set once the server is intentionally stopped: nothing more to poll.
  serverDown: false,
};

const ALERT_LATCH_MS = 5000;

function fmtElapsed(seconds) {
  const s = Math.max(0, Math.floor(seconds || 0));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function fmtStamp(stamp) {
  // "20261121_123045" -> "21 Nov 12:30:45"
  const m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/.exec(stamp || "");
  if (!m) return stamp || "";
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${m[3]} ${months[Number(m[2]) - 1]} ${m[4]}:${m[5]}:${m[6]}`;
}

function showError(message) {
  els.errorBox.textContent = message;
  els.errorBox.hidden = !message;
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      if (body && body.detail) detail = body.detail;
    } catch (_) { /* keep the generic message */ }
    throw new Error(detail);
  }
  return res.json();
}

// --- listings ----------------------------------------------------------

async function loadSources() {
  const videos = await api("/api/sources");
  els.videoSelect.innerHTML = "";
  for (const v of videos) {
    const opt = document.createElement("option");
    opt.value = v.name;
    const dur = v.duration_s ? ` — ${Math.round(v.duration_s)}s` : "";
    opt.textContent = `${v.name} (${v.size_mb} MB${dur})`;
    els.videoSelect.appendChild(opt);
  }
  els.noVideos.hidden = videos.length > 0;
  els.videoSelect.disabled = videos.length === 0;
}

async function loadWeights() {
  const weights = await api("/api/weights");
  els.weightsSelect.innerHTML = "";
  for (const w of weights) {
    const opt = document.createElement("option");
    opt.value = w.id;
    opt.textContent =
      w.kind === "openvino"
        ? `${w.label} (OpenVINO)`
        : w.kind === "directml"
          ? `${w.label} (DirectML / AMD GPU)`
          : `${w.label} (.pt)`;
    els.weightsSelect.appendChild(opt);
  }
  els.noWeights.hidden = weights.length > 0;
  els.weightsSelect.disabled = weights.length === 0;
  els.btnStart.disabled = weights.length === 0;
}

async function loadSnapshots() {
  const snaps = await api("/api/snapshots");
  els.snapshotsGrid.innerHTML = "";
  els.snapshotsEmpty.hidden = snaps.length > 0;
  for (const s of snaps) {
    const card = document.createElement("button");
    card.className = "snapshot-card";
    card.type = "button";
    card.innerHTML =
      `<img src="${s.url}" alt="${s.class} threat snapshot" loading="lazy">` +
      `<span class="snapshot-meta"><span class="thumb-class">${s.class}</span> · track ${s.track}<br>` +
      `<span class="thumb-time">${fmtStamp(s.timestamp)}</span></span>`;
    card.addEventListener("click", () => {
      els.snapshotFull.src = s.url;
      els.snapshotCaption.textContent = `${s.class} — track ${s.track} — ${fmtStamp(s.timestamp)}`;
      els.dialog.showModal();
    });
    els.snapshotsGrid.appendChild(card);
  }
  return snaps.length;
}

// --- siren (Web Audio; created only after a user gesture) --------------

function ensureAudio() {
  if (!state.audioCtx) {
    try {
      state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    } catch (_) { state.audioCtx = null; }
  }
  if (state.audioCtx && state.audioCtx.state === "suspended") {
    state.audioCtx.resume().catch(() => {});
  }
}

function playSiren() {
  if (!state.audioCtx) return;
  const ctx = state.audioCtx;
  const now = ctx.currentTime;
  // Two-tone siren, ~1.2 s, quiet enough for a demo room.
  for (let i = 0; i < 3; i++) {
    for (const [offset, freq] of [[0, 880], [0.2, 660]]) {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, now + i * 0.4 + offset);
      gain.gain.linearRampToValueAtTime(0.12, now + i * 0.4 + offset + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.4 + offset + 0.19);
      osc.connect(gain).connect(ctx.destination);
      osc.start(now + i * 0.4 + offset);
      osc.stop(now + i * 0.4 + offset + 0.2);
    }
  }
}

// --- run control --------------------------------------------------------

function setRunningUI(running) {
  state.running = running;
  els.btnStart.disabled = running || els.weightsSelect.disabled || state.serverDown;
  els.btnStop.disabled = !running;
  for (const el of els.runForm.querySelectorAll("input, select")) {
    if (el.id === "opt-sound") continue; // siren toggle stays usable
    el.disabled = running;
  }
  els.recBadge.hidden = !(running && $("opt-save").checked);
}

function detachStream() {
  // The backend MJPEG stream ends with the run; drop the <img>
  // attachment so the next run gets a fresh stream instead of the
  // previous run's last frame, and show the idle placeholder.
  state.streamAttached = false;
  state.lastAlertAt = 0;
  els.stream.hidden = true;
  els.stream.removeAttribute("src");
  els.placeholder.hidden = false;
}

function applyStatus(s) {
  const pill = els.pill;
  pill.className = "pill";
  if (s.state === "running") {
    pill.classList.add("pill-running");
    pill.textContent = "Running";
    setRunningUI(true);
    els.stream.hidden = false;
    els.placeholder.hidden = true;
    if (!state.streamAttached) {
      state.streamAttached = true;
      els.stream.src = `/api/stream?t=${Date.now()}`;
    }
  } else if (s.state === "error") {
    pill.classList.add("pill-error");
    pill.textContent = "Error";
    setRunningUI(false);
    detachStream();
    showError(s.error || "The run failed. Check the server log for details.");
  } else if (s.state === "stopped") {
    pill.classList.add("pill-stopped");
    pill.textContent = "Stopped";
    setRunningUI(false);
    detachStream();
    els.runStatusLine.textContent = s.saved_video
      ? `Run stopped. Video saved to ${s.saved_video}.`
      : "Run stopped.";
  } else {
    pill.classList.add("pill-idle");
    pill.textContent = "Idle";
    setRunningUI(false);
    detachStream();
  }

  els.elapsed.textContent = fmtElapsed(s.elapsed_s);
  els.statDisplayFps.textContent = s.display_fps ? s.display_fps.toFixed(1) : "—";
  els.statDetectFps.textContent = s.detect_fps ? s.detect_fps.toFixed(1) : "—";
  els.statFrames.textContent = s.frames || "—";
  els.statDetections.textContent = s.detections;
  els.statConfirmed.textContent = s.confirmed;
  els.statSnapshots.textContent = s.snapshots;

  const alerting = s.state === "running" && s.confirmed > 0;
  if (alerting) {
    state.lastAlertAt = Date.now();
    els.alertText.textContent =
      `Confirmed threat${s.confirmed > 1 ? "s" : ""}: ${s.confirmed} active — snapshot saved`;
  }
  const latched = Date.now() - state.lastAlertAt < ALERT_LATCH_MS;
  els.alertBanner.hidden = !(alerting || latched);
  if (latched && !alerting) {
    els.alertText.textContent = "Confirmed threat — snapshot saved";
  }

  if (s.snapshots > state.lastSnapshotCount) {
    state.lastSnapshotCount = s.snapshots;
    loadSnapshots().catch(() => {});
    if ($("opt-sound").checked) {
      ensureAudio();
      playSiren();
    }
  }
}

async function pollStatus() {
  try {
    const s = await api("/api/run/status");
    state.pollFailures = 0;
    els.backendBanner.hidden = true;
    applyStatus(s);
  } catch (_) {
    state.pollFailures += 1;
    if (state.pollFailures >= 2) els.backendBanner.hidden = false;
  }
}

els.runForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  ensureAudio();
  showError("");
  const sourceType = els.runForm.querySelector('input[name="source-type"]:checked').value;
  const body = {
    source_type: sourceType,
    source: sourceType === "webcam" ? els.webcamIndex.value : els.videoSelect.value,
    weights_id: els.weightsSelect.value,
    conf_gun: Number($("conf-gun").value),
    conf_knife: Number($("conf-knife").value),
    imgsz: Number($("imgsz").value),
    min_hits: Number($("min-hits").value),
    loop: $("opt-loop").checked,
    hide_unconfirmed: $("opt-hide").checked,
    save_video: $("opt-save").checked,
  };
  els.btnStart.disabled = true;
  els.runStatusLine.textContent = "Starting… (loading the model takes a few seconds)";
  try {
    const s = await api("/api/run/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.lastSnapshotCount = 0;
    els.runStatusLine.textContent = `Running: ${s.source_label} · ${s.weights_label}`;
    applyStatus(s);
  } catch (err) {
    showError(err.message);
    els.runStatusLine.textContent = "";
    els.btnStart.disabled = false;
  }
});

els.btnStop.addEventListener("click", async () => {
  els.btnStop.disabled = true;
  try {
    applyStatus(await api("/api/run/stop", { method: "POST" }));
  } catch (err) {
    showError(err.message);
    els.btnStop.disabled = false;
  }
});

// --- server shutdown ----------------------------------------------------

let pollTimer = null;

els.btnShutdown.addEventListener("click", async () => {
  if (!window.confirm("Stop the demo server? The console will stop responding. Start it again from the terminal when you need it.")) return;
  els.btnShutdown.disabled = true;
  try {
    await api("/api/server/shutdown", { method: "POST" });
    state.serverDown = true;
    if (pollTimer !== null) clearInterval(pollTimer);
    setRunningUI(false);
    els.btnShutdown.disabled = true;
    els.backendBanner.textContent = "Server stopped — start it again from the terminal to demo again.";
    els.backendBanner.hidden = false;
    els.runStatusLine.textContent = "Server stopped.";
  } catch (err) {
    showError(err.message);
    els.btnShutdown.disabled = false;
  }
});

// --- source type toggle -------------------------------------------------

els.runForm.querySelectorAll('input[name="source-type"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    const isVideo = els.runForm.querySelector('input[name="source-type"]:checked').value === "video";
    els.videoSelect.disabled = !isVideo || els.videoSelect.options.length === 0;
    els.webcamIndex.disabled = isVideo;
  });
});

// --- upload -------------------------------------------------------------

els.uploadInput.addEventListener("change", async () => {
  const file = els.uploadInput.files[0];
  if (!file) return;
  els.uploadStatus.textContent = `Uploading ${file.name}…`;
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Upload failed (${res.status}).`);
    }
    const saved = await res.json();
    await loadSources();
    els.videoSelect.value = saved.name;
    els.uploadStatus.textContent = `Uploaded ${saved.name}.`;
  } catch (err) {
    els.uploadStatus.textContent = err.message;
  } finally {
    els.uploadInput.value = "";
  }
});

// --- snapshots ----------------------------------------------------------

els.btnClearSnapshots.addEventListener("click", async () => {
  if (!window.confirm("Delete all snapshots?")) return;
  try {
    const res = await api("/api/snapshots/clear", { method: "POST" });
    state.lastSnapshotCount = 0;
    els.runStatusLine.textContent = `Removed ${res.removed} snapshot(s).`;
    await loadSnapshots();
  } catch (err) {
    showError(err.message);
  }
});

els.snapshotClose.addEventListener("click", () => els.dialog.close());
els.dialog.addEventListener("click", (event) => {
  if (event.target === els.dialog) els.dialog.close();
});

// --- fullscreen ---------------------------------------------------------

els.btnFullscreen.addEventListener("click", () => {
  if (document.fullscreenElement) {
    document.exitFullscreen().catch(() => {});
  } else {
    els.videoPanel.requestFullscreen().catch(() => {});
  }
});
document.addEventListener("fullscreenchange", () => {
  els.btnFullscreen.textContent = document.fullscreenElement ? "Exit fullscreen" : "Fullscreen";
});

// --- init -----------------------------------------------------------------

(async function init() {
  try {
    await Promise.all([loadSources(), loadWeights(), loadSnapshots()]);
    const s = await api("/api/run/status");
    state.lastSnapshotCount = s.snapshots;
    // applyStatus attaches the stream when the state is running, so a
    // page reloaded mid-run reattaches through the same code path.
    applyStatus(s);
  } catch (err) {
    showError(`Could not reach the backend: ${err.message}`);
    els.backendBanner.hidden = false;
  }
  pollTimer = setInterval(pollStatus, 500);
  // Snapshots can also appear without a status change (older runs); refresh rarely.
  setInterval(() => { if (!state.running) loadSnapshots().catch(() => {}); }, 10000);
})();
