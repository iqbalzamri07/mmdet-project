(() => {
  const $ = (id) => document.getElementById(id);

  const state = {
    labels: [],
    postures: ["sitting", "standing"],
    activities: [],
    counts: {},
    totalCount: 0,
    videos: [],
    videoQuery: "",
    videoId: null,
    meta: null,
    segments: [],
    pendingStart: null,
    pendingEnd: null,
    cropMode: false,
    cropDraft: null, // {x1,y1,x2,y2} in video pixel space
    drawing: false,
    drawOrigin: null,
    trainJobId: null,
    pollTimer: null,
  };

  const videoEl = $("video");
  const overlay = $("overlay");
  const ctx = overlay.getContext("2d");

  function toast(msg, type = "ok") {
    const el = $("toast");
    el.hidden = false;
    el.className = `toast ${type}`;
    el.textContent = msg;
    clearTimeout(el._t);
    el._t = setTimeout(() => {
      el.hidden = true;
    }, 3200);
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail || data.error || res.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  }

  function isPosture(name) {
    return (state.postures || []).includes(name);
  }

  function isActivity(name) {
    return (state.activities || []).includes(name);
  }

  function displayLabel(seg) {
    const parts = [seg.posture, seg.activity].filter(Boolean);
    if (parts.length) return parts.join(" + ");
    return seg.label || "—";
  }

  function normalizeSeg(seg) {
    const next = { ...seg };
    if (next.posture || next.activity) {
      next.label = displayLabel(next);
      return next;
    }
    const legacy = next.label || "";
    const parts = legacy
      .split(/[+,]/)
      .map((s) => s.trim())
      .filter(Boolean);
    parts.forEach((part) => {
      if (isPosture(part) && !next.posture) next.posture = part;
      else if (isActivity(part) && !next.activity) next.activity = part;
    });
    if (!next.posture && !next.activity && legacy) {
      if (isPosture(legacy)) next.posture = legacy;
      else next.activity = legacy;
    }
    next.label = displayLabel(next);
    return next;
  }

  function renderChip(label) {
    const count = state.counts[label] || 0;
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `<span class="chip-name">${label}</span><span class="chip-count">${count}</span>`;
    return chip;
  }

  function fillSelect(select, values, extra) {
    const selected = select.value;
    select.innerHTML = "";
    (extra || []).forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.value;
      opt.textContent = item.label;
      select.appendChild(opt);
    });
    values.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = `${name} (${state.counts[name] || 0})`;
      select.appendChild(opt);
    });
    if (selected && [...select.options].some((o) => o.value === selected)) {
      select.value = selected;
    }
  }

  function renderLabels() {
    const chips = $("labelChips");
    chips.innerHTML = "";
    const groups = [
      { title: "Posture", items: state.postures || [] },
      { title: "Activity", items: state.activities || [] },
    ];
    groups.forEach((group) => {
      if (!group.items.length) return;
      const wrap = document.createElement("div");
      wrap.innerHTML = `<span class="chip-group-title">${group.title}</span>`;
      const row = document.createElement("div");
      row.className = "chip-row";
      group.items.forEach((label) => row.appendChild(renderChip(label)));
      wrap.appendChild(row);
      chips.appendChild(wrap);
    });

    const postureSel = $("postureSelect");
    const activitySel = $("activitySelect");
    if (postureSel) fillSelect(postureSel, state.postures || []);
    if (activitySel) {
      fillSelect(activitySel, state.activities || [], [{ value: "", label: "none" }]);
    }
    const totalEl = $("classCountTotal");
    if (totalEl) {
      const total = state.totalCount ?? Object.values(state.counts).reduce((a, b) => a + b, 0);
      totalEl.textContent = `${total} annotated`;
    }
  }

  async function loadLabelCounts() {
    const data = await api("/api/labels");
    state.labels = data.labels || [];
    state.postures = data.postures || ["sitting", "standing"];
    state.activities = data.activities || state.labels.filter((l) => !state.postures.includes(l));
    state.counts = data.counts || {};
    state.totalCount = data.total || 0;
    renderLabels();
  }

  function filteredVideos() {
    const q = (state.videoQuery || "").trim().toLowerCase();
    if (!q) return state.videos;
    return state.videos.filter((v) => {
      const name = (v.filename || "").toLowerCase();
      const id = (v.id || "").toLowerCase();
      return name.includes(q) || id.includes(q);
    });
  }

  function appendVideoItem(list, v) {
    const li = document.createElement("li");
    if (v.id === state.videoId) li.classList.add("active");
    li.innerHTML = `
      <div class="video-row">
        <div class="video-info">
          <div class="name">${v.filename}</div>
          <div class="meta">${Math.round(v.duration || 0)}s · ${v.segments || 0} segments · ${v.total_frames || 0} frames</div>
        </div>
        <button type="button" class="btn-delete-video" title="Delete video" aria-label="Delete ${v.filename}">×</button>
      </div>`;
    li.querySelector(".video-info").onclick = () => selectVideo(v.id);
    li.querySelector(".btn-delete-video").onclick = (e) => {
      e.stopPropagation();
      deleteVideo(v.id, v.filename);
    };
    list.appendChild(li);
  }

  function appendVideoGroup(list, items) {
    list.innerHTML = "";
    if (!items.length) {
      const empty = document.createElement("li");
      empty.className = "meta video-group-empty";
      empty.textContent = "None";
      list.appendChild(empty);
      return;
    }
    items.forEach((v) => appendVideoItem(list, v));
  }

  function renderVideoList() {
    const needList = $("videoListNeed");
    const hasList = $("videoListHas");
    const countEl = $("libraryCount");
    const videos = filteredVideos();
    const q = (state.videoQuery || "").trim();
    const total = state.videos.length;
    const unlabeled = videos.filter((v) => !(v.segments > 0));
    const labeled = videos.filter((v) => v.segments > 0);
    const labeledAll = state.videos.filter((v) => (v.segments || 0) > 0).length;
    if (countEl) {
      const labeledBit = `${labeledAll} labeled`;
      if (q) {
        countEl.textContent = `${videos.length} of ${total} · ${labeledBit}`;
      } else {
        countEl.textContent = `${total} video${total === 1 ? "" : "s"} · ${labeledBit}`;
      }
    }
    const needTitle = $("needLabelsTitle");
    const hasTitle = $("hasSegmentsTitle");
    if (needTitle) needTitle.textContent = `Need labels (${unlabeled.length})`;
    if (hasTitle) hasTitle.textContent = `Has segments (${labeled.length})`;
    if (!state.videos.length) {
      needList.innerHTML = '<li class="meta">No videos yet</li>';
      hasList.innerHTML = "";
      if (hasTitle) hasTitle.textContent = "Has segments (0)";
      return;
    }
    if (!videos.length) {
      needList.innerHTML = '<li class="meta">No videos match that search</li>';
      hasList.innerHTML = '<li class="meta">No videos match that search</li>';
      return;
    }
    appendVideoGroup(needList, unlabeled);
    appendVideoGroup(hasList, labeled);
  }

  async function deleteVideo(id, filename) {
    const ok = window.confirm(`Delete "${filename}" and its annotations?`);
    if (!ok) return;
    try {
      await api(`/api/videos/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (state.videoId === id) {
        state.videoId = null;
        state.meta = null;
        state.segments = [];
        videoEl.removeAttribute("src");
        videoEl.load();
        $("stageActive").classList.add("hidden");
        $("stageEmpty").classList.remove("hidden");
        $("segmentsPanel").classList.add("hidden");
      }
      await loadVideos();
      await loadLabelCounts();
      toast("Video deleted", "ok");
    } catch (err) {
      toast(err.message || "Delete failed", "error");
    }
  }

  function renderSegments() {
    const body = $("segBody");
    body.innerHTML = "";
    state.segments.forEach((seg, idx) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${seg.posture || "—"}</td>
        <td>${seg.activity || "—"}</td>
        <td>${seg.start_frame}</td>
        <td>${seg.end_frame}</td>
        <td>${seg.bbox ? "yes" : "—"}</td>
        <td><button type="button" data-idx="${idx}">Delete</button></td>`;
      tr.querySelector("button").onclick = () => {
        state.segments.splice(idx, 1);
        renderSegments();
        drawTimeline();
      };
      body.appendChild(tr);
    });
    drawTimeline();
  }

  function updatePending() {
    $("pendingInfo").textContent = `Start: ${state.pendingStart ?? "—"} · End: ${state.pendingEnd ?? "—"}`;
  }

  function currentFrame() {
    const fps = state.meta?.fps || 30;
    return Math.max(0, Math.floor(videoEl.currentTime * fps));
  }

  function seekToFrame(frame) {
    const fps = state.meta?.fps || 30;
    const total = state.meta?.total_frames || 1;
    const f = Math.min(Math.max(0, frame), Math.max(0, total - 1));
    videoEl.currentTime = f / fps;
  }

  function syncSeek() {
    if (!state.meta) return;
    const total = Math.max(1, state.meta.total_frames - 1);
    const frame = currentFrame();
    $("seek").value = String(Math.round((frame / total) * 1000));
    $("timecode").textContent = `${frame} / ${state.meta.total_frames}`;
    drawOverlay();
  }

  function resizeOverlay() {
    const rect = overlay.parentElement.getBoundingClientRect();
    overlay.width = rect.width * devicePixelRatio;
    overlay.height = rect.height * devicePixelRatio;
    overlay.style.width = `${rect.width}px`;
    overlay.style.height = `${rect.height}px`;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    drawOverlay();
  }

  function videoDisplayRect() {
    const wrap = overlay.parentElement.getBoundingClientRect();
    const vw = videoEl.videoWidth || state.meta?.width || 1;
    const vh = videoEl.videoHeight || state.meta?.height || 1;
    const scale = Math.min(wrap.width / vw, wrap.height / vh);
    const dw = vw * scale;
    const dh = vh * scale;
    const ox = (wrap.width - dw) / 2;
    const oy = (wrap.height - dh) / 2;
    return { ox, oy, dw, dh, vw, vh, scale };
  }

  function clientToVideo(x, y) {
    const r = videoDisplayRect();
    const vx = (x - r.ox) / r.scale;
    const vy = (y - r.oy) / r.scale;
    return {
      x: Math.min(Math.max(0, vx), r.vw),
      y: Math.min(Math.max(0, vy), r.vh),
    };
  }

  function drawOverlay() {
    const wrap = overlay.parentElement.getBoundingClientRect();
    ctx.clearRect(0, 0, wrap.width, wrap.height);
    const r = videoDisplayRect();
    const bbox = state.cropDraft;
    if (!bbox) return;
    const x = r.ox + (bbox.x1 / r.vw) * r.dw;
    const y = r.oy + (bbox.y1 / r.vh) * r.dh;
    const w = ((bbox.x2 - bbox.x1) / r.vw) * r.dw;
    const h = ((bbox.y2 - bbox.y1) / r.vh) * r.dh;
    ctx.fillStyle = "rgba(14, 118, 110, 0.18)";
    ctx.strokeStyle = "#0f766e";
    ctx.lineWidth = 2;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
  }

  function drawTimeline() {
    const el = $("timelineMarks");
    el.innerHTML = "";
    if (!state.meta?.total_frames) return;
    const total = state.meta.total_frames;
    state.segments.forEach((seg) => {
      const span = document.createElement("span");
      const left = (seg.start_frame / total) * 100;
      const width = Math.max(0.4, ((seg.end_frame - seg.start_frame + 1) / total) * 100);
      span.style.left = `${left}%`;
      span.style.width = `${width}%`;
      span.title = `${displayLabel(seg)} ${seg.start_frame}-${seg.end_frame}`;
      el.appendChild(span);
    });
  }

  async function loadLabels() {
    await loadLabelCounts();
  }

  async function loadVideos() {
    const data = await api("/api/videos");
    state.videos = data.videos || [];
    renderVideoList();
  }

  async function selectVideo(id) {
    state.videoId = id;
    state.pendingStart = null;
    state.pendingEnd = null;
    state.cropDraft = null;
    updatePending();
    $("stageEmpty").classList.add("hidden");
    $("stageActive").classList.remove("hidden");
    $("segmentsPanel").classList.remove("hidden");
    renderVideoList();

    const meta = await api(`/api/videos/${id}/meta`);
    state.meta = meta;
    state.segments = (meta.segments || []).map((s) => normalizeSeg(s));
    videoEl.onerror = async () => {
      toast("Video codec not supported in browser — converting to H.264…", "error");
      try {
        await api(`/api/videos/${id}/transcode`, { method: "POST" });
        videoEl.src = `/api/videos/${id}/file?t=${Date.now()}`;
        videoEl.load();
        toast("Converted to H.264 — try play again", "ok");
      } catch (err) {
        toast(err.message || "Transcode failed", "error");
      }
    };
    videoEl.src = `/api/videos/${id}/file?t=${Date.now()}`;
    videoEl.load();
    await videoEl.play().catch(() => {});
    videoEl.pause();
    videoEl.onloadedmetadata = () => {
      resizeOverlay();
      syncSeek();
    };
    renderSegments();
    toast(`Loaded ${meta.filename}`);
  }

  async function saveAnnotations() {
    if (!state.videoId || !state.meta) return;
    const payload = {
      filename: state.meta.filename,
      fps: state.meta.fps,
      width: state.meta.width,
      height: state.meta.height,
      total_frames: state.meta.total_frames,
      duration: state.meta.duration,
      segments: state.segments,
    };
    await api(`/api/annotations/${state.videoId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await loadVideos();
    await loadLabelCounts();
    toast("Annotations saved", "ok");
  }

  // --- Events ---
  $("fileInput").addEventListener("change", async (e) => {
    const files = [...(e.target.files || [])];
    e.target.value = "";
    if (!files.length) return;
    const uploaded = [];
    const skipped = [];
    const failed = [];
    for (let i = 0; i < files.length; i++) {
      toast(`Uploading ${i + 1}/${files.length}: ${files[i].name}`, "ok");
      const fd = new FormData();
      fd.append("file", files[i]);
      try {
        const data = await api("/api/videos/upload", { method: "POST", body: fd });
        if (data.skipped) skipped.push(data);
        else uploaded.push(data);
      } catch (err) {
        failed.push(`${files[i].name}: ${err.message || "failed"}`);
      }
    }
    try {
      await loadVideos();
      const lastNew = uploaded[uploaded.length - 1];
      if (lastNew?.video) {
        await selectVideo(lastNew.video.id);
      }
    } catch (err) {
      toast(err.message, "error");
      return;
    }
    if (!uploaded.length && !skipped.length) {
      toast(failed[0] || "Upload failed", "error");
    } else if (failed.length) {
      toast(`Uploaded ${uploaded.length}, skipped ${skipped.length}, ${failed.length} failed`, "error");
    } else if (uploaded.length === 1 && uploaded[0].converted_to_h264) {
      toast(uploaded[0].message || "Converted to H.264 for playback", "ok");
    } else if (uploaded.length && skipped.length) {
      toast(`Uploaded ${uploaded.length}, skipped ${skipped.length} (already in library)`, "ok");
    } else if (skipped.length && !uploaded.length) {
      toast(
        skipped.length === 1
          ? skipped[0].message || "Already in library"
          : `Skipped ${skipped.length} — already in library`,
        "ok"
      );
    } else {
      toast(uploaded.length === 1 ? "Video uploaded" : `Uploaded ${uploaded.length} videos`, "ok");
    }
  });

  $("btnRefresh").onclick = () => loadVideos().catch((e) => toast(e.message, "error"));

  $("videoSearch").addEventListener("input", (e) => {
    state.videoQuery = e.target.value;
    renderVideoList();
  });

  $("btnAddLabel").onclick = async () => {
    const input = $("newLabelInput");
    const name = input.value.trim().replace(/\s+/g, "_");
    if (!name) return;
    if (state.labels.includes(name)) {
      toast("Label already exists", "error");
      return;
    }
    if (isPosture(name)) {
      toast("sitting and standing are postures, not activities", "error");
      return;
    }
    const labels = [...state.labels, name];
    const activities = [...(state.activities || []), name];
    try {
      const data = await api("/api/labels", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          labels,
          postures: state.postures,
          activities,
        }),
      });
      state.labels = data.labels;
      await loadLabelCounts();
      input.value = "";
      toast(`Added activity ${name}`);
    } catch (err) {
      toast(err.message, "error");
    }
  };

  $("btnPlay").onclick = () => {
    if (videoEl.paused) videoEl.play();
    else videoEl.pause();
    $("btnPlay").textContent = videoEl.paused ? "▶" : "❚❚";
  };

  $("btnPrev").onclick = () => {
    seekToFrame(currentFrame() - 1);
  };
  $("btnNext").onclick = () => {
    seekToFrame(currentFrame() + 1);
  };

  $("seek").oninput = (e) => {
    if (!state.meta) return;
    const total = Math.max(1, state.meta.total_frames - 1);
    const frame = Math.round((Number(e.target.value) / 1000) * total);
    seekToFrame(frame);
  };

  videoEl.addEventListener("timeupdate", syncSeek);
  videoEl.addEventListener("seeked", syncSeek);
  videoEl.addEventListener("play", () => {
    $("btnPlay").textContent = "❚❚";
  });
  videoEl.addEventListener("pause", () => {
    $("btnPlay").textContent = "▶";
  });

  $("btnMarkStart").onclick = () => {
    state.pendingStart = currentFrame();
    updatePending();
  };
  $("btnMarkEnd").onclick = () => {
    state.pendingEnd = currentFrame();
    updatePending();
  };

  $("btnSaveSeg").onclick = () => {
    if (state.pendingStart == null || state.pendingEnd == null) {
      toast("Mark start and end first", "error");
      return;
    }
    let start = state.pendingStart;
    let end = state.pendingEnd;
    if (end < start) [start, end] = [end, start];
    const posture = $("postureSelect").value;
    const activity = $("activitySelect").value;
    if (!posture) {
      toast("Pick a posture (sitting or standing)", "error");
      return;
    }
    const seg = {
      id: Math.random().toString(36).slice(2, 10),
      posture,
      activity,
      label: [posture, activity].filter(Boolean).join(" + "),
      start_frame: start,
      end_frame: end,
      bbox: state.cropDraft
        ? [state.cropDraft.x1, state.cropDraft.y1, state.cropDraft.x2, state.cropDraft.y2]
        : null,
      note: "",
    };
    state.segments.push(seg);
    state.pendingStart = null;
    state.pendingEnd = null;
    updatePending();
    renderSegments();
    toast(`Saved ${seg.label} ${start}–${end}`);
  };

  $("btnSaveAll").onclick = () => saveAnnotations().catch((e) => toast(e.message, "error"));

  $("btnToggleCrop").onclick = () => {
    state.cropMode = !state.cropMode;
    $("btnToggleCrop").textContent = state.cropMode ? "Crop: ON" : "Draw crop";
    $("btnToggleCrop").classList.toggle("btn-primary", state.cropMode);
    overlay.classList.toggle("crop-on", state.cropMode);
    toast(state.cropMode ? "Drag on video to draw crop box" : "Crop mode off");
  };

  $("btnClearCrop").onclick = () => {
    state.cropDraft = null;
    drawOverlay();
  };

  overlay.addEventListener("mousedown", (e) => {
    if (!state.cropMode) return;
    const rect = overlay.getBoundingClientRect();
    const pt = clientToVideo(e.clientX - rect.left, e.clientY - rect.top);
    state.drawing = true;
    state.drawOrigin = pt;
    state.cropDraft = { x1: pt.x, y1: pt.y, x2: pt.x, y2: pt.y };
  });

  overlay.addEventListener("mousemove", (e) => {
    if (!state.drawing || !state.drawOrigin) return;
    const rect = overlay.getBoundingClientRect();
    const pt = clientToVideo(e.clientX - rect.left, e.clientY - rect.top);
    state.cropDraft = {
      x1: Math.min(state.drawOrigin.x, pt.x),
      y1: Math.min(state.drawOrigin.y, pt.y),
      x2: Math.max(state.drawOrigin.x, pt.x),
      y2: Math.max(state.drawOrigin.y, pt.y),
    };
    drawOverlay();
  });

  window.addEventListener("mouseup", () => {
    state.drawing = false;
  });

  window.addEventListener("resize", resizeOverlay);

  $("btnExport").onclick = async () => {
    try {
      await saveAnnotations().catch(() => {});
      const summary = await api("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clear_existing: true, sync_labels: true }),
      });
      toast(`Exported ${summary.clips} clips`, "ok");
    } catch (err) {
      toast(err.message, "error");
    }
  };

  async function pollTrain() {
    try {
      const data = await api(
        state.trainJobId
          ? `/api/train/status?job_id=${encodeURIComponent(state.trainJobId)}`
          : "/api/train/status"
      );
      const job = data.job;
      const pill = $("trainPill");
      if (!job) {
        pill.textContent = "idle";
        pill.className = "status-pill";
        $("btnStopTrain").hidden = true;
        return;
      }
      state.trainJobId = job.job_id;
      pill.textContent = job.status;
      pill.className = `status-pill ${job.status}`;
      $("btnStopTrain").hidden = !["running", "training", "exporting", "queued"].includes(job.status);
      $("btnStopTrain").dataset.jobId = job.job_id;
      const log = await api(`/api/train/log/${job.job_id}?tail=80`);
      $("trainLog").textContent = (log.lines || []).join("\n") || "(waiting for logs…)";
      if (["completed", "failed", "stopped"].includes(job.status)) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        if (job.status === "completed") toast("Training completed", "ok");
        if (job.status === "failed") toast(job.error || "Training failed", "error");
      }
    } catch (err) {
      console.error(err);
    }
  }

  $("btnTrain").onclick = async () => {
    try {
      await saveAnnotations().catch(() => {});
      const data = await api("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ export_first: true, sync_labels: true }),
      });
      state.trainJobId = data.job.job_id;
      toast("Training started");
      if (state.pollTimer) clearInterval(state.pollTimer);
      state.pollTimer = setInterval(pollTrain, 2500);
      pollTrain();
    } catch (err) {
      toast(err.message, "error");
    }
  };

  $("btnStopTrain").onclick = async () => {
    const id = $("btnStopTrain").dataset.jobId;
    if (!id) return;
    try {
      await api(`/api/train/stop/${id}`, { method: "POST" });
      toast("Stop requested");
      pollTrain();
    } catch (err) {
      toast(err.message, "error");
    }
  };

  // Keyboard shortcuts (label mode only)
  window.addEventListener("keydown", (e) => {
    if (document.getElementById("modeLabel").classList.contains("hidden")) return;
    if (e.target.matches("input, textarea, select")) return;
    if (e.code === "Space") {
      e.preventDefault();
      $("btnPlay").click();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      $("btnPrev").click();
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      $("btnNext").click();
    } else if (e.key === "s" || e.key === "S") {
      $("btnMarkStart").click();
    } else if (e.key === "e" || e.key === "E") {
      $("btnMarkEnd").click();
    } else if (e.key === "Enter") {
      $("btnSaveSeg").click();
    }
  });

  // ----- Test mode -----
  const testState = {
    file: null,
    jobId: null,
    pollTimer: null,
  };

  function setMode(mode) {
    const isLabel = mode === "label";
    $("modeLabel").classList.toggle("hidden", !isLabel);
    $("modeTest").classList.toggle("hidden", isLabel);
    $("labelActions").classList.toggle("hidden", !isLabel);
    $("tabLabel").classList.toggle("active", isLabel);
    $("tabTest").classList.toggle("active", !isLabel);
    if (!isLabel) loadModels();
  }

  $("tabLabel").onclick = () => setMode("label");
  $("tabTest").onclick = () => setMode("test");

  async function loadModels() {
    try {
      const data = await api("/api/models");
      const sel = $("modelSelect");
      sel.innerHTML = "";
      const models = data.models || [];
      if (!models.length) {
        sel.innerHTML = '<option value="">No checkpoints found — train first</option>';
        $("modelHint").textContent = "No .pth files in work_dirs/slowfast_multilabel/";
        return;
      }
      models.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.name;
        opt.textContent = `${m.name}${m.recommended ? " ★" : ""} (${m.size_mb} MB)`;
        sel.appendChild(opt);
      });
      const best = models.find((m) => m.name.startsWith("best_acc")) || models[0];
      sel.value = best.name;
      $("modelHint").textContent = `Selected: ${best.name}`;
    } catch (err) {
      toast(err.message, "error");
    }
  }

  $("btnRefreshModels").onclick = () => loadModels();
  $("modelSelect").onchange = () => {
    $("modelHint").textContent = `Selected: ${$("modelSelect").value}`;
  };

  $("testFileInput").onchange = (e) => {
    const file = e.target.files?.[0] || null;
    testState.file = file;
    $("testFileName").textContent = file ? file.name : "No file selected";
  };

  async function pollTest() {
    if (!testState.jobId) return;
    try {
      const data = await api(`/api/test/status?job_id=${encodeURIComponent(testState.jobId)}`);
      const job = data.job;
      if (!job) return;
      const pill = $("testPill");
      pill.textContent = job.status;
      pill.className = `status-pill ${job.status}`;
      const lines = job.log || [];
      $("testLog").textContent = lines.length ? lines.join("\n") : `(${job.status})`;

      if (job.status === "completed") {
        clearInterval(testState.pollTimer);
        testState.pollTimer = null;
        showTestResult(job);
        toast("Inference complete", "ok");
      } else if (job.status === "failed") {
        clearInterval(testState.pollTimer);
        testState.pollTimer = null;
        toast(job.error || "Inference failed", "error");
      }
    } catch (err) {
      console.error(err);
    }
  }

  function showTestResult(job) {
    $("testEmpty").classList.add("hidden");
    $("testActive").classList.remove("hidden");
    const url = `/api/test/result/${job.job_id}/video?t=${Date.now()}`;
    const vid = $("resultVideo");
    vid.src = url;
    vid.load();
    $("btnDownloadResult").href = url;
    $("btnDownloadResult").download = `${job.job_id}.mp4`;

    const body = $("predBody");
    body.innerHTML = "";
    const persons = job.persons || [];
    if (!persons.length) {
      body.innerHTML = '<tr><td colspan="5">No persons detected</td></tr>';
      return;
    }
    persons.forEach((p) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>P${p.id}</td>
        <td>${p.posture || "—"}</td>
        <td>${p.activity || "—"}</td>
        <td>${(p.score * 100).toFixed(1)}%</td>
        <td>${p.frames}</td>`;
      body.appendChild(tr);
    });
  }

  $("btnRunTest").onclick = async () => {
    const checkpoint = $("modelSelect").value;
    if (!checkpoint) {
      toast("Select a model checkpoint", "error");
      return;
    }
    if (!testState.file) {
      toast("Upload a test video first", "error");
      return;
    }
    try {
      $("testPill").textContent = "starting";
      $("testPill").className = "status-pill running";
      $("testLog").textContent = "Uploading and starting inference…";
      const fd = new FormData();
      fd.append("file", testState.file);
      fd.append("checkpoint", checkpoint);
      const res = await fetch("/api/test/run", {
        method: "POST",
        body: fd,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || data.error || res.statusText);
      }
      testState.jobId = data.job.job_id;
      toast("Inference started — this may take a few minutes");
      if (testState.pollTimer) clearInterval(testState.pollTimer);
      testState.pollTimer = setInterval(pollTest, 2500);
      pollTest();
    } catch (err) {
      toast(err.message, "error");
      $("testPill").textContent = "failed";
      $("testPill").className = "status-pill failed";
    }
  };

  async function init() {
    try {
      await loadLabels();
      await loadVideos();
      await pollTrain();
      if (state.trainJobId) {
        state.pollTimer = setInterval(pollTrain, 2500);
      }
    } catch (err) {
      toast(err.message, "error");
    }
  }

  init();
})();
