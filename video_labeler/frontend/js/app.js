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
    videoPage: { need: 1, has: 1 },
    videoPaging: { total_all: 0, total_labeled: 0, needTotal: 0, hasTotal: 0, needPages: 1, hasPages: 1 },
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
    editingClasses: false,
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
    chip.className = state.editingClasses ? "chip chip-editing" : "chip";
    chip.innerHTML = `<span class="chip-name">${label}</span><span class="chip-count">${count}</span>`;
    if (state.editingClasses) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip-remove";
      btn.title = `Remove ${label}`;
      btn.setAttribute("aria-label", `Remove ${label}`);
      btn.textContent = "×";
      btn.onclick = (e) => {
        e.stopPropagation();
        removeClass(label);
      };
      chip.appendChild(btn);
    }
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
      wrap.className = "chip-group";
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
    const editBtn = $("btnEditLabels");
    if (editBtn) {
      editBtn.textContent = state.editingClasses ? "Done" : "Edit";
      editBtn.classList.toggle("btn-primary", state.editingClasses);
      editBtn.classList.toggle("btn-ghost", !state.editingClasses);
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

  async function removeClass(name) {
    const count = state.counts[name] || 0;
    const kind = isPosture(name) ? "posture" : "activity";
    if (isPosture(name) && (state.postures || []).length <= 1) {
      toast("Keep at least one posture", "error");
      return;
    }
    const extra = count ? ` It is used in ${count} segment${count === 1 ? "" : "s"} (those tags stay until you re-save).` : "";
    if (!window.confirm(`Remove ${kind} "${name}"?${extra}`)) return;
    try {
      await api(`/api/labels/${encodeURIComponent(name)}`, { method: "DELETE" });
      await loadLabelCounts();
      toast(`Removed ${name}`, "ok");
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function appendVideoItem(list, v) {
    const li = document.createElement("li");
    if (v.id === state.videoId) li.classList.add("active");
    li.innerHTML = `
      <div class="video-row">
        <div class="video-info">
          <div class="name" title="${v.filename}">${v.filename}</div>
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

  function renderVideoList() {
    const needList = $("videoListNeed");
    const hasList = $("videoListHas");
    const countEl = $("libraryCount");
    const { total_all, total_labeled, needTotal, hasTotal, needPages, hasPages } = state.videoPaging;
    if (countEl) {
      const q = (state.videoQuery || "").trim();
      if (q) {
        countEl.textContent = `${needTotal + hasTotal} results · ${total_all} total · ${total_labeled} with segments`;
      } else {
        countEl.textContent = `${total_all} video${total_all === 1 ? "" : "s"} · ${total_labeled} with segments`;
      }
    }
    const needTitle = $("needLabelsTitle");
    const hasTitle = $("hasSegmentsTitle");
    if (needTitle) needTitle.textContent = `${needTotal} video${needTotal === 1 ? "" : "s"}`;
    if (hasTitle) hasTitle.textContent = `${hasTotal} video${hasTotal === 1 ? "" : "s"}`;
    const libTabNeed = $("libTabNeed");
    const libTabHas = $("libTabHas");
    if (libTabNeed) libTabNeed.textContent = `Need segments (${needTotal})`;
    if (libTabHas) libTabHas.textContent = `Has segments (${hasTotal})`;

    needList.innerHTML = "";
    hasList.innerHTML = "";

    const needVideos = state.videos.filter((v) => !(v.segments > 0));
    const hasVideos = state.videos.filter((v) => v.segments > 0);

    if (!needVideos.length && !hasVideos.length && !total_all) {
      needList.innerHTML = '<li class="meta">No videos yet</li>';
      hasList.innerHTML = "";
      return;
    }
    if (!needVideos.length) {
      needList.innerHTML = '<li class="meta">None on this page</li>';
    } else {
      needVideos.forEach((v) => appendVideoItem(needList, v));
    }
    if (state.videoPage.need < needPages) {
      const btn = document.createElement("li");
      btn.className = "load-more";
      btn.innerHTML = `<button class="btn btn-ghost btn-small" type="button">Load more…</button>`;
      btn.querySelector("button").onclick = () => {
        state.videoPage.need++;
        fetchVideos(false);
      };
      needList.appendChild(btn);
    }

    if (!hasVideos.length) {
      hasList.innerHTML = '<li class="meta">None on this page</li>';
    } else {
      hasVideos.forEach((v) => appendVideoItem(hasList, v));
    }
    if (state.videoPage.has < hasPages) {
      const btn = document.createElement("li");
      btn.className = "load-more";
      btn.innerHTML = `<button class="btn btn-ghost btn-small" type="button">Load more…</button>`;
      btn.querySelector("button").onclick = () => {
        state.videoPage.has++;
        fetchVideos(false);
      };
      hasList.appendChild(btn);
    }
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
        $("modeLabel")?.classList.remove("segments-open");
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
        <td><button type="button" data-idx="${idx}">Delete</button></td>`;
      tr.querySelector("button").onclick = () => {
        state.segments.splice(idx, 1);
        renderSegments();
        drawTimeline();
      };
      body.appendChild(tr);
    });
    drawTimeline();
    drawOverlay();
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

  function relayoutPlayers() {
    requestAnimationFrame(() => {
      if (!$("stageActive").classList.contains("hidden")) resizeOverlay();
      if (testState.cameraOn) drawLiveOverlay();
    });
  }

  function updateLabelLayout() {
    const layout = $("modeLabel");
    const panel = $("segmentsPanel");
    if (!layout || !panel) return;
    layout.classList.toggle("segments-open", !panel.classList.contains("hidden"));
    relayoutPlayers();
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

  function parseBbox(bbox) {
    if (!bbox || bbox.length !== 4) return null;
    let [x1, y1, x2, y2] = bbox.map(Number);
    const vw = videoEl.videoWidth || state.meta?.width || 1;
    const vh = videoEl.videoHeight || state.meta?.height || 1;
    if (Math.max(x1, y1, x2, y2) <= 1.5) {
      x1 *= vw;
      x2 *= vw;
      y1 *= vh;
      y2 *= vh;
    }
    return { x1, y1, x2, y2 };
  }

  function segmentsAtFrame(frame) {
    return state.segments.filter((s) => s.start_frame <= frame && frame <= s.end_frame);
  }

  function drawBboxOnCanvas(bbox, style) {
    const rect = typeof bbox?.x1 === "number" ? bbox : parseBbox(bbox);
    if (!rect) return;
    const r = videoDisplayRect();
    const x = r.ox + (rect.x1 / r.vw) * r.dw;
    const y = r.oy + (rect.y1 / r.vh) * r.dh;
    const w = ((rect.x2 - rect.x1) / r.vw) * r.dw;
    const h = ((rect.y2 - rect.y1) / r.vh) * r.dh;
    ctx.fillStyle = style.fill;
    ctx.strokeStyle = style.stroke;
    ctx.lineWidth = style.lineWidth ?? 2;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
  }

  function drawOverlay() {
    const wrap = overlay.parentElement.getBoundingClientRect();
    ctx.clearRect(0, 0, wrap.width, wrap.height);
    const frame = currentFrame();
    segmentsAtFrame(frame).forEach((seg) => {
      if (seg.bbox) {
        drawBboxOnCanvas(seg.bbox, {
          fill: "rgba(224, 90, 60, 0.15)",
          stroke: "#e05a3c",
        });
      }
    });
    if (state.cropDraft) {
      drawBboxOnCanvas(state.cropDraft, {
        fill: "rgba(14, 118, 110, 0.18)",
        stroke: "#0f766e",
      });
    }
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

  const PER_PAGE = 50;

  async function fetchVideos(reset = true) {
    if (reset) {
      state.videoPage = { need: 1, has: 1 };
      state.videos = [];
    }
    const q = (state.videoQuery || "").trim();
    const params = new URLSearchParams({ per_page: PER_PAGE });
    if (q) params.set("q", q);

    params.set("page", state.videoPage.need);
    params.set("labeled", "false");
    const needData = await api(`/api/videos?${params}`);

    params.set("page", state.videoPage.has);
    params.set("labeled", "true");
    const hasData = await api(`/api/videos?${params}`);

    const needVideos = needData.videos || [];
    const hasVideos = hasData.videos || [];

    if (reset) {
      state.videos = [...needVideos, ...hasVideos];
    } else {
      const existingIds = new Set(state.videos.map((v) => v.id));
      needVideos.forEach((v) => { if (!existingIds.has(v.id)) state.videos.push(v); });
      hasVideos.forEach((v) => { if (!existingIds.has(v.id)) state.videos.push(v); });
    }

    state.videoPaging = {
      total_all: needData.total_all || 0,
      total_labeled: needData.total_labeled || 0,
      needTotal: needData.total || 0,
      hasTotal: hasData.total || 0,
      needPages: needData.pages || 1,
      hasPages: hasData.pages || 1,
    };
    renderVideoList();
  }

  async function loadVideos() {
    await fetchVideos(true);
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
    updateLabelLayout();
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
    relayoutPlayers();
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
        setNavTab("videos");
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

  function setNavTab(tab) {
    const tabs = { videos: "navTabVideos", dataset: "navTabDataset", classes: "navTabClasses" };
    const panels = { videos: "navPanelVideos", dataset: "navPanelDataset", classes: "navPanelClasses" };
    Object.entries(tabs).forEach(([key, btnId]) => {
      const active = key === tab;
      $(btnId)?.classList.toggle("active", active);
      const panel = $(panels[key]);
      if (panel) {
        panel.hidden = !active;
        panel.classList.toggle("active", active);
      }
    });
  }

  $("navTabVideos")?.addEventListener("click", () => setNavTab("videos"));
  $("navTabDataset")?.addEventListener("click", () => setNavTab("dataset"));
  $("navTabClasses")?.addEventListener("click", () => setNavTab("classes"));

  function setLibraryTab(tab) {
    const isNeed = tab === "need";
    $("libTabNeed")?.classList.toggle("active", isNeed);
    $("libTabHas")?.classList.toggle("active", !isNeed);
    $("libTabNeed")?.setAttribute("aria-selected", isNeed ? "true" : "false");
    $("libTabHas")?.setAttribute("aria-selected", !isNeed ? "true" : "false");
    $("libraryPanelNeed")?.classList.toggle("active", isNeed);
    $("libraryPanelHas")?.classList.toggle("active", !isNeed);
    if ($("libraryPanelNeed")) $("libraryPanelNeed").hidden = !isNeed;
    if ($("libraryPanelHas")) $("libraryPanelHas").hidden = isNeed;
  }

  $("libTabNeed")?.addEventListener("click", () => setLibraryTab("need"));
  $("libTabHas")?.addEventListener("click", () => setLibraryTab("has"));

  let _searchTimer = null;
  $("videoSearch").addEventListener("input", (e) => {
    state.videoQuery = e.target.value;
    clearTimeout(_searchTimer);
    _searchTimer = setTimeout(() => fetchVideos(true), 300);
  });

  $("btnEditLabels").onclick = () => {
    state.editingClasses = !state.editingClasses;
    renderLabels();
  };

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
    const btn = $("btnToggleCrop");
    btn.classList.toggle("is-active", state.cropMode);
    btn.setAttribute("aria-pressed", state.cropMode ? "true" : "false");
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

  window.addEventListener("resize", relayoutPlayers);

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
        if (job.status === "completed") loadModels();
      }
    } catch (err) {
      console.error(err);
    }
  }

  $("btnTrain").onclick = async () => {
    try {
      await saveAnnotations().catch(() => {});
      const raw = Number($("trainEpochs")?.value || 100);
      const epochs = Math.round(raw);
      if (!Number.isFinite(epochs) || epochs < 1 || epochs > 300) {
        toast("Epochs must be between 1 and 300", "error");
        return;
      }
      const data = await api("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ export_first: true, sync_labels: true, epochs }),
      });
      state.trainJobId = data.job.job_id;
      toast(`Training started (${epochs} epochs)`);
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
    source: "video",
    cameraOn: false,
    stream: null,
    liveBusy: false,
    lastPersons: [],
    libraryQuery: "",
    libraryPage: 1,
    libraryResults: [],
    libraryPaging: { total: 0, pages: 1 },
  };

  function formatTestDate(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso.slice(0, 16).replace("T", " ");
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function setTestNavTab(tab) {
    const isRun = tab === "run";
    $("testNavRun")?.classList.toggle("active", isRun);
    $("testNavResults")?.classList.toggle("active", !isRun);
    $("testPanelRun")?.classList.toggle("active", isRun);
    $("testPanelResults")?.classList.toggle("active", !isRun);
    if ($("testPanelRun")) $("testPanelRun").hidden = !isRun;
    if ($("testPanelResults")) $("testPanelResults").hidden = isRun;
    if (!isRun) loadTestLibrary().catch((e) => toast(e.message, "error"));
  }

  $("testNavRun")?.addEventListener("click", () => setTestNavTab("run"));
  $("testNavResults")?.addEventListener("click", () => setTestNavTab("results"));

  function renderTestLibrary() {
    const list = $("testResultList");
    if (!list) return;
    list.innerHTML = "";
    const items = testState.libraryResults || [];
    const total = testState.libraryPaging?.total || 0;
    $("testLibraryCount").textContent = `${total} result${total === 1 ? "" : "s"}`;

    if (!items.length) {
      list.innerHTML = '<li class="meta">No processed videos yet — run inference first.</li>';
      return;
    }

    items.forEach((item) => {
      const li = document.createElement("li");
      li.className = "test-result-item";
      if (item.job_id === testState.jobId) li.classList.add("active");
      const name = item.source_name || item.job_id;
      li.innerHTML = `
        <div class="video-row">
          <div class="video-info">
            <div class="name" title="${name}">${name}</div>
            <div class="meta">${formatTestDate(item.finished_at)} · ${item.checkpoint}</div>
            <div class="meta-line">
              <span class="badge done">${item.person_count} person${item.person_count === 1 ? "" : "s"}</span>
              <span class="badge">${item.summary}</span>
            </div>
          </div>
        </div>`;
      li.querySelector(".video-info").onclick = () => {
        selectTestResult(item.job_id).catch((e) => toast(e.message, "error"));
      };
      list.appendChild(li);
    });

    const page = testState.libraryPage;
    const pages = testState.libraryPaging?.pages || 1;
    if (page < pages) {
      const more = document.createElement("li");
      more.className = "load-more";
      more.innerHTML = `<button class="btn btn-ghost btn-small" type="button">Load more…</button>`;
      more.querySelector("button").onclick = () => {
        testState.libraryPage++;
        loadTestLibrary(false).catch((e) => toast(e.message, "error"));
      };
      list.appendChild(more);
    }
  }

  async function loadTestLibrary(reset = true) {
    if (reset) testState.libraryPage = 1;
    const params = new URLSearchParams({
      page: testState.libraryPage,
      per_page: 50,
    });
    const q = (testState.libraryQuery || "").trim();
    if (q) params.set("q", q);
    const data = await api(`/api/test/library?${params}`);
    const results = data.results || [];
    if (reset || testState.libraryPage === 1) {
      testState.libraryResults = results;
    } else {
      const seen = new Set(testState.libraryResults.map((r) => r.job_id));
      results.forEach((r) => {
        if (!seen.has(r.job_id)) testState.libraryResults.push(r);
      });
    }
    testState.libraryPaging = {
      total: data.total || 0,
      pages: data.pages || 1,
    };
    renderTestLibrary();
  }

  async function selectTestResult(jobId) {
    const data = await api(`/api/test/status?job_id=${encodeURIComponent(jobId)}`);
    if (!data.job) throw new Error("Result not found");
    testState.jobId = jobId;
    showTestResult(data.job);
    renderTestLibrary();
  }

  $("btnRefreshTestLibrary")?.addEventListener("click", () => {
    loadTestLibrary(true).catch((e) => toast(e.message, "error"));
  });

  let _testSearchTimer = null;
  $("testLibrarySearch")?.addEventListener("input", (e) => {
    testState.libraryQuery = e.target.value;
    clearTimeout(_testSearchTimer);
    _testSearchTimer = setTimeout(() => {
      loadTestLibrary(true).catch((err) => toast(err.message, "error"));
    }, 300);
  });

  function setTestSource(source) {
    testState.source = source;
    $("srcVideo").classList.toggle("active", source === "video");
    $("srcCamera").classList.toggle("active", source === "camera");
    $("panelTestVideo").classList.toggle("hidden", source !== "video");
    $("panelTestCamera").classList.toggle("hidden", source !== "camera");
    if (source === "video") stopCamera();
  }

  $("srcVideo").onclick = () => setTestSource("video");
  $("srcCamera").onclick = () => setTestSource("camera");

  function setMode(mode) {
    const isLabel = mode === "label";
    $("modeLabel").classList.toggle("hidden", !isLabel);
    $("modeTest").classList.toggle("hidden", isLabel);
    $("tabLabel").classList.toggle("active", isLabel);
    $("tabTest").classList.toggle("active", !isLabel);
    if (isLabel) stopCamera();
    if (!isLabel) {
      loadModels();
      loadTestLibrary(true).catch(() => {});
    }
    relayoutPlayers();
  }

  $("tabLabel").onclick = () => setMode("label");
  $("tabTest").onclick = () => setMode("test");

  async function loadModels() {
    try {
      const data = await api("/api/models");
      const models = data.models || [];
      const pthModels = models.filter((m) => (m.format || "pth") !== "onnx");
      fillModelSelect($("modelSelect"), models, "No checkpoints found — train first");
      fillModelSelect($("onnxSourceSelect"), pthModels, "No .pth checkpoints — train first");
      if (!models.length) {
        $("modelHint").textContent = "No .pth/.onnx files in work_dirs/slowfast_multilabel/";
        return;
      }
      const selected = $("modelSelect")?.value;
      const picked = models.find((m) => m.name === selected) || models.find((m) => m.recommended) || models[0];
      $("modelHint").textContent = picked
        ? `Selected: ${picked.name} (${picked.format === "onnx" ? "ONNX" : "PyTorch"})`
        : "";
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function fillModelSelect(sel, models, emptyText) {
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = "";
    if (!models.length) {
      sel.innerHTML = `<option value="">${emptyText}</option>`;
      return;
    }
    models.forEach((m) => {
      const fmt = m.format === "onnx" ? "ONNX" : "PyTorch";
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = `${m.name} · ${fmt} (${m.size_mb} MB)`;
      sel.appendChild(opt);
    });
    const keep = models.find((m) => m.name === prev);
    const best =
      models.find((m) => m.recommended) ||
      models.find((m) => (m.format || "pth") !== "onnx") ||
      models[0];
    sel.value = keep ? keep.name : best.name;
  }

  $("btnRefreshModels").onclick = () => loadModels();
  $("modelSelect").onchange = () => {
    const name = $("modelSelect").value;
    const isOnnx = name.toLowerCase().endsWith(".onnx");
    $("modelHint").textContent = name ? `Selected: ${name} (${isOnnx ? "ONNX" : "PyTorch"})` : "";
  };

  let onnxExportJobId = null;
  let onnxPollTimer = null;

  async function pollOnnxExport() {
    if (!onnxExportJobId) return;
    try {
      const data = await api(`/api/models/export-onnx/status?job_id=${encodeURIComponent(onnxExportJobId)}`);
      const job = data.job;
      if (!job) return;
      const hint = $("onnxExportHint");
      const lines = job.log || [];
      if (hint) {
        hint.textContent = lines.length ? lines.slice(-2).join(" · ") : `(${job.status})`;
      }
      if (["completed", "failed"].includes(job.status)) {
        clearInterval(onnxPollTimer);
        onnxPollTimer = null;
        if (job.status === "completed") {
          toast(`ONNX saved: ${PathName(job.output || job.checkpoint)}`, "ok");
          loadModels();
        } else {
          toast(job.error || "ONNX export failed", "error");
        }
      }
    } catch (err) {
      console.error(err);
    }
  }

  function PathName(p) {
    if (!p) return "model.onnx";
    const parts = String(p).split(/[/\\]/);
    return parts[parts.length - 1];
  }

  $("btnExportOnnx")?.addEventListener("click", async () => {
    const checkpoint = $("onnxSourceSelect")?.value;
    if (!checkpoint) {
      toast("Train a .pth checkpoint first", "error");
      return;
    }
    try {
      const data = await api("/api/models/export-onnx", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checkpoint }),
      });
      onnxExportJobId = data.job.job_id;
      toast("ONNX export started — this can take a few minutes");
      if (onnxPollTimer) clearInterval(onnxPollTimer);
      onnxPollTimer = setInterval(pollOnnxExport, 2000);
      pollOnnxExport();
    } catch (err) {
      toast(err.message, "error");
    }
  });

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
      const done = job.status === "completed" || Boolean(job.finished_at && job.output_video);
      const pill = $("testPill");
      pill.textContent = done ? "completed" : job.status;
      pill.className = `status-pill ${done ? "completed" : job.status}`;
      const lines = job.log || [];
      $("testLog").textContent = lines.length ? lines.join("\n") : `(${job.status})`;

      if (done) {
        clearInterval(testState.pollTimer);
        testState.pollTimer = null;
        showTestResult(job);
        loadTestLibrary(true).catch(() => {});
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
    stopCamera();
    $("testEmpty").classList.add("hidden");
    $("testActive").classList.remove("hidden");
    $("liveStack").classList.add("hidden");
    $("resultStack").classList.remove("hidden");
    $("btnDownloadResult").classList.remove("hidden");
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
    relayoutPlayers();
  }

  $("btnRunTest").onclick = async () => {
    stopCamera();
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

  function showLiveStage() {
    $("testEmpty").classList.add("hidden");
    $("testActive").classList.remove("hidden");
    $("resultStack").classList.add("hidden");
    $("liveStack").classList.remove("hidden");
    $("btnDownloadResult").classList.add("hidden");
    relayoutPlayers();
  }

  function drawLiveOverlay() {
    const video = $("liveVideo");
    const canvas = $("liveOverlay");
    const wrap = canvas.parentElement.getBoundingClientRect();
    const ctxLive = canvas.getContext("2d");
    canvas.width = wrap.width * devicePixelRatio;
    canvas.height = wrap.height * devicePixelRatio;
    canvas.style.width = `${wrap.width}px`;
    canvas.style.height = `${wrap.height}px`;
    ctxLive.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    ctxLive.clearRect(0, 0, wrap.width, wrap.height);
    const vw = video.videoWidth || 1;
    const vh = video.videoHeight || 1;
    const scale = Math.min(wrap.width / vw, wrap.height / vh);
    const dw = vw * scale;
    const dh = vh * scale;
    const ox = (wrap.width - dw) / 2;
    const oy = (wrap.height - dh) / 2;
    (testState.lastPersons || []).forEach((p, i) => {
      const [x1, y1, x2, y2] = p.bbox || [0, 0, 0, 0];
      const x = ox + (x1 / vw) * dw;
      const y = oy + (y1 / vh) * dh;
      const w = ((x2 - x1) / vw) * dw;
      const h = ((y2 - y1) / vh) * dh;
      ctxLive.strokeStyle = "#0f766e";
      ctxLive.lineWidth = 2;
      ctxLive.strokeRect(x, y, w, h);
      const text = `P${p.id ?? i}: ${p.label || "unknown"} ${((p.score || 0) * 100).toFixed(0)}%`;
      ctxLive.font = "600 13px IBM Plex Sans, sans-serif";
      const tw = ctxLive.measureText(text).width;
      ctxLive.fillStyle = "#0f766e";
      ctxLive.fillRect(x, Math.max(0, y - 20), tw + 10, 20);
      ctxLive.fillStyle = "#fff";
      ctxLive.fillText(text, x + 5, Math.max(14, y - 5));
    });
  }

  function captureLiveJpeg() {
    const video = $("liveVideo");
    if (!video.videoWidth) return Promise.resolve(null);
    const maxW = 640;
    const scale = Math.min(1, maxW / video.videoWidth);
    const w = Math.round(video.videoWidth * scale);
    const h = Math.round(video.videoHeight * scale);
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    c.getContext("2d").drawImage(video, 0, 0, w, h);
    return new Promise((resolve) => c.toBlob((b) => resolve(b), "image/jpeg", 0.7));
  }

  async function liveLoop() {
    const CLIP = 16;
    while (testState.cameraOn) {
      const checkpoint = $("modelSelect").value;
      if (!checkpoint) {
        toast("Select a model checkpoint", "error");
        stopCamera();
        break;
      }
      const blobs = [];
      for (let i = 0; i < CLIP && testState.cameraOn; i++) {
        const blob = await captureLiveJpeg();
        if (blob) blobs.push(blob);
        await new Promise((r) => setTimeout(r, 80));
      }
      if (!testState.cameraOn || blobs.length < 5) continue;
      const fd = new FormData();
      fd.append("checkpoint", checkpoint);
      blobs.forEach((b, i) => fd.append("frames", b, `f${i}.jpg`));
      $("testPill").textContent = "live";
      $("testPill").className = "status-pill running";
      $("testLog").textContent = `Sending ${blobs.length} frames…`;
      try {
        const data = await api("/api/test/live", { method: "POST", body: fd });
        testState.lastPersons = data.persons || [];
        drawLiveOverlay();
        const body = $("predBody");
        body.innerHTML = "";
        if (!testState.lastPersons.length) {
          body.innerHTML = '<tr><td colspan="5">No persons detected</td></tr>';
        } else {
          testState.lastPersons.forEach((p) => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
              <td>P${p.id}</td>
              <td>${p.posture || "—"}</td>
              <td>${p.activity || "—"}</td>
              <td>${((p.score || 0) * 100).toFixed(1)}%</td>
              <td>${p.frames}</td>`;
            body.appendChild(tr);
          });
        }
        $("testLog").textContent = `${testState.lastPersons.length} person(s) · last clip ${blobs.length} frames`;
      } catch (err) {
        $("testLog").textContent = err.message || "Live inference failed";
        $("testPill").textContent = "failed";
        $("testPill").className = "status-pill failed";
      }
    }
  }

  async function startCamera() {
    const checkpoint = $("modelSelect").value;
    if (!checkpoint) {
      toast("Select a model checkpoint", "error");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      testState.stream = stream;
      testState.cameraOn = true;
      testState.lastPersons = [];
      showLiveStage();
      const live = $("liveVideo");
      live.srcObject = stream;
      await live.play().catch(() => {});
      $("btnStartCam").classList.add("hidden");
      $("btnStopCam").classList.remove("hidden");
      $("testPill").textContent = "live";
      $("testPill").className = "status-pill running";
      $("testLog").textContent = "Camera on — loading models on first clip…";
      toast("Camera started", "ok");
      liveLoop();
    } catch (err) {
      toast(err.message || "Could not open camera", "error");
    }
  }

  function stopCamera() {
    testState.cameraOn = false;
    if (testState.stream) {
      testState.stream.getTracks().forEach((t) => t.stop());
      testState.stream = null;
    }
    const live = $("liveVideo");
    if (live) live.srcObject = null;
    $("btnStartCam")?.classList.remove("hidden");
    $("btnStopCam")?.classList.add("hidden");
    if ($("testPill") && $("testPill").textContent === "live") {
      $("testPill").textContent = "idle";
      $("testPill").className = "status-pill";
    }
  }

  $("btnStartCam").onclick = () => startCamera();
  $("btnStopCam").onclick = () => {
    stopCamera();
    toast("Camera stopped", "ok");
  };

  async function init() {
    try {
      await loadLabels();
      await loadVideos();
      await loadModels();
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
