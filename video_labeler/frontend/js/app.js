(() => {
  const $ = (id) => document.getElementById(id);

  const state = {
    labels: [],
    postures: [],
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
    editingSegIdx: null,
    cropMode: false,
    cropDraft: null, // {x1,y1,x2,y2} in video pixel space
    drawing: false,
    drawOrigin: null,
    trainJobId: null,
    pollTimer: null,
    editingClasses: false,
    clientId: localStorage.getItem("actionmark_client_id") || "",
    collabName: localStorage.getItem("actionmark_name") || "",
    libraryRevision: 0,
    locks: {}, // video_id -> { name, client_id }
    selectedVideos: new Set(),
    collabPollTimer: null,
    lockHeartbeatTimer: null,
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

  function isActivity(name) {
    return (state.activities || []).includes(name);
  }

  function displayLabel(seg) {
    const activity = (seg.activity || "").trim();
    return activity || seg.label || "—";
  }

  function normalizeSeg(seg) {
    // Activity-only: strip posture and ensure `label` matches `activity`.
    const next = { ...seg };
    delete next.posture;

    // Prefer explicit `activity`, else try to recover from legacy `label` like "standing + walking".
    let activity = (next.activity || "").trim();
    if (!activity) {
      const legacy = next.label || "";
      const parts = legacy.split(/[+,]/).map((s) => s.trim()).filter(Boolean);
      activity = parts.find((p) => isActivity(p)) || "";
    }

    next.activity = activity;
    next.label = activity || "";
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
    const groups = [{ title: "Activity", items: state.activities || [] }];
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

    const activitySel = $("activitySelect");
    if (activitySel) fillSelect(activitySel, state.activities || [], [{ value: "", label: "none" }]);
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
    state.postures = data.postures || [];
    state.activities = data.activities || state.labels;
    state.counts = data.counts || {};
    state.totalCount = data.total || 0;
    renderLabels();
  }

  async function removeClass(name) {
    const count = state.counts[name] || 0;
    const kind = "activity";
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

  function getActiveVideoListEl() {
    const panelHas = $("libraryPanelHas");
    if (panelHas && !panelHas.hidden) return $("videoListHas");
    return $("videoListNeed");
  }

  function captureLibraryScroll() {
    const el = getActiveVideoListEl();
    return el ? el.scrollTop : 0;
  }

  function restoreLibraryScroll(top) {
    const el = getActiveVideoListEl();
    if (!el || top == null) return;
    const apply = () => {
      el.scrollTop = top;
    };
    requestAnimationFrame(() => {
      apply();
      requestAnimationFrame(apply);
    });
  }

  let _libraryScrollRestore = null;

  function appendVideoItem(list, v, index) {
    const li = document.createElement("li");
    if (v.id === state.videoId) li.classList.add("active");
    if (state.selectedVideos.has(v.id)) li.classList.add("selected");
    const lock = state.locks[v.id];
    const lockedByOther = lock && lock.client_id !== state.clientId;
    if (lockedByOther) li.classList.add("locked-by-other");
    let lockHtml = "";
    if (lock) {
      const mine = lock.client_id === state.clientId;
      lockHtml = `<div class="lock-badge ${mine ? "mine" : ""}">${mine ? "You are editing" : `Locked by ${lock.name || "someone"}`}</div>`;
    }
    let annotatorLine = "";
    if (v.segments > 0 && v.last_annotator) {
      annotatorLine = `<div class="meta">Saved by ${v.last_annotator}${v.updated_at ? ` · ${formatAnnotateTime(v.updated_at)}` : ""}</div>`;
    }
    let processingHtml = "";
    if (v.processing_status === "transcoding") {
      processingHtml = `<div class="lock-badge processing">Converting to H.264…</div>`;
    } else if (v.processing_status === "failed") {
      processingHtml = `<div class="lock-badge processing-failed">Convert failed</div>`;
    }
    li.innerHTML = `
      <div class="video-row">
        <label class="video-check" title="Select for bulk delete">
          <input type="checkbox" data-video-id="${v.id}" ${state.selectedVideos.has(v.id) ? "checked" : ""} />
        </label>
        <span class="video-num">${index}</span>
        <div class="video-info">
          <div class="name" title="${v.filename}">${v.filename}</div>
          <div class="meta">${Math.round(v.duration || 0)}s · ${v.segments || 0} segments · ${v.total_frames || 0} frames</div>
          ${annotatorLine}
          ${processingHtml}
          ${lockHtml}
        </div>
        <button type="button" class="btn-delete-video" title="Delete video" aria-label="Delete ${v.filename}">×</button>
      </div>`;
    const check = li.querySelector('input[type="checkbox"]');
    check.onclick = (e) => e.stopPropagation();
    check.onchange = () => {
      if (check.checked) state.selectedVideos.add(v.id);
      else state.selectedVideos.delete(v.id);
      li.classList.toggle("selected", check.checked);
      updateBulkBars();
    };
    li.querySelector(".video-info").onclick = () => selectVideo(v.id);
    li.querySelector(".btn-delete-video").onclick = (e) => {
      e.stopPropagation();
      deleteVideo(v.id, v.filename);
    };
    list.appendChild(li);
  }

  function videosInTab(tab) {
    if (tab === "has") return state.videos.filter((v) => v.segments > 0);
    return state.videos.filter((v) => !(v.segments > 0));
  }

  function pruneSelectedVideos() {
    const alive = new Set((state.videos || []).map((v) => v.id));
    for (const id of [...state.selectedVideos]) {
      if (!alive.has(id)) state.selectedVideos.delete(id);
    }
  }

  function updateBulkBars() {
    pruneSelectedVideos();
    for (const tab of ["need", "has"]) {
      const videos = videosInTab(tab);
      const selected = videos.filter((v) => state.selectedVideos.has(v.id));
      const countEl = $(tab === "need" ? "bulkCountNeed" : "bulkCountHas");
      const clearBtn = $(tab === "need" ? "btnClearSelNeed" : "btnClearSelHas");
      const delBtn = $(tab === "need" ? "btnDeleteSelNeed" : "btnDeleteSelHas");
      const selectBtn = $(tab === "need" ? "btnSelectAllNeed" : "btnSelectAllHas");
      if (countEl) {
        countEl.textContent = selected.length
          ? `${selected.length} selected`
          : videos.length
            ? ""
            : "";
      }
      if (clearBtn) clearBtn.disabled = selected.length === 0;
      if (delBtn) {
        delBtn.disabled = selected.length === 0;
        delBtn.textContent = selected.length ? `Delete (${selected.length})` : "Delete";
      }
      if (selectBtn) {
        selectBtn.disabled = videos.length === 0;
        const allSelected = videos.length > 0 && videos.every((v) => state.selectedVideos.has(v.id));
        selectBtn.textContent = allSelected ? "Deselect all" : "Select all";
      }
    }
  }

  function selectAllInTab(tab) {
    const videos = videosInTab(tab);
    const allSelected = videos.length > 0 && videos.every((v) => state.selectedVideos.has(v.id));
    if (allSelected) {
      videos.forEach((v) => state.selectedVideos.delete(v.id));
    } else {
      videos.forEach((v) => state.selectedVideos.add(v.id));
    }
    renderVideoList({ preserveScroll: true });
  }

  function clearSelectionInTab(tab) {
    videosInTab(tab).forEach((v) => state.selectedVideos.delete(v.id));
    renderVideoList({ preserveScroll: true });
  }

  function renderVideoList({ preserveScroll = false } = {}) {
    const scrollTop = preserveScroll ? captureLibraryScroll() : null;
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
      renderActiveEditors();
      updateBulkBars();
      if (preserveScroll && scrollTop != null) restoreLibraryScroll(scrollTop);
      else if (_libraryScrollRestore != null) {
        restoreLibraryScroll(_libraryScrollRestore);
        _libraryScrollRestore = null;
      }
      return;
    }
    if (!needVideos.length) {
      needList.innerHTML = '<li class="meta">None on this page</li>';
    } else {
      needVideos.forEach((v, i) => appendVideoItem(needList, v, i + 1));
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
      hasVideos.forEach((v, i) => appendVideoItem(hasList, v, i + 1));
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
    renderActiveEditors();
    updateBulkBars();
    if (preserveScroll && scrollTop != null) {
      restoreLibraryScroll(scrollTop);
    } else if (_libraryScrollRestore != null) {
      restoreLibraryScroll(_libraryScrollRestore);
      _libraryScrollRestore = null;
    }
  }

  async function refreshVideoPagingCounts() {
    const q = (state.videoQuery || "").trim();
    const params = new URLSearchParams({ per_page: PER_PAGE, page: "1" });
    if (q) params.set("q", q);
    params.set("labeled", "false");
    const needData = await api(`/api/videos?${params}`);
    params.set("labeled", "true");
    const hasData = await api(`/api/videos?${params}`);
    state.videoPaging = {
      total_all: needData.total_all || 0,
      total_labeled: needData.total_labeled || 0,
      needTotal: needData.total || 0,
      hasTotal: hasData.total || 0,
      needPages: needData.pages || 1,
      hasPages: hasData.pages || 1,
    };
    if (state.videoPage.need > state.videoPaging.needPages) {
      state.videoPage.need = Math.max(1, state.videoPaging.needPages);
    }
    if (state.videoPage.has > state.videoPaging.hasPages) {
      state.videoPage.has = Math.max(1, state.videoPaging.hasPages);
    }
  }

  async function reloadVideosKeepingScroll() {
    _libraryScrollRestore = captureLibraryScroll();
    const needPages = state.videoPage.need;
    const hasPages = state.videoPage.has;
    const q = (state.videoQuery || "").trim();
    const base = new URLSearchParams({ per_page: PER_PAGE });
    if (q) base.set("q", q);

    const needVideos = [];
    let needMeta = null;
    for (let p = 1; p <= needPages; p++) {
      const params = new URLSearchParams(base);
      params.set("page", String(p));
      params.set("labeled", "false");
      needMeta = await api(`/api/videos?${params}`);
      needVideos.push(...(needMeta.videos || []));
    }

    const hasVideos = [];
    let hasMeta = null;
    for (let p = 1; p <= hasPages; p++) {
      const params = new URLSearchParams(base);
      params.set("page", String(p));
      params.set("labeled", "true");
      hasMeta = await api(`/api/videos?${params}`);
      hasVideos.push(...(hasMeta.videos || []));
    }

    state.videos = [...needVideos, ...hasVideos];
    state.videoPaging = {
      total_all: needMeta?.total_all || 0,
      total_labeled: needMeta?.total_labeled || 0,
      needTotal: needMeta?.total || 0,
      hasTotal: hasMeta?.total || 0,
      needPages: needMeta?.pages || 1,
      hasPages: hasMeta?.pages || 1,
    };
    renderVideoList();
    if (state.videoId && !state.videos.some((v) => v.id === state.videoId)) {
      try {
        await api(`/api/videos/${encodeURIComponent(state.videoId)}/meta`);
      } catch {
        await clearVideoSelection({ silent: true });
        toast("This video was removed from the library", "ok");
      }
    }
  }

  function isVideoSelectable(v) {
    if (v.processing_status === "transcoding") return false;
    const lock = state.locks?.[v.id];
    if (!lock) return true;
    return lock.client_id === state.clientId;
  }

  function pickNextVideoAfterDelete(deletedIds) {
    const deleted = new Set(Array.isArray(deletedIds) ? deletedIds : [deletedIds]);
    const order = state.videos || [];
    const available = order.filter((v) => !deleted.has(v.id) && isVideoSelectable(v));
    if (!available.length) return null;
    const curIdx = order.findIndex((v) => deleted.has(v.id) && v.id === state.videoId);
    const next =
      curIdx >= 0
        ? available.find((v) => order.findIndex((x) => x.id === v.id) > curIdx)
        : null;
    return next || available[0];
  }

  async function deleteVideo(id, filename) {
    const ok = window.confirm(`Delete "${filename}" and its annotations?`);
    if (!ok) return;
    await deleteVideosBulk([id]);
  }

  async function deleteSelectedInTab(tab) {
    const ids = videosInTab(tab)
      .filter((v) => state.selectedVideos.has(v.id))
      .map((v) => v.id);
    if (!ids.length) return;
    const ok = window.confirm(
      `Delete ${ids.length} video${ids.length === 1 ? "" : "s"} and their annotations? This cannot be undone.`
    );
    if (!ok) return;
    await deleteVideosBulk(ids);
  }

  async function deleteAllSelected() {
    const ids = [...state.selectedVideos];
    if (!ids.length) {
      toast("No videos selected — press A on a video first", "error");
      return;
    }
    const ok = window.confirm(
      `Delete ${ids.length} video${ids.length === 1 ? "" : "s"} and their annotations? This cannot be undone.`
    );
    if (!ok) return;
    await deleteVideosBulk(ids);
  }

  async function deleteVideosBulk(ids) {
    if (!ids.length) return;
    try {
      _libraryScrollRestore = captureLibraryScroll();
      const wasOpen = ids.includes(state.videoId);
      const nextVideo = wasOpen ? pickNextVideoAfterDelete(ids) : null;
      toast(`Deleting ${ids.length} video${ids.length === 1 ? "" : "s"}…`, "ok");
      const data = await api("/api/videos/bulk-delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids }),
      });
      const deleted = new Set(data.deleted || ids);
      state.videos = state.videos.filter((v) => !deleted.has(v.id));
      deleted.forEach((id) => state.selectedVideos.delete(id));
      await refreshVideoPagingCounts();
      renderVideoList();
      await loadLabelCounts();
      const count = data.count ?? deleted.size;
      if (wasOpen && nextVideo) {
        await selectVideo(nextVideo.id);
        toast(`Deleted ${count} · opened ${nextVideo.filename || nextVideo.id}`, "ok");
      } else if (wasOpen) {
        await clearVideoSelection({ silent: true });
        toast(`Deleted ${count} video${count === 1 ? "" : "s"}`, "ok");
      } else {
        toast(`Deleted ${count} video${count === 1 ? "" : "s"}`, "ok");
      }
    } catch (err) {
      _libraryScrollRestore = null;
      toast(err.message || "Delete failed", "error");
    }
  }

  function updateSegmentEditorUI() {
    const editing = state.editingSegIdx != null;
    $("btnSaveSeg").textContent = editing ? "Update segment" : "Save segment";
    $("btnCancelEdit").hidden = !editing;
    $("pendingInfo").classList.toggle("editing-segment", editing);
  }

  function clearSegmentEdit() {
    if (state.editingSegIdx == null) return;
    state.editingSegIdx = null;
    state.pendingStart = null;
    state.pendingEnd = null;
    state.cropDraft = null;
    updateCropWarn();
    updatePending();
    updateSegmentEditorUI();
    renderSegments();
  }

  function loadSegmentForEdit(idx) {
    const seg = state.segments[idx];
    if (!seg) return;
    state.editingSegIdx = idx;
    state.pendingStart = seg.start_frame;
    state.pendingEnd = seg.end_frame;
    $("activitySelect").value = seg.activity || "";
    if (seg.bbox?.length === 4) {
      const [x1, y1, x2, y2] = seg.bbox.map(Number);
      state.cropDraft = { x1, y1, x2, y2 };
    } else {
      state.cropDraft = null;
    }
    updateCropWarn();
    updatePending();
    updateSegmentEditorUI();
    seekToFrame(seg.start_frame);
    renderSegments();
    toast(`Editing ${displayLabel(seg)} (${seg.start_frame}–${seg.end_frame})`);
  }

  function renderSegments() {
    const body = $("segBody");
    body.innerHTML = "";
    state.segments.forEach((seg, idx) => {
      const tr = document.createElement("tr");
      if (idx === state.editingSegIdx) tr.classList.add("editing");
      tr.innerHTML = `
        <td>${seg.activity || "—"}</td>
        <td>${seg.start_frame}</td>
        <td>${seg.end_frame}</td>
        <td class="seg-row-actions">
          <button type="button" class="seg-edit" data-idx="${idx}">Edit</button>
          <button type="button" class="seg-delete" data-idx="${idx}">Delete</button>
        </td>`;
      tr.querySelector(".seg-edit").onclick = () => loadSegmentForEdit(idx);
      tr.querySelector(".seg-delete").onclick = () => {
        if (state.editingSegIdx === idx) state.editingSegIdx = null;
        else if (state.editingSegIdx != null && idx < state.editingSegIdx) {
          state.editingSegIdx -= 1;
        }
        state.segments.splice(idx, 1);
        updateSegmentEditorUI();
        renderSegments();
      };
      body.appendChild(tr);
    });
    drawTimeline();
    drawOverlay();
  }

  function updatePending() {
    const base = `Start: ${state.pendingStart ?? "—"} · End: ${state.pendingEnd ?? "—"}`;
    $("pendingInfo").textContent =
      state.editingSegIdx != null ? `${base} · editing saved segment` : base;
  }

  function currentFrame() {
    const fps = state.meta?.fps || 30;
    return Math.max(0, Math.floor(videoEl.currentTime * fps + 1e-6));
  }

  function seekToFrame(frame) {
    if (!state.meta && !videoEl.duration) return;
    const fps = state.meta?.fps || 30;
    const total = state.meta?.total_frames || Math.max(1, Math.floor((videoEl.duration || 0) * fps));
    const f = Math.min(Math.max(0, frame), Math.max(0, total - 1));
    const t = f / fps;
    // Pause so browsers apply the seek reliably
    if (!videoEl.paused) videoEl.pause();
    try {
      videoEl.currentTime = t;
    } catch (_) {
      /* ignore seek errors while loading */
    }
    syncSeek();
  }

  function seekBySeconds(delta) {
    if (!videoEl || (!Number.isFinite(videoEl.duration) && !state.meta)) return;
    const fps = state.meta?.fps || 30;
    const duration =
      Number.isFinite(videoEl.duration) && videoEl.duration > 0
        ? videoEl.duration
        : (state.meta?.total_frames || 1) / fps;
    const next = Math.min(Math.max(0, (videoEl.currentTime || 0) + delta), Math.max(0, duration - 0.001));
    if (!videoEl.paused) videoEl.pause();
    try {
      videoEl.currentTime = next;
    } catch (_) {
      /* ignore */
    }
    syncSeek();
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
    state.segments.forEach((seg, idx) => {
      const span = document.createElement("span");
      const left = (seg.start_frame / total) * 100;
      const width = Math.max(0.4, ((seg.end_frame - seg.start_frame + 1) / total) * 100);
      span.style.left = `${left}%`;
      span.style.width = `${width}%`;
      span.title = `${displayLabel(seg)} ${seg.start_frame}-${seg.end_frame}`;
      if (idx === state.editingSegIdx) span.classList.add("editing");
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

  function collabDisplayName() {
    const input = $("collabName");
    const raw = (input?.value || state.collabName || "").trim();
    return raw || "Annotator";
  }

  function persistCollabName() {
    const name = ($("collabName")?.value || "").trim();
    state.collabName = name;
    localStorage.setItem("actionmark_name", name);
  }

  function formatAnnotateTime(iso) {
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

  function renderAnnotateMeta(meta) {
    const el = $("annotateMeta");
    const logEl = $("annotateLog");
    const summary = $("annotateLogSummary");
    if (!el) return;
    if (!meta) {
      el.textContent = "Select a video to annotate.";
      if (logEl) logEl.innerHTML = "";
      if (summary) summary.textContent = "Save history";
      return;
    }
    const who = meta.last_annotator || "";
    const when = meta.updated_at || "";
    if (who && when) {
      el.textContent = `Last saved by ${who} · ${formatAnnotateTime(when)}`;
    } else if (who) {
      el.textContent = `Last saved by ${who}`;
    } else if ((meta.segments || []).length) {
      el.textContent = "Saved (annotator not recorded yet — save again with your name set)";
    } else {
      el.textContent = "No segments saved yet.";
    }
    if (logEl) {
      const log = [...(meta.annotation_log || [])].reverse().slice(0, 8);
      if (summary) {
        summary.textContent = log.length
          ? `Save history (${log.length})`
          : "Save history";
      }
      if (!log.length) {
        logEl.innerHTML = "<li>No save history yet.</li>";
      } else {
        logEl.innerHTML = log
          .map(
            (entry) =>
              `<li><span class="log-who">${entry.annotator || "Annotator"}</span> · ${entry.segments ?? 0} seg · ${formatAnnotateTime(entry.at)}</li>`
          )
          .join("");
      }
    }
  }

  function updateLockPill() {
    const pill = $("lockPill");
    if (!pill) return;
    if (!state.videoId) {
      pill.hidden = true;
      return;
    }
    const lock = state.locks[state.videoId];
    if (!lock) {
      pill.hidden = false;
      pill.className = "lock-pill";
      pill.textContent = "Editing";
      return;
    }
    const mine = lock.client_id === state.clientId;
    pill.hidden = false;
    pill.className = mine ? "lock-pill" : "lock-pill other";
    pill.textContent = mine ? "You hold the lock" : `Locked by ${lock.name || "someone"}`;
  }

  async function ensureCollabClient() {
    if (state.clientId) return state.clientId;
    const data = await api("/api/collab/hello", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: collabDisplayName() }),
    });
    state.clientId = data.client_id;
    state.libraryRevision = data.revision || 0;
    localStorage.setItem("actionmark_client_id", state.clientId);
    return state.clientId;
  }

  async function releaseCurrentLock() {
    if (!state.videoId || !state.clientId) return;
    const vid = state.videoId;
    try {
      await api(
        `/api/collab/lock/${encodeURIComponent(vid)}?client_id=${encodeURIComponent(state.clientId)}`,
        { method: "DELETE" }
      );
    } catch (_) {
      /* ignore */
    }
    if (state.locks[vid]?.client_id === state.clientId) {
      delete state.locks[vid];
    }
  }

  async function acquireVideoLock(videoId) {
    await ensureCollabClient();
    const data = await api(`/api/collab/lock/${encodeURIComponent(videoId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_id: state.clientId, name: collabDisplayName() }),
    });
    if (data.lock) {
      state.locks[videoId] = {
        client_id: data.lock.client_id,
        name: data.lock.name,
        filename: data.lock.filename || "",
      };
    }
    renderActiveEditors();
    return data;
  }

  function startLockHeartbeat() {
    if (state.lockHeartbeatTimer) clearInterval(state.lockHeartbeatTimer);
    state.lockHeartbeatTimer = setInterval(async () => {
      if (!state.clientId || !state.videoId) return;
      try {
        const data = await api("/api/collab/heartbeat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            client_id: state.clientId,
            video_id: state.videoId,
            name: collabDisplayName(),
          }),
        });
        applyLocks(data.locks || []);
        if (typeof data.revision === "number" && data.revision > state.libraryRevision) {
          state.libraryRevision = data.revision;
          await reloadVideosKeepingScroll();
        }
      } catch (_) {
        /* ignore transient errors */
      }
    }, 15000);
  }

  function applyLocks(locks) {
    const map = {};
    (locks || []).forEach((l) => {
      map[l.video_id] = {
        client_id: l.client_id,
        name: l.name,
        filename: l.filename || "",
      };
    });
    state.locks = map;
    renderVideoList({ preserveScroll: true });
    renderActiveEditors();
    updateLockPill();
  }

  function resolveLockFilename(videoId, lock) {
    if (lock?.filename) return lock.filename;
    const hit = (state.videos || []).find((v) => v.id === videoId);
    return hit?.filename || videoId;
  }

  function renderActiveEditors() {
    const wrap = $("activeEditors");
    const list = $("activeEditorsList");
    if (!wrap || !list) return;
    const entries = Object.entries(state.locks || {});
    if (!entries.length) {
      wrap.hidden = true;
      list.innerHTML = "";
      return;
    }
    wrap.hidden = false;
    list.innerHTML = "";
    entries
      .sort((a, b) => {
        const aMine = a[1].client_id === state.clientId ? 0 : 1;
        const bMine = b[1].client_id === state.clientId ? 0 : 1;
        if (aMine !== bMine) return aMine - bMine;
        return resolveLockFilename(a[0], a[1]).localeCompare(resolveLockFilename(b[0], b[1]));
      })
      .forEach(([videoId, lock]) => {
        const mine = lock.client_id === state.clientId;
        const li = document.createElement("li");
        if (mine) li.classList.add("mine");
        const file = resolveLockFilename(videoId, lock);
        li.innerHTML = `
          <span class="ae-who">${mine ? "You" : lock.name || "Someone"}</span>
          <span class="ae-file" title="${file}">${file}</span>`;
        li.onclick = () => {
          // Switch to the right tab and try to open / highlight
          const v = (state.videos || []).find((x) => x.id === videoId);
          if (v) {
            setLibraryTab(v.segments > 0 ? "has" : "need");
          }
          if (mine || !state.locks[videoId] || state.locks[videoId].client_id === state.clientId) {
            selectVideo(videoId).catch((e) => toast(e.message, "error"));
          } else {
            toast(`Locked by ${lock.name || "someone"} — ${file}`, "ok");
            // Scroll list item into view if present
            requestAnimationFrame(() => {
              const rows = document.querySelectorAll(".video-list li.active, .video-list li.locked-by-other");
              // Find by filename text
              document.querySelectorAll(".video-list .name").forEach((el) => {
                if (el.getAttribute("title") === file || el.textContent === file) {
                  el.closest("li")?.scrollIntoView({ block: "nearest", behavior: "smooth" });
                }
              });
            });
          }
        };
        list.appendChild(li);
      });
  }

  async function pollCollabStatus() {
    try {
      await ensureCollabClient();
      const data = await api(`/api/collab/status?since=${state.libraryRevision || 0}`);
      applyLocks(data.locks || []);
      if (data.changed && data.revision > state.libraryRevision) {
        state.libraryRevision = data.revision;
        // Refresh library only — keep scroll, loaded pages, and open video
        await reloadVideosKeepingScroll();
      } else if (typeof data.revision === "number") {
        state.libraryRevision = data.revision;
      }
    } catch (_) {
      /* ignore */
    }
  }

  function startCollabPolling() {
    if (state.collabPollTimer) clearInterval(state.collabPollTimer);
    state.collabPollTimer = setInterval(pollCollabStatus, 4000);
    pollCollabStatus();
  }

  async function clearVideoSelection({ silent = false } = {}) {
    const hadVideo = !!state.videoId;
    if (state.lockHeartbeatTimer) {
      clearInterval(state.lockHeartbeatTimer);
      state.lockHeartbeatTimer = null;
    }
    await releaseCurrentLock();
    state.videoId = null;
    state.meta = null;
    state.segments = [];
    state.pendingStart = null;
    state.pendingEnd = null;
    state.editingSegIdx = null;
    state.cropDraft = null;
    state.drawing = false;
    updateCropWarn();
    updatePending();
    updateSegmentEditorUI();
    videoEl.pause();
    videoEl.removeAttribute("src");
    videoEl.load();
    $("stageActive").classList.add("hidden");
    $("stageEmpty").classList.remove("hidden");
    $("segmentsPanel").classList.add("hidden");
    $("modeLabel")?.classList.remove("segments-open");
    updateLabelLayout();
    updateLockPill();
    renderAnnotateMeta(null);
    renderVideoList({ preserveScroll: true });
    renderActiveEditors();
    setMobilePanel("modeLabel", "nav");
    if (hadVideo && !silent) toast("Video deselected");
  }

  async function selectVideo(id) {
    if (state.videoId === id) {
      await clearVideoSelection();
      return;
    }

    const listed = (state.videos || []).find((v) => v.id === id);
    if (listed?.processing_status === "transcoding") {
      toast("This video is still converting — try again in a moment", "error");
      return;
    }

    const previousId = state.videoId;
    try {
      await ensureCollabClient();
      await acquireVideoLock(id);
      if (previousId && previousId !== id) {
        try {
          await api(
            `/api/collab/lock/${encodeURIComponent(previousId)}?client_id=${encodeURIComponent(state.clientId)}`,
            { method: "DELETE" }
          );
        } catch (_) {
          /* ignore */
        }
        if (state.locks[previousId]?.client_id === state.clientId) {
          delete state.locks[previousId];
        }
      }
    } catch (err) {
      toast(err.message || "Video is locked by someone else", "error");
      renderVideoList({ preserveScroll: true });
      return;
    }

    state.videoId = id;
    state.pendingStart = null;
    state.pendingEnd = null;
    state.editingSegIdx = null;
    state.cropDraft = null;
    updateCropWarn();
    updatePending();
    updateSegmentEditorUI();
    $("stageEmpty").classList.add("hidden");
    $("stageActive").classList.remove("hidden");
    $("segmentsPanel").classList.remove("hidden");
    updateLabelLayout();
    updateLockPill();
    renderVideoList({ preserveScroll: true });
    startLockHeartbeat();

    const meta = await api(`/api/videos/${id}/meta`);
    state.meta = meta;
    state.segments = (meta.segments || []).map((s) => normalizeSeg(s));
    renderAnnotateMeta(meta);
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
    setMobilePanel("modeLabel", "stage");
  }

  async function saveAnnotations({ offerNext = false } = {}) {
    if (!state.videoId || !state.meta) return;
    persistCollabName();
    const annotator = collabDisplayName();
    const payload = {
      filename: state.meta.filename,
      fps: state.meta.fps,
      width: state.meta.width,
      height: state.meta.height,
      total_frames: state.meta.total_frames,
      duration: state.meta.duration,
      segments: state.segments,
      annotator,
    };
    const data = await api(`/api/annotations/${state.videoId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (typeof data.revision === "number") {
      state.libraryRevision = data.revision;
    }
    const fresh = await api(`/api/videos/${state.videoId}/meta`);
    state.meta = { ...state.meta, ...fresh };
    renderAnnotateMeta(state.meta);
    await reloadVideosKeepingScroll();
    await loadLabelCounts();
    if (offerNext) {
      toast(`Saved by ${annotator}. Click Next video to continue.`, "ok");
      $("btnNextUnlabeled")?.classList.add("pulse-once");
      setTimeout(() => $("btnNextUnlabeled")?.classList.remove("pulse-once"), 1800);
    } else {
      toast(`Saved by ${annotator}`, "ok");
    }
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
      if (lastNew?.video && files.length === 1 && !state.videoId) {
        setNavTab("videos");
        if (lastNew.processing || lastNew.video.processing_status === "transcoding") {
          toast(lastNew.message || "Uploaded — converting in background", "ok");
        } else {
          await selectVideo(lastNew.video.id);
        }
      } else if (uploaded.length) {
        await reloadVideosKeepingScroll();
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

  $("btnSelectAllNeed")?.addEventListener("click", () => selectAllInTab("need"));
  $("btnClearSelNeed")?.addEventListener("click", () => clearSelectionInTab("need"));
  $("btnDeleteSelNeed")?.addEventListener("click", () => deleteSelectedInTab("need"));
  $("btnSelectAllHas")?.addEventListener("click", () => selectAllInTab("has"));
  $("btnClearSelHas")?.addEventListener("click", () => clearSelectionInTab("has"));
  $("btnDeleteSelHas")?.addEventListener("click", () => deleteSelectedInTab("has"));

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
    const labels = [...state.labels, name];
    const activities = [...(state.activities || []), name];
    try {
      const data = await api("/api/labels", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          labels,
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
    seekBySeconds(-1);
  };
  $("btnNext").onclick = () => {
    seekBySeconds(1);
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

  $("btnCancelEdit").onclick = () => {
    clearSegmentEdit();
    toast("Edit cancelled");
  };

  $("btnSaveSeg").onclick = () => {
    if (state.pendingStart == null || state.pendingEnd == null) {
      toast("Mark start and end first", "error");
      return;
    }
    let start = state.pendingStart;
    let end = state.pendingEnd;
    if (end < start) [start, end] = [end, start];
    const activity = $("activitySelect").value;
    if (!activity) {
      toast("Pick an activity", "error");
      return;
    }
    const warn = cropWarnMessage(state.cropDraft);
    const editing = state.editingSegIdx != null;
    const prev = editing ? state.segments[state.editingSegIdx] : null;
    const seg = {
      id: prev?.id || Math.random().toString(36).slice(2, 10),
      activity,
      label: activity,
      start_frame: start,
      end_frame: end,
      bbox: state.cropDraft
        ? [state.cropDraft.x1, state.cropDraft.y1, state.cropDraft.x2, state.cropDraft.y2]
        : null,
      note: prev?.note || "",
    };
    if (editing) {
      state.segments[state.editingSegIdx] = seg;
      state.editingSegIdx = null;
    } else {
      state.segments.push(seg);
    }
    state.pendingStart = null;
    state.pendingEnd = null;
    state.cropDraft = null;
    updateCropWarn();
    updatePending();
    updateSegmentEditorUI();
    renderSegments();
    const verb = editing ? "Updated" : "Saved";
    toast(
      warn
        ? `${verb} ${seg.label} ${start}–${end} (crop looks large — prefer one person)`
        : `${verb} ${seg.label} ${start}–${end}`,
      warn ? "error" : "ok"
    );
  };

  $("btnSaveAll").onclick = () =>
    saveAnnotations({ offerNext: true }).catch((e) => toast(e.message, "error"));

  async function loadMoreNeedPages(count = 1) {
    const needPages = state.videoPaging?.needPages || 1;
    let loaded = 0;
    while (loaded < count && state.videoPage.need < needPages) {
      state.videoPage.need++;
      const q = (state.videoQuery || "").trim();
      const params = new URLSearchParams({
        per_page: PER_PAGE,
        page: String(state.videoPage.need),
        labeled: "false",
      });
      if (q) params.set("q", q);
      const needData = await api(`/api/videos?${params}`);
      const existingIds = new Set(state.videos.map((v) => v.id));
      (needData.videos || []).forEach((v) => {
        if (!existingIds.has(v.id)) state.videos.push(v);
      });
      state.videoPaging = {
        ...state.videoPaging,
        needTotal: needData.total || state.videoPaging.needTotal,
        needPages: needData.pages || state.videoPaging.needPages,
        total_all: needData.total_all || state.videoPaging.total_all,
        total_labeled: needData.total_labeled || state.videoPaging.total_labeled,
      };
      loaded++;
    }
    return loaded;
  }

  async function goNextUnlabeled() {
    try {
      const prevId = state.videoId;
      // Snapshot Need order while current video is still in place (even if just labeled)
      const needOrderBefore = (state.videos || [])
        .filter((v) => !(v.segments > 0) || v.id === prevId)
        .map((v) => v.id);
      const prevNeedIdx = prevId ? needOrderBefore.indexOf(prevId) : -1;
      const seenNeedIds = new Set(needOrderBefore);

      await reloadVideosKeepingScroll();

      const isAvailable = (v) => {
        if (!v || v.id === prevId) return false;
        if (v.segments > 0) return false;
        const lock = state.locks?.[v.id];
        if (!lock) return true;
        return lock.client_id === state.clientId;
      };

      const findInLoaded = () => {
        // Prefer the next ids that were below the current one in the list
        if (prevNeedIdx >= 0) {
          for (let i = prevNeedIdx + 1; i < needOrderBefore.length; i++) {
            const v = (state.videos || []).find((x) => x.id === needOrderBefore[i]);
            if (isAvailable(v)) return v;
          }
        }
        return null;
      };

      let next = findInLoaded();

      // End of loaded Need list → fetch next page(s) instead of jumping to #1
      while (!next && state.videoPage.need < (state.videoPaging?.needPages || 1)) {
        toast("Loading more videos…", "ok");
        const added = await loadMoreNeedPages(1);
        if (!added) break;
        // First newly loaded unlabeled video (API order = list order)
        next = (state.videos || []).find((v) => !seenNeedIds.has(v.id) && isAvailable(v)) || null;
        (state.videos || []).forEach((v) => {
          if (!(v.segments > 0)) seenNeedIds.add(v.id);
        });
        if (!next) next = findInLoaded();
      }

      const available = (state.videos || []).filter(isAvailable);
      if (!next) {
        if (!available.length) {
          toast("No unlabeled videos left (or all are locked)", "ok");
          return;
        }
        next = available[0];
        toast(`Reached end of Need list · opened first unlabeled: ${next.filename || next.id}`);
      } else {
        toast(`Opened next unlabeled: ${next.filename || next.id}`);
      }

      setNavTab("videos");
      setLibraryTab("need");
      renderVideoList({ preserveScroll: true });
      await selectVideo(next.id);
      requestAnimationFrame(() => {
        const active = document.querySelector("#videoListNeed li.active");
        active?.scrollIntoView({ block: "nearest", behavior: "smooth" });
      });
      setMobilePanel("modeLabel", "stage");
    } catch (err) {
      toast(err.message, "error");
    }
  }

  $("btnNextUnlabeled")?.addEventListener("click", () => {
    goNextUnlabeled();
  });

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
    updateCropWarn();
    drawOverlay();
  };

  function frameSize() {
    const w = state.meta?.width || videoEl.videoWidth || 0;
    const h = state.meta?.height || videoEl.videoHeight || 0;
    return { w, h };
  }

  function cropCoverage(draft) {
    if (!draft) return 0;
    const { w, h } = frameSize();
    if (!w || !h) return 0;
    const bw = Math.max(0, draft.x2 - draft.x1);
    const bh = Math.max(0, draft.y2 - draft.y1);
    return (bw * bh) / (w * h);
  }

  function cropWarnMessage(draft) {
    if (!draft) return "";
    const cov = cropCoverage(draft);
    if (cov >= 0.7) {
      return "Crop covers most of the frame — prefer a tight box around one person.";
    }
    if (cov >= 0.4) {
      return "Crop looks large — try tighter around one person.";
    }
    return "";
  }

  function updateCropWarn() {
    const el = $("cropWarn");
    if (!el) return;
    const msg = cropWarnMessage(state.cropDraft);
    if (msg) {
      el.hidden = false;
      el.textContent = msg;
    } else {
      el.hidden = true;
      el.textContent = "";
    }
  }

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
    if (state.drawing) {
      state.drawing = false;
      updateCropWarn();
      const msg = cropWarnMessage(state.cropDraft);
      if (msg) toast(msg, "error");
    } else {
      state.drawing = false;
    }
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

  function getActiveLibraryTab() {
    const panelHas = $("libraryPanelHas");
    if (panelHas && !panelHas.hidden) return "has";
    return "need";
  }

  // Keyboard shortcuts (label mode only)
  window.addEventListener("keydown", (e) => {
    if (document.getElementById("modeLabel").classList.contains("hidden")) return;
    if (e.target.matches("input, textarea, select")) return;
    if (e.code === "Space") {
      e.preventDefault();
      $("btnPlay").click();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      seekBySeconds(e.shiftKey ? -5 : -1);
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      seekBySeconds(e.shiftKey ? 5 : 1);
    } else if (e.key === "s" || e.key === "S") {
      $("btnMarkStart").click();
    } else if (e.key === "e" || e.key === "E") {
      $("btnMarkEnd").click();
    } else if (e.key === "Enter") {
      $("btnSaveSeg").click();
    } else if (e.key === "n" || e.key === "N") {
      e.preventDefault();
      goNextUnlabeled();
    } else if (e.key === "a" || e.key === "A") {
      e.preventDefault();
      if (!state.videoId) {
        toast("Open a video first to select it", "error");
        return;
      }
      const id = state.videoId;
      if (state.selectedVideos.has(id)) {
        state.selectedVideos.delete(id);
        toast("Deselected current video");
      } else {
        state.selectedVideos.add(id);
        toast("Selected current video");
      }
      renderVideoList({ preserveScroll: true });
    } else if (e.shiftKey && (e.key === "Delete" || e.key === "Backspace")) {
      e.preventDefault();
      deleteAllSelected();
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
    libVideo: null,
    compareMode: false,
    compareJobA: null,
    compareJobB: null,
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
            <div class="name" title="${escapeHtml(name)}">${escapeHtml(name)}</div>
            <div class="meta">${formatTestDate(item.finished_at)} · ${escapeHtml(item.checkpoint || "—")}</div>
            <div class="meta-line">
              <span class="badge done">${item.person_count} person${item.person_count === 1 ? "" : "s"}</span>
              <span class="badge">${escapeHtml(item.summary || "")}</span>
            </div>
          </div>
          <button type="button" class="btn-delete-video" title="Delete result" aria-label="Delete result">×</button>
        </div>`;
      li.querySelector(".video-info").onclick = () => {
        selectTestResult(item.job_id).catch((e) => toast(e.message, "error"));
      };
      li.querySelector(".btn-delete-video").onclick = (e) => {
        e.stopPropagation();
        deleteTestResult(item.job_id, name).catch((err) => toast(err.message, "error"));
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

  async function deleteTestResult(jobId, label) {
    const ok = window.confirm(`Delete result for "${label}"? This removes the output video.`);
    if (!ok) return;
    await api(`/api/test/result/${encodeURIComponent(jobId)}`, { method: "DELETE" });
    if (testState.jobId === jobId) {
      testState.jobId = null;
      if (testState.pollTimer) {
        clearInterval(testState.pollTimer);
        testState.pollTimer = null;
      }
      $("testActive")?.classList.add("hidden");
      $("testCompare")?.classList.add("hidden");
      $("testEmpty")?.classList.remove("hidden");
      $("testPill").textContent = "idle";
      $("testPill").className = "status-pill";
      $("testLog").textContent = "Result deleted.";
    }
    toast("Result deleted", "ok");
    await loadTestLibrary(true);
  }

  async function deleteTestUpload(filename) {
    const ok = window.confirm(`Delete test upload "${filename}"?`);
    if (!ok) return;
    await api(`/api/test/inputs/${encodeURIComponent(filename)}`, { method: "DELETE" });
    if (testState.libVideo?.filename === filename) {
      selectTestLibVideo(null);
    }
    toast("Test upload deleted", "ok");
    await loadTestLibraryVideos($("testLibraryVideoSearch")?.value || "");
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
    $("srcLibrary")?.classList.toggle("active", source === "library");
    $("srcCamera").classList.toggle("active", source === "camera");
    $("panelTestVideo").classList.toggle("hidden", source !== "video");
    $("panelTestLibrary")?.classList.toggle("hidden", source !== "library");
    $("panelTestCamera").classList.toggle("hidden", source !== "camera");
    if (source === "video" || source === "library") stopCamera();
    if (source === "library") {
      loadTestLibraryVideos($("testLibraryVideoSearch")?.value || "").catch((e) => toast(e.message, "error"));
    }
  }

  $("srcVideo").onclick = () => setTestSource("video");
  $("srcLibrary")?.addEventListener("click", () => setTestSource("library"));
  $("srcCamera").onclick = () => setTestSource("camera");

  function updateTestLibSelectionUI() {
    const v = testState.libVideo;
    const card = $("testLibSelected");
    const btn = $("btnRunTestLibrary");
    const btnCmp = $("btnCompareLibrary");
    if (!card || !btn) return;
    if (!v) {
      card.hidden = true;
      btn.disabled = true;
      btn.textContent = "Run inference";
      if (btnCmp) {
        btnCmp.disabled = true;
        btnCmp.textContent = "Compare A vs B";
      }
      return;
    }
    card.hidden = false;
    $("testLibSelectedName").textContent = v.filename;
    $("testLibSelectedMeta").textContent = v.meta || "Test upload";
    btn.disabled = false;
    btn.textContent = "Run on selected video";
    if (btnCmp) {
      btnCmp.disabled = false;
      btnCmp.textContent = "Compare A vs B";
    }
  }

  function setTestRunMode(mode) {
    const compare = mode === "compare";
    testState.compareMode = compare;
    $("runModeSingle")?.classList.toggle("active", !compare);
    $("runModeCompare")?.classList.toggle("active", compare);
    $("modelSelectBWrap")?.classList.toggle("hidden", !compare);
    if ($("modelSelectLabel")) {
      $("modelSelectLabel").textContent = compare ? "Checkpoint A" : "Checkpoint";
    }
    if ($("runModeHint")) {
      $("runModeHint").textContent = compare
        ? "Runs A then B on the same clip (one after another)."
        : "Run one checkpoint on a clip.";
    }
    $("btnRunTest")?.classList.toggle("hidden", compare);
    $("btnCompareUpload")?.classList.toggle("hidden", !compare);
    $("btnRunTestLibrary")?.classList.toggle("hidden", compare);
    $("btnCompareLibrary")?.classList.toggle("hidden", !compare);
    $("srcCamera")?.classList.toggle("hidden", compare);
    if (compare && testState.source === "camera") {
      setTestSource("video");
    }
    if (compare) {
      loadModels();
    }
  }

  $("runModeSingle")?.addEventListener("click", () => setTestRunMode("single"));
  $("runModeCompare")?.addEventListener("click", () => setTestRunMode("compare"));

  function formatBytes(n) {
    if (!n && n !== 0) return "";
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  }

  function selectTestLibVideo(video) {
    testState.libVideo = video
      ? {
          filename: video.filename,
          id: video.id,
          meta: [formatBytes(video.size), formatTestDate(video.mtime)].filter(Boolean).join(" · "),
        }
      : null;
    updateTestLibSelectionUI();
    $("testLibraryList")
      ?.querySelectorAll("li[data-id]")
      .forEach((li) => {
        li.classList.toggle("active", !!video && li.dataset.id === video.filename);
      });
  }

  async function loadTestLibraryVideos(q = "") {
    const list = $("testLibraryList");
    if (!list) return;
    const qs = new URLSearchParams({ per_page: "80", page: "1" });
    if (q.trim()) qs.set("q", q.trim());
    list.innerHTML = `<li class="video-group-empty">Loading…</li>`;
    const data = await api(`/api/test/inputs?${qs}`);
    const videos = data.videos || [];
    const total = data.total ?? videos.length;
    list.innerHTML = "";
    if (!videos.length) {
      list.innerHTML = `<li class="video-group-empty">${
        q.trim() ? "No uploads match that search." : "No test uploads yet."
      }</li>`;
      $("testLibraryHint").textContent = q.trim()
        ? "Try another search."
        : "Use Upload first — those files appear here for re-run.";
      updateTestLibSelectionUI();
      return;
    }
    videos.forEach((v, i) => {
      const li = document.createElement("li");
      li.dataset.id = v.filename;
      li.innerHTML = `
        <div class="video-row">
          <span class="video-num">${i + 1}</span>
          <div class="video-info">
            <div class="name" title="${escapeHtml(v.filename)}">${escapeHtml(v.filename)}</div>
            <div class="meta">${formatBytes(v.size)} · ${formatTestDate(v.mtime)}</div>
          </div>
          <button type="button" class="btn-delete-video" title="Delete upload" aria-label="Delete ${escapeHtml(v.filename)}">×</button>
        </div>`;
      if (testState.libVideo?.filename === v.filename) li.classList.add("active");
      li.querySelector(".video-info").onclick = () => selectTestLibVideo(v);
      li.querySelector(".video-info").ondblclick = () => {
        selectTestLibVideo(v);
        $("btnRunTestLibrary")?.click();
      };
      li.querySelector(".btn-delete-video").onclick = (e) => {
        e.stopPropagation();
        deleteTestUpload(v.filename).catch((err) => toast(err.message, "error"));
      };
      list.appendChild(li);
    });
    const more = total > videos.length ? ` · showing ${videos.length}` : "";
    $("testLibraryHint").textContent = `${total} test upload${total === 1 ? "" : "s"}${more}`;
    updateTestLibSelectionUI();
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  let testLibSearchTimer = null;
  $("testLibraryVideoSearch")?.addEventListener("input", () => {
    clearTimeout(testLibSearchTimer);
    testLibSearchTimer = setTimeout(() => {
      loadTestLibraryVideos($("testLibraryVideoSearch").value).catch((e) => toast(e.message, "error"));
    }, 250);
  });

  $("btnRefreshTestLibVideos")?.addEventListener("click", () => {
    loadTestLibraryVideos($("testLibraryVideoSearch")?.value || "").catch((e) => toast(e.message, "error"));
  });

  $("btnClearTestLib")?.addEventListener("click", () => selectTestLibVideo(null));

  async function startTestJob(formData, startingMsg) {
    $("testPill").textContent = "starting";
    $("testPill").className = "status-pill running";
    $("testLog").textContent = startingMsg;
    const res = await fetch("/api/test/run", { method: "POST", body: formData });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || data.error || res.statusText);
    }
    testState.jobId = data.job.job_id;
    toast("Inference started — this may take a few minutes");
    if (testState.pollTimer) clearInterval(testState.pollTimer);
    testState.pollTimer = setInterval(pollTest, 2500);
    pollTest();
    setMobilePanel("modeTest", "stage");
  }

  function setMobilePanel(layoutId, panel) {
    const layout = $(layoutId);
    if (!layout) return;
    layout.classList.remove("mobile-panel-nav", "mobile-panel-stage", "mobile-panel-annotate");
    layout.classList.add(`mobile-panel-${panel}`);
    const switchId = layoutId === "modeTest" ? "mobileSwitchTest" : "mobileSwitchLabel";
    const sw = $(switchId);
    if (sw) {
      sw.querySelectorAll(".mobile-switch-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.panel === panel);
      });
    }
    // Annotate only useful when a video is open
    if (layoutId === "modeLabel" && panel === "annotate" && !state.videoId) {
      // still show annotate rail empty hint via CSS
    }
    relayoutPlayers();
  }

  function wireMobileSwitch(switchId, layoutId) {
    $(switchId)?.querySelectorAll(".mobile-switch-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (layoutId === "modeLabel" && btn.dataset.panel === "annotate" && !state.videoId) {
          toast("Pick a video first", "error");
          setMobilePanel(layoutId, "nav");
          return;
        }
        setMobilePanel(layoutId, btn.dataset.panel);
      });
    });
  }

  wireMobileSwitch("mobileSwitchLabel", "modeLabel");
  wireMobileSwitch("mobileSwitchTest", "modeTest");

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
      if (testState.source === "library") {
        loadTestLibraryVideos($("testLibraryVideoSearch")?.value || "").catch(() => {});
      }
    }
    setMobilePanel(isLabel ? "modeLabel" : "modeTest", "nav");
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
      fillModelSelect($("modelSelectB"), models, "No checkpoints found — train first");
      fillModelSelect($("onnxSourceSelect"), pthModels, "No .pth checkpoints — train first");
      if (!models.length) {
        $("modelHint").textContent = "No .pth/.onnx files in work_dirs/slowfast_multilabel/";
        return;
      }
      const selected = $("modelSelect")?.value;
      const picked = models.find((m) => m.name === selected) || models.find((m) => m.recommended) || models[0];
      // Prefer a different second model for compare
      const selB = $("modelSelectB");
      if (selB && models.length > 1) {
        const other =
          models.find((m) => m.name !== picked?.name && m.recommended) ||
          models.find((m) => m.name !== picked?.name) ||
          models[0];
        if (!selB.value || selB.value === picked?.name) selB.value = other.name;
      }
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
    $("testCompare")?.classList.add("hidden");
    $("testActive").classList.remove("hidden");
    $("liveStack").classList.add("hidden");
    $("resultStack").classList.remove("hidden");
    $("btnDownloadResult").classList.remove("hidden");
    setMobilePanel("modeTest", "stage");
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

  function fillPredTable(bodyId, persons) {
    const body = $(bodyId);
    if (!body) return;
    body.innerHTML = "";
    if (!persons?.length) {
      body.innerHTML = '<tr><td colspan="4">No persons detected</td></tr>';
      return;
    }
    persons.forEach((p) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>P${p.id}</td>
        <td>${p.posture || "—"}</td>
        <td>${p.activity || "—"}</td>
        <td>${((p.score || 0) * 100).toFixed(1)}%</td>`;
      body.appendChild(tr);
    });
  }

  function showCompareResults(jobA, jobB) {
    stopCamera();
    $("testEmpty").classList.add("hidden");
    $("testActive").classList.add("hidden");
    $("testCompare")?.classList.remove("hidden");
    setMobilePanel("modeTest", "stage");

    const nameA = PathName(jobA.checkpoint) || "Model A";
    const nameB = PathName(jobB.checkpoint) || "Model B";
    if ($("compareTitleA")) $("compareTitleA").textContent = `A · ${nameA}`;
    if ($("compareTitleB")) $("compareTitleB").textContent = `B · ${nameB}`;

    const urlA = `/api/test/result/${jobA.job_id}/video?t=${Date.now()}`;
    const urlB = `/api/test/result/${jobB.job_id}/video?t=${Date.now()}`;
    const vidA = $("resultVideoA");
    const vidB = $("resultVideoB");
    if (vidA) {
      vidA.src = urlA;
      vidA.load();
    }
    if (vidB) {
      vidB.src = urlB;
      vidB.load();
    }
    if ($("btnDownloadCompareA")) {
      $("btnDownloadCompareA").href = urlA;
      $("btnDownloadCompareA").download = `${jobA.job_id}.mp4`;
    }
    if ($("btnDownloadCompareB")) {
      $("btnDownloadCompareB").href = urlB;
      $("btnDownloadCompareB").download = `${jobB.job_id}.mp4`;
    }
    fillPredTable("predBodyA", jobA.persons || []);
    fillPredTable("predBodyB", jobB.persons || []);
    relayoutPlayers();
  }

  async function waitForTestJob(jobId, label) {
    for (;;) {
      const data = await api(`/api/test/status?job_id=${encodeURIComponent(jobId)}`);
      const job = data.job;
      if (!job) throw new Error(`${label} job missing`);
      const lines = job.log || [];
      $("testLog").textContent = `[${label}] ${lines.length ? lines.slice(-6).join("\n") : `(${job.status})`}`;
      const done = job.status === "completed" || Boolean(job.finished_at && job.output_video);
      if (done) return job;
      if (job.status === "failed") {
        throw new Error(job.error || `${label} failed`);
      }
      await new Promise((r) => setTimeout(r, 2500));
    }
  }

  async function uploadTestFileOnce(file) {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/test/upload", { method: "POST", body: fd });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.error || res.statusText);
    return data.filename;
  }

  async function startCompareOnTestVideo(testVideoName) {
    const ckptA = $("modelSelect")?.value;
    const ckptB = $("modelSelectB")?.value;
    if (!ckptA || !ckptB) {
      toast("Pick checkpoint A and B", "error");
      return;
    }
    if (ckptA === ckptB) {
      toast("Pick two different checkpoints", "error");
      return;
    }
    stopCamera();
    if (testState.pollTimer) {
      clearInterval(testState.pollTimer);
      testState.pollTimer = null;
    }
    try {
      $("testPill").textContent = "compare A";
      $("testPill").className = "status-pill running";
      $("testLog").textContent = `Compare: starting A (${ckptA})…`;

      const fdA = new FormData();
      fdA.append("checkpoint", ckptA);
      fdA.append("test_video", testVideoName);
      const resA = await fetch("/api/test/run", { method: "POST", body: fdA });
      const dataA = await resA.json().catch(() => ({}));
      if (!resA.ok) throw new Error(dataA.detail || dataA.error || resA.statusText);
      const jobIdA = dataA.job.job_id;
      testState.jobId = jobIdA;
      toast("Compare: running model A…");
      const jobA = await waitForTestJob(jobIdA, "A");

      $("testPill").textContent = "compare B";
      $("testLog").textContent = `Compare: starting B (${ckptB})…`;
      const fdB = new FormData();
      fdB.append("checkpoint", ckptB);
      fdB.append("test_video", testVideoName);
      const resB = await fetch("/api/test/run", { method: "POST", body: fdB });
      const dataB = await resB.json().catch(() => ({}));
      if (!resB.ok) throw new Error(dataB.detail || dataB.error || resB.statusText);
      const jobIdB = dataB.job.job_id;
      testState.jobId = jobIdB;
      toast("Compare: running model B…");
      const jobB = await waitForTestJob(jobIdB, "B");

      testState.compareJobA = jobA.job_id;
      testState.compareJobB = jobB.job_id;
      $("testPill").textContent = "completed";
      $("testPill").className = "status-pill completed";
      $("testLog").textContent = `Compare done.\nA: ${ckptA}\nB: ${ckptB}`;
      showCompareResults(jobA, jobB);
      loadTestLibrary(true).catch(() => {});
      toast("Compare complete — A vs B side by side", "ok");
    } catch (err) {
      toast(err.message, "error");
      $("testPill").textContent = "failed";
      $("testPill").className = "status-pill failed";
    }
  }

  $("btnCompareUpload")?.addEventListener("click", async () => {
    if (!testState.file) {
      toast("Upload a test video first", "error");
      return;
    }
    try {
      $("testPill").textContent = "uploading";
      $("testPill").className = "status-pill running";
      $("testLog").textContent = "Uploading once for compare…";
      const filename = await uploadTestFileOnce(testState.file);
      await startCompareOnTestVideo(filename);
    } catch (err) {
      toast(err.message, "error");
      $("testPill").textContent = "failed";
      $("testPill").className = "status-pill failed";
    }
  });

  $("btnCompareLibrary")?.addEventListener("click", async () => {
    const filename = testState.libVideo?.filename;
    if (!filename) {
      toast("Pick a test upload", "error");
      return;
    }
    await startCompareOnTestVideo(filename);
  });

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
      const fd = new FormData();
      fd.append("file", testState.file);
      fd.append("checkpoint", checkpoint);
      await startTestJob(fd, "Uploading and starting inference…");
    } catch (err) {
      toast(err.message, "error");
      $("testPill").textContent = "failed";
      $("testPill").className = "status-pill failed";
    }
  };

  $("btnRunTestLibrary")?.addEventListener("click", async () => {
    stopCamera();
    const checkpoint = $("modelSelect").value;
    const filename = testState.libVideo?.filename;
    if (!checkpoint) {
      toast("Select a model checkpoint", "error");
      return;
    }
    if (!filename) {
      toast("Pick a test upload", "error");
      return;
    }
    try {
      const fd = new FormData();
      fd.append("checkpoint", checkpoint);
      fd.append("test_video", filename);
      await startTestJob(fd, `Starting inference on ${filename}…`);
    } catch (err) {
      toast(err.message, "error");
      $("testPill").textContent = "failed";
      $("testPill").className = "status-pill failed";
    }
  });

  function showLiveStage() {
    $("testEmpty").classList.add("hidden");
    $("testCompare")?.classList.add("hidden");
    $("testActive").classList.remove("hidden");
    $("resultStack").classList.add("hidden");
    $("liveStack").classList.remove("hidden");
    $("btnDownloadResult").classList.add("hidden");
    setMobilePanel("modeTest", "stage");
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
      const nameInput = $("collabName");
      if (nameInput) {
        nameInput.value = state.collabName || "";
        nameInput.addEventListener("input", () => {
          persistCollabName();
        });
        nameInput.addEventListener("change", persistCollabName);
      }
      await ensureCollabClient();
      await loadLabels();
      await loadVideos();
      await loadModels();
      startCollabPolling();
      startLockHeartbeat();
      await pollTrain();
      if (state.trainJobId) {
        state.pollTimer = setInterval(pollTrain, 2500);
      }
    } catch (err) {
      toast(err.message, "error");
    }
  }

  window.addEventListener("beforeunload", () => {
    if (!state.clientId) return;
    const payload = JSON.stringify({ client_id: state.clientId, name: collabDisplayName() });
    try {
      navigator.sendBeacon?.(
        "/api/collab/bye",
        new Blob([payload], { type: "application/json" })
      );
    } catch (_) {
      /* ignore */
    }
  });

  init();
})();
