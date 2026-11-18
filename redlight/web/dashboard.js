(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const state = { cameras: [], configs: [], events: [], engines: {detectors: [], trackers: []}, view: 'overview' };
  const labels = {overview:'TỔNG QUAN', live:'XEM CAMERA', events:'SỰ KIỆN VI PHẠM', cameras:'QUẢN LÝ CAMERA'};
  const statusLabels = {running:'Đang chạy', starting:'Đang kết nối', reconnecting:'Đang thử lại', stopped:'Đã dừng', error:'Lỗi kết nối'};
  const reviewLabels = {pending:'Chờ kiểm tra', confirmed:'Đã xác nhận', dismissed:'Không vi phạm'};
  let toastTimer;
  let liveSignature = '';

  function toast(message) {
    const el = $('toast'); el.textContent = message; el.classList.add('show');
    clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('show'), 3500);
  }
  async function api(path, options = {}) {
    const res = await fetch(path, {credentials:'same-origin', ...options});
    if (res.status === 401) { if (!$('loginDialog').open) $('loginDialog').showModal(); throw new Error('Cần đăng nhập quản trị'); }
    if (!res.ok) {
      let message = `Lỗi ${res.status}`;
      try { message = (await res.json()).detail || message; } catch (_) { /* no JSON */ }
      throw new Error(message);
    }
    return res.json();
  }
  function view(name) {
    if (!labels[name]) return;
    if (state.view === 'live' && name !== 'live') { $('liveGrid').innerHTML = ''; liveSignature = ''; }
    state.view = name;
    document.querySelectorAll('.view').forEach(el => el.classList.toggle('active', el.id === `view-${name}`));
    document.querySelectorAll('.nav-item').forEach(el => el.classList.toggle('active', el.dataset.view === name));
    $('breadcrumbCurrent').textContent = labels[name];
    if (name === 'live') renderLive();
    history.replaceState(null, '', `#${name}`);
  }
  function ready(cam) {
    const cfg = state.configs.find(c => c.camera_id === cam.camera_id);
    return Boolean(cfg?.has_stop_line && cfg?.has_light_roi);
  }
  function statusBadge(cam) {
    const status = cam.status || 'stopped';
    return `<span class="badge ${esc(status)}">${esc(statusLabels[status] || status)}</span>`;
  }
  const empty = (title, subtitle) => `<div class="empty"><strong>${esc(title)}</strong>${esc(subtitle)}</div>`;
  const date = (iso) => { try { return new Date(iso).toLocaleString('vi-VN', {dateStyle:'short',timeStyle:'short'}); } catch (_) { return iso; } };

  async function refresh() {
    try {
      const [cameras, configs, events, engines] = await Promise.all([
        api('/api/managed-cameras'), api('/api/cameras'), api('/api/events'), api('/api/engines')
      ]);
      Object.assign(state, {cameras, configs, events, engines});
      render();
    } catch (err) { if (!String(err.message).includes('đăng nhập')) toast(err.message); }
  }
  function render() {
    $('statCameras').textContent = state.cameras.length;
    $('statRunning').textContent = state.cameras.filter(c => c.status === 'running').length;
    $('statEvents').textContent = state.events.length < 100 ? state.events.length : '100+';
    $('statSetup').textContent = state.cameras.filter(c => !ready(c)).length;
    $('overviewCameras').innerHTML = state.cameras.length ? state.cameras.slice(0, 5).map(c => `<div class="list-row"><div class="camera-avatar">▣</div><div class="list-main"><strong>${esc(c.name)}</strong><small>${esc(c.camera_id)} · ${esc(c.source)}</small></div>${statusBadge(c)}</div>`).join('') : empty('Chưa có camera', 'Bấm “Thêm camera” để bắt đầu.');
    $('overviewEvents').innerHTML = state.events.length ? state.events.slice(0, 5).map(e => `<div class="list-row"><div class="camera-avatar" style="background:#fff1eb;color:#d36539">⚑</div><div class="list-main"><strong>Vượt đèn đỏ · ${esc(e.camera_id)}</strong><small>${esc(date(e.occurred_at))} · Xe #${esc(e.track_id)}</small></div><button class="text-button" data-event="${esc(e.event_id)}">Xem →</button></div>`).join('') : empty('Chưa có sự kiện', 'Sự kiện sẽ xuất hiện sau khi bật xử lý camera.');
    renderCameras(); renderEvents();
    if (state.view === 'live') renderLive();
    const filter = $('eventFilter'), previous = filter.value;
    filter.innerHTML = '<option value="">Tất cả camera</option>' + state.cameras.map(c => `<option value="${esc(c.camera_id)}">${esc(c.name)}</option>`).join('');
    filter.value = previous;
  }
  function renderCameras() {
    const table = $('cameraTable');
    if (!state.cameras.length) { table.innerHTML = empty('Chưa có camera nào', 'Thêm camera RTSP hoặc video để bắt đầu cấu hình.'); return; }
    table.innerHTML = `<div class="table-row table-head"><span>Camera</span><span>Nguồn</span><span>Engine phát hiện</span><span>Engine bám vết</span><span>Trạng thái</span><span>Thao tác</span></div>` + state.cameras.map(c => `<div class="table-row"><div><div class="cell-title">${esc(c.name)}</div><div class="cell-sub">${esc(c.camera_id)}${ready(c) ? '' : ' · Cần vẽ vùng'}</div></div><div>${esc(c.source_type === 'rtsp' ? 'RTSP' : 'Video')}<div class="cell-sub">${esc(c.source)}</div></div><div>${esc(c.detector)}</div><div>${esc(c.tracker)}</div><div>${statusBadge(c)}${c.error ? `<div class="status-note">${esc(c.error)}</div>` : ''}</div><div class="actions"><button class="action" data-edit="${esc(c.camera_id)}">Sửa</button><button class="action" data-geometry="${esc(c.camera_id)}">Vẽ vùng</button><button class="action" data-toggle="${esc(c.camera_id)}">${['running','starting','reconnecting'].includes(c.status) ? 'Dừng' : 'Bật'}</button><button class="action danger" data-delete="${esc(c.camera_id)}">Xóa</button></div></div>`).join('');
  }
  function renderEvents() {
    const filter = $('eventFilter').value;
    const events = state.events.filter(e => !filter || e.camera_id === filter);
    $('eventsTable').innerHTML = events.length ? `<div class="table-row table-head"><span>Sự kiện</span><span>Thời gian</span><span>Camera</span><span>Xe</span><span>Chi tiết</span></div>` + events.map(e => `<div class="table-row"><div class="event-cell"><img class="event-thumb" src="/api/events/${esc(e.camera_id)}/${esc(e.event_id)}/${e.region_image ? 'region' : 'image'}.jpg" alt="Ảnh sự kiện"><div><strong>Nghi vấn vượt đèn đỏ</strong><div class="cell-sub">${esc(reviewLabels[e.review_status || 'pending'])}</div></div></div><span>${esc(date(e.occurred_at))}</span><span>${esc(e.camera_id)}</span><span>${esc(e.vehicle_class)} #${esc(e.track_id)}</span><button class="text-button" data-event="${esc(e.event_id)}">Xem ảnh →</button></div>`).join('') : empty('Chưa có sự kiện', 'Thử chọn camera khác hoặc bật xử lý.');
  }
  function renderLive() {
    const grid = $('liveGrid');
    const signature = state.cameras.map(c => `${c.camera_id}:${c.name}:${c.status}:${c.error}`).join('|');
    if (signature === liveSignature && grid.childElementCount) return;
    liveSignature = signature;
    if (!state.cameras.length) { grid.innerHTML = empty('Chưa có camera', 'Thêm camera để xem luồng ở đây.'); return; }
    grid.innerHTML = state.cameras.map(c => `<article class="live-card"><div class="live-screen">${c.status === 'running' ? `<img src="/api/managed-cameras/${esc(c.camera_id)}/preview.mjpg?t=${Date.now()}" alt="Luồng ${esc(c.name)}">` : `<span>${c.status === 'error' || c.status === 'reconnecting' ? esc(c.error || 'Lỗi luồng') : 'Camera chưa bật xử lý'}</span>`}</div><div class="live-info"><div><strong>${esc(c.name)}</strong><small>${statusLabels[c.status] || 'Đã dừng'} · ${esc(c.fps || 0)} FPS</small></div><button class="action" data-toggle="${esc(c.camera_id)}">${['running','starting','reconnecting'].includes(c.status) ? 'Dừng' : 'Bật xử lý'}</button></div></article>`).join('');
  }
  function openForm(camera = null) {
    $('cameraForm').reset();
    $('formError').hidden = true;
    $('dialogTitle').textContent = camera ? 'Sửa camera' : 'Thêm camera';
    $('cameraId').value = camera?.camera_id || '';
    $('cameraId').disabled = !!camera;
    $('cameraName').value = camera?.name || '';
    $('sourceType').value = camera?.source_type || 'rtsp';
    $('cameraSource').value = '';
    $('cameraEnabled').checked = camera?.enabled ?? true;
    for (const kind of ['detector','tracker']) {
      const names = state.engines[kind === 'detector' ? 'detectors' : 'trackers'];
      $(kind).innerHTML = names.map(name => `<option value="${esc(name)}">${esc(name)}</option>`).join('');
      $(kind).value = camera?.[kind] || names[0];
    }
    sourceMode(); $('cameraDialog').showModal(); $('cameraId').focus();
  }
  function sourceMode() {
    const video = $('sourceType').value === 'video';
    $('sourceLabel').hidden = video;
    $('uploadLabel').hidden = !video;
    $('sourceHelp').textContent = 'Ví dụ: rtsp://user:password@192.168.1.10:554/stream';
    $('cameraSource').placeholder = $('cameraId').disabled ? 'Để trống nếu giữ nguyên nguồn' : 'rtsp://...';
  }
  async function saveForm(event) {
    event.preventDefault();
    const id = $('cameraId').value.trim();
    const body = {name:$('cameraName').value.trim(), source_type:$('sourceType').value,
      source:$('cameraSource').value.trim(), detector:$('detector').value, tracker:$('tracker').value,
      enabled:$('cameraEnabled').checked};
    const error = $('formError'); error.hidden = true;
    try {
      if (body.source_type === 'video' && $('videoUpload').files[0]) {
        const fd = new FormData(); fd.append('file', $('videoUpload').files[0]);
        const uploaded = await api('/api/videos/upload', {method:'POST', body:fd}); body.source = uploaded.path;
      }
      await api(`/api/managed-cameras/${encodeURIComponent(id)}`, {method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      $('cameraDialog').close(); toast('Đã lưu camera'); await refresh(); view('cameras');
    } catch (err) { error.textContent = err.message; error.hidden = false; }
  }
  function showEvent(id) {
    const e = state.events.find(item => item.event_id === id); if (!e) return;
    $('eventDetail').innerHTML = `<p style="font-size:12px;color:#72849a">Khung hình gốc</p><img src="/api/events/${esc(e.camera_id)}/${esc(e.event_id)}/image.jpg" alt="Khung hình bằng chứng">${e.region_image ? `<p style="font-size:12px;color:#72849a">Vùng xảy ra sự kiện</p><img src="/api/events/${esc(e.camera_id)}/${esc(e.event_id)}/region.jpg" alt="Vùng xảy ra sự kiện">` : ''}${e.plate_image ? `<p style="font-size:12px;color:#72849a">Biển số được phát hiện</p><img src="/api/events/${esc(e.camera_id)}/${esc(e.event_id)}/plate.jpg" alt="Biển số được phát hiện">` : ''}<div class="event-meta"><div><small>Thời điểm</small><strong>${esc(date(e.occurred_at))}</strong></div><div><small>Camera</small><strong>${esc(e.camera_id)}</strong></div><div><small>Phương tiện</small><strong>${esc(e.vehicle_class)} · ID ${esc(e.track_id)}</strong></div><div><small>Biển số</small><strong>${e.plate_text ? esc(e.plate_text) : 'Chưa đọc được · ' + esc(e.plate_status || 'chưa xử lý')}</strong></div></div>`;
    const controls = document.createElement('div');
    controls.className = 'review-controls';
    controls.innerHTML = `<span>Trạng thái: <strong>${esc(reviewLabels[e.review_status || 'pending'])}</strong></span><div><button class="secondary-button" data-review="dismissed" data-id="${esc(e.event_id)}">Không vi phạm</button> <button class="primary-button" data-review="confirmed" data-id="${esc(e.event_id)}">Xác nhận vi phạm</button></div>`;
    $('eventDetail').appendChild(controls);
    $('eventDialog').showModal();
  }
  async function action(event) {
    const button = event.target.closest('button'); if (!button) return;
    const d = button.dataset;
    if (d.view) { view(d.view); return; }
    if (d.action === 'add') { openForm(); return; }
    if (d.geometry) { location.href = `/editor?camera=${encodeURIComponent(d.geometry)}`; return; }
    if (d.event) { showEvent(d.event); return; }
    if (d.review) {
      const e = state.events.find(item => item.event_id === d.id); if (!e) return;
      try {
        await api(`/api/events/${encodeURIComponent(e.camera_id)}/${encodeURIComponent(e.event_id)}/review`,
          {method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({status:d.review})});
        $('eventDialog').close(); await refresh(); toast('Đã cập nhật kết quả kiểm tra');
      } catch (err) { toast(err.message); }
      return;
    }
    if (d.edit) { openForm(state.cameras.find(c => c.camera_id === d.edit)); return; }
    if (d.toggle) {
      const c = state.cameras.find(c => c.camera_id === d.toggle);
      if (!c) return;
      if (c.status !== 'running' && c.status !== 'starting' && c.status !== 'reconnecting' && !ready(c)) {
        toast('Hãy vẽ vạch dừng và vùng đèn trước khi bật xử lý');
        location.href = `/editor?camera=${encodeURIComponent(c.camera_id)}`; return;
      }
      try { await api(`/api/managed-cameras/${encodeURIComponent(c.camera_id)}/${['running','starting','reconnecting'].includes(c.status) ? 'stop' : 'start'}`, {method:'POST'}); await refresh(); }
      catch (err) { toast(err.message); }
      return;
    }
    if (d.delete) {
      if (!confirm(`Xóa camera ${d.delete}? Sự kiện đã ghi vẫn được giữ lại.`)) return;
      try { await api(`/api/managed-cameras/${encodeURIComponent(d.delete)}`, {method:'DELETE'}); await refresh(); toast('Đã xóa camera'); }
      catch (err) { toast(err.message); }
    }
  }
  document.body.addEventListener('click', action);
  $('cameraForm').addEventListener('submit', saveForm);
  $('sourceType').addEventListener('change', sourceMode);
  $('closeDialog').onclick = $('cancelDialog').onclick = () => $('cameraDialog').close();
  $('closeEvent').onclick = () => $('eventDialog').close();
  $('eventFilter').onchange = renderEvents;
  $('refresh').onclick = refresh;
  $('helpStart').onclick = () => openForm();
  $('loginForm').addEventListener('submit', async e => {
    e.preventDefault();
    const token = $('adminToken').value;
    try {
      const res = await fetch('/api/auth/login', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})});
      if (!res.ok) throw new Error('Mã quản trị không đúng');
      $('adminToken').value = ''; $('loginDialog').close(); await refresh();
    } catch (err) { $('loginError').textContent = err.message; $('loginError').hidden = false; }
  });
  view(location.hash.slice(1) in labels ? location.hash.slice(1) : 'overview');
  refresh();
  setInterval(() => { if (!document.hidden) refresh(); }, 10000);
})();
