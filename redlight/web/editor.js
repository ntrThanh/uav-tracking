/* Camera Geometry & Traffic Light Editor
   Clean, friendly, zero-emoji UI logic. */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const canvas = $('canvas');
  const ctx = canvas.getContext('2d');

  const COLORS = {
    line: '#ff1744',
    lineArrow: '#d50000',
    zone: '#2563eb',
    zoneFill: 'rgba(37, 99, 235, 0.22)',
    zoneDraft: '#d97706',
    roi: '#f59e0b',
    roiFill: 'rgba(245, 158, 11, 0.2)',
  };

  const DIRS = {
    up: [0, -1],
    down: [0, 1],
    left: [-1, 0],
    right: [1, 0],
  };

  const HINTS = {
    select: 'Chế độ xem: Click vào đối tượng để kiểm tra, dùng nút góc trên để phóng to/thu nhỏ ảnh.',
    stopline: 'Vẽ vạch dừng: Click điểm A rồi click điểm B để tạo vạch. Mũi tên chỉ hướng xe được phép qua.',
    zone: 'Vẽ vùng làn đường: Click lần lượt các đỉnh. Nháy đúp chuột hoặc bấm Enter để hoàn tất đa giác.',
    roi: 'Vẽ vùng đèn: Nhấp giữ và kéo chuột tạo khung chữ nhật bao trọn cụm đèn tín hiệu.',
  };

  const state = {
    cameraId: new URLSearchParams(location.search).get('camera') || 'cam01',
    img: null,
    config: emptyConfig('cam01'),
    tool: 'select',
    draft: [],
    drag: null,
    zoom: 1.0,
  };

  function emptyConfig(id) {
    return {
      camera_id: id,
      reference_frame: null,
      stop_line: null,
      zones: [],
      light_roi: null,
      light_override: null,
    };
  }

  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  const r4 = (v) => Math.round(v * 10000) / 10000;
  const toPx = ([x, y]) => [x * canvas.width, y * canvas.height];

  function toNorm(evt) {
    const rect = canvas.getBoundingClientRect();
    const clientX = evt.clientX - rect.left;
    const clientY = evt.clientY - rect.top;
    const normX = clamp01(clientX / rect.width);
    const normY = clamp01(clientY / rect.height);
    return [r4(normX), r4(normY)];
  }

  // ------------------------------------------------------------- Toast Notification
  function showToast(message, type = 'info') {
    const container = $('toastContainer');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast-msg ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(6px)';
      toast.style.transition = 'all 0.2s';
      setTimeout(() => toast.remove(), 200);
    }, 3000);
  }

  function setStatus(text) {
    const el = $('statusText');
    if (el) el.textContent = text;
  }

  const escapeHtml = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // ------------------------------------------------------------- Tool Selection
  function setTool(toolName) {
    state.tool = toolName;
    state.draft = [];
    state.drag = null;

    document.querySelectorAll('.tool-btn').forEach((btn) => {
      btn.classList.toggle('active', btn.getAttribute('data-tool') === toolName);
    });

    const activeLabel = $('activeToolName');
    if (activeLabel) {
      const names = { select: 'Chế độ: Xem / Chọn', stopline: 'Chế độ: Vẽ vạch dừng', zone: 'Chế độ: Vẽ vùng làn đường', roi: 'Chế độ: Vẽ vùng đèn' };
      activeLabel.textContent = names[toolName] || toolName;
    }

    const hint = $('hintText');
    if (hint) {
      hint.textContent = HINTS[toolName] || '';
    }

    // Show/hide direction selector
    const dirCard = $('lineSettingCard');
    if (dirCard) {
      dirCard.style.display = (toolName === 'stopline' || state.config.stop_line) ? 'block' : 'none';
    }

    $('btnFinishZone').disabled = toolName !== 'zone' || state.draft.length < 3;
    $('btnUndo').disabled = state.draft.length === 0;

    canvas.style.cursor = toolName === 'select' ? 'default' : 'crosshair';
    draw();
  }

  function initTools() {
    document.querySelectorAll('.tool-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        setTool(btn.getAttribute('data-tool'));
      });
    });

    document.querySelectorAll('.dir-item').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.dir-item').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const dir = btn.getAttribute('data-dir');
        $('direction').value = dir;
        if (state.config.stop_line) {
          state.config.stop_line.allowed_direction = dir;
          updateLayersList();
          updateJson();
          draw();
        }
      });
    });
  }

  // ------------------------------------------------------------- Canvas Drawing
  function fitCanvas() {
    const parent = $('canvasViewport');
    if (!parent) return;
    const maxW = Math.max(360, parent.clientWidth - 40);
    const maxH = Math.max(360, parent.clientHeight - 40);
    const iw = state.img ? state.img.naturalWidth : 1280;
    const ih = state.img ? state.img.naturalHeight : 720;
    const scale = Math.min(maxW / iw, maxH / ih);

    canvas.width = Math.round(iw * scale * state.zoom);
    canvas.height = Math.round(ih * scale * state.zoom);
    draw();
  }

  function drawHandle([x, y], color, label = null) {
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();

    if (label) {
      ctx.font = '600 11px Inter, sans-serif';
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x + 8, y + 4);
    }
  }

  function drawLabel(text, [x, y], color) {
    ctx.font = '600 11px Inter, sans-serif';
    const textW = ctx.measureText(text).width + 12;
    const yy = Math.max(18, y - 6);

    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.beginPath();
    ctx.roundRect(x, yy - 14, textW, 18, 3);
    ctx.fill();

    ctx.fillStyle = color;
    ctx.fillRect(x, yy - 14, 3, 18);

    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, x + 7, yy);
  }

  function drawPolygon(pts, color, fillColor, label, closed) {
    if (!pts.length) return;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    pts.slice(1).forEach((p) => ctx.lineTo(p[0], p[1]));
    if (closed) ctx.closePath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.stroke();

    if (closed && fillColor) {
      ctx.fillStyle = fillColor;
      ctx.fill();
    }

    pts.forEach((p, i) => drawHandle(p, color, closed ? null : String(i + 1)));
    if (label && pts.length > 0) drawLabel(label, pts[0], color);
  }

  function drawStopLine(line) {
    const a = toPx(line.a);
    const b = toPx(line.b);

    // Stopline
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = COLORS.line;
    ctx.stroke();

    // Direction arrow
    const [dx, dy] = DIRS[line.allowed_direction] || DIRS.down;
    const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const arrowLen = 42;
    const tip = [m[0] + arrowLen * dx, m[1] + arrowLen * dy];

    ctx.beginPath();
    ctx.moveTo(m[0], m[1]);
    ctx.lineTo(tip[0], tip[1]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = COLORS.lineArrow;
    ctx.stroke();

    const ang = Math.atan2(dy, dx);
    ctx.beginPath();
    ctx.moveTo(tip[0], tip[1]);
    ctx.lineTo(tip[0] - 10 * Math.cos(ang - 0.45), tip[1] - 10 * Math.sin(ang - 0.45));
    ctx.lineTo(tip[0] - 10 * Math.cos(ang + 0.45), tip[1] - 10 * Math.sin(ang + 0.45));
    ctx.closePath();
    ctx.fillStyle = COLORS.lineArrow;
    ctx.fill();

    drawHandle(a, COLORS.line, 'A');
    drawHandle(b, COLORS.line, 'B');
    drawLabel(`Vạch dừng (${line.allowed_direction})`, [m[0] + 8, m[1] - 8], COLORS.line);
  }

  function drawRect(r, color, fillColor, label) {
    const [x, y] = toPx([r.x, r.y]);
    const rw = r.w * canvas.width;
    const rh = r.h * canvas.height;

    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, rw, rh);

    if (fillColor) {
      ctx.fillStyle = fillColor;
      ctx.fillRect(x, y, rw, rh);
    }

    drawHandle([x, y], color);
    drawHandle([x + rw, y], color);
    drawHandle([x, y + rh], color);
    drawHandle([x + rw, y + rh], color);

    if (label) drawLabel(label, [x, y], color);
  }

  function rectFromDrag(d) {
    return {
      x: r4(Math.min(d.start[0], d.cur[0])),
      y: r4(Math.min(d.start[1], d.cur[1])),
      w: r4(Math.abs(d.cur[0] - d.start[0])),
      h: r4(Math.abs(d.cur[1] - d.start[1])),
    };
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (state.img) {
      ctx.drawImage(state.img, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.fillStyle = '#94a3b8';
      ctx.font = '600 15px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Chưa có ảnh hoặc video tham chiếu', canvas.width / 2, canvas.height / 2 - 10);
      ctx.font = '400 12px Inter, sans-serif';
      ctx.fillStyle = '#64748b';
      ctx.fillText('Nhập đường dẫn video ở cột bên trái hoặc tải tệp ảnh lên', canvas.width / 2, canvas.height / 2 + 14);
      ctx.textAlign = 'left';
    }

    // Draw saved stopline
    if (state.config.stop_line) {
      drawStopLine(state.config.stop_line);
    }

    // Draw saved zones
    state.config.zones.forEach((z) => {
      drawPolygon(z.polygon.map(toPx), COLORS.zone, COLORS.zoneFill, z.name, true);
    });

    // Draw saved light ROI
    if (state.config.light_roi) {
      drawRect(state.config.light_roi, COLORS.roi, COLORS.roiFill, 'Vùng đèn tín hiệu');
    }

    // Draw active drafts
    if (state.tool === 'stopline' && state.draft.length === 1) {
      drawHandle(toPx(state.draft[0]), COLORS.line, 'A');
    } else if (state.tool === 'zone' && state.draft.length > 0) {
      drawPolygon(state.draft.map(toPx), COLORS.zoneDraft, null, null, false);
    } else if (state.tool === 'roi' && state.drag) {
      drawRect(rectFromDrag(state.drag), COLORS.roi, COLORS.roiFill, 'Đang kéo vùng đèn...');
    }
  }

  // ------------------------------------------------------------- Mouse Handling
  canvas.addEventListener('mousemove', (e) => {
    const [nx, ny] = toNorm(e);
    const pxW = state.img ? state.img.naturalWidth : canvas.width;
    const pxH = state.img ? state.img.naturalHeight : canvas.height;
    $('coords').textContent = `Tọa độ: (${nx.toFixed(3)}, ${ny.toFixed(3)}) | Pixel: (${Math.round(nx * pxW)}, ${Math.round(ny * pxH)})`;

    if (state.tool === 'roi' && state.drag) {
      state.drag.cur = [nx, ny];
      draw();
    }
  });

  canvas.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    const p = toNorm(e);

    if (state.tool === 'stopline') {
      if (state.draft.length === 0) {
        state.draft = [p];
        $('btnUndo').disabled = false;
        draw();
      } else {
        const dir = $('direction').value || 'down';
        state.config.stop_line = { a: state.draft[0], b: p, allowed_direction: dir };
        state.draft = [];
        $('btnUndo').disabled = true;
        updateLayersList();
        updateJson();
        showToast('Đã tạo vạch dừng thành công', 'success');
        setTool('select');
      }
    } else if (state.tool === 'zone') {
      state.draft.push(p);
      $('btnFinishZone').disabled = state.draft.length < 3;
      $('btnUndo').disabled = false;
      draw();
    } else if (state.tool === 'roi') {
      state.drag = { start: p, cur: p };
    }
  });

  canvas.addEventListener('mouseup', () => {
    if (state.tool === 'roi' && state.drag) {
      const r = rectFromDrag(state.drag);
      state.drag = null;
      if (r.w >= 0.01 && r.h >= 0.01) {
        state.config.light_roi = r;
        updateLayersList();
        updateJson();
        updateLightCropPreview();
        showToast('Đã xác định vùng đèn tín hiệu', 'success');
        setTool('select');
      } else {
        showToast('Khung vẽ quá nhỏ, vui lòng kéo lại', 'error');
        draw();
      }
    }
  });

  canvas.addEventListener('dblclick', () => {
    if (state.tool === 'zone') {
      finishZone();
    }
  });

  function finishZone() {
    if (state.draft.length < 3) {
      showToast('Vùng đa giác cần ít nhất 3 điểm', 'error');
      return;
    }
    const defaultName = `Vùng_${state.config.zones.length + 1}`;
    const name = window.prompt('Nhập tên cho vùng làn đường / giao lộ:', defaultName);
    if (!name) return;

    state.config.zones.push({ name: name.trim(), polygon: state.draft });
    state.draft = [];
    $('btnFinishZone').disabled = true;
    $('btnUndo').disabled = true;
    updateLayersList();
    updateJson();
    showToast(`Đã thêm: ${name}`, 'success');
    setTool('select');
  }

  function undoLastPoint() {
    if (state.draft.length > 0) {
      state.draft.pop();
      $('btnFinishZone').disabled = state.tool !== 'zone' || state.draft.length < 3;
      $('btnUndo').disabled = state.draft.length === 0;
      draw();
    }
  }

  // ------------------------------------------------------------- Layers & UI Updates
  function updateLayersList() {
    const lineLayer = $('layerStopLine');
    if (state.config.stop_line) {
      lineLayer.style.display = 'flex';
      $('subStopLine').textContent = `Hướng cho phép: ${state.config.stop_line.allowed_direction}`;
    } else {
      lineLayer.style.display = 'none';
    }

    const roiLayer = $('layerLightRoi');
    if (state.config.light_roi) {
      roiLayer.style.display = 'flex';
      const r = state.config.light_roi;
      $('subLightRoi').textContent = `Tọa độ: (${r.x}, ${r.y}, ${r.w}, ${r.h})`;
    } else {
      roiLayer.style.display = 'none';
    }

    const zonesCont = $('zoneListContainer');
    zonesCont.innerHTML = '';
    state.config.zones.forEach((z, idx) => {
      const item = document.createElement('div');
      item.className = 'element-item';
      item.innerHTML = `
        <div class="element-indicator color-blue"></div>
        <div class="element-details">
          <div class="element-name">${escapeHtml(z.name)}</div>
          <div class="element-meta">${z.polygon.length} đỉnh đa giác</div>
        </div>
        <button class="btn-text-danger" type="button" data-idx="${idx}">Xóa</button>
      `;
      zonesCont.appendChild(item);
    });

    zonesCont.querySelectorAll('.btn-text-danger').forEach((btn) => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.getAttribute('data-idx'), 10);
        state.config.zones.splice(idx, 1);
        updateLayersList();
        updateJson();
        draw();
      });
    });

    const hasAny = state.config.stop_line || state.config.light_roi || state.config.zones.length > 0;
    $('emptyLayersMsg').style.display = hasAny ? 'none' : 'block';
  }

  function updateJson() {
    const pre = $('json');
    if (pre) {
      pre.textContent = JSON.stringify(state.config, null, 2);
    }
  }

  // ------------------------------------------------------------- Traffic Light Preview & API
  function updateLightCropPreview() {
    const emptyMsg = $('lightCropEmpty');
    const imgEl = $('lightCropImg');
    const roi = state.config.light_roi;

    if (!roi || !state.img) {
      if (emptyMsg) emptyMsg.style.display = 'block';
      if (imgEl) imgEl.style.display = 'none';
      return;
    }

    const iw = state.img.naturalWidth;
    const ih = state.img.naturalHeight;
    const x0 = Math.round(roi.x * iw);
    const y0 = Math.round(roi.y * ih);
    const w = Math.max(1, Math.round(roi.w * iw));
    const h = Math.max(1, Math.round(roi.h * ih));

    const scratch = document.createElement('canvas');
    scratch.width = w;
    scratch.height = h;
    const sctx = scratch.getContext('2d');
    sctx.drawImage(state.img, x0, y0, w, h, 0, 0, w, h);

    if (imgEl) {
      imgEl.src = scratch.toDataURL('image/jpeg');
      imgEl.style.display = 'block';
    }
    if (emptyMsg) emptyMsg.style.display = 'none';
  }

  async function testLightDetection() {
    if (!state.config.light_roi) {
      showToast('Vui lòng vẽ vùng đèn trước khi kiểm tra', 'error');
      return;
    }

    setStatus('Đang phân tích màu đèn...');
    try {
      const res = await fetch(`/api/cameras/${state.cameraId}/detect-light`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roi: state.config.light_roi }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Lỗi kiểm tra đèn');
      }

      const data = await res.json();
      const stateNames = { red: 'ĐÈN ĐỎ', yellow: 'ĐÈN VÀNG', green: 'ĐÈN XANH', unknown: 'TẮT / CHƯA RÕ' };
      const text = $('lightResultText');
      const confBar = $('lightConfBar');
      const confVal = $('lightConfVal');

      if (text) {
        text.textContent = stateNames[data.state] || data.state;
        text.className = `light-status-label ${data.state}`;
      }
      if (confBar) confBar.style.width = `${Math.round(data.confidence * 100)}%`;
      if (confVal) confVal.textContent = `Độ tin cậy: ${Math.round(data.confidence * 100)}% (độ sáng: ${(data.lit_ratio * 100).toFixed(1)}%)`;

      if (data.crop_base64) {
        const imgEl = $('lightCropImg');
        if (imgEl) {
          imgEl.src = data.crop_base64;
          imgEl.style.display = 'block';
        }
        const emptyMsg = $('lightCropEmpty');
        if (emptyMsg) emptyMsg.style.display = 'none';
      }

      showToast(`Kết quả nhận diện: ${stateNames[data.state]}`, 'success');
      setStatus('Sẵn sàng');
    } catch (err) {
      showToast(err.message, 'error');
      setStatus('Lỗi nhận diện');
    }
  }

  // ------------------------------------------------------------- Camera Config API
  async function loadCameraList() {
    try {
      const res = await fetch('/api/cameras');
      if (!res.ok) return;
      const items = await res.json();
      const select = $('cameraList');
      select.innerHTML = '<option value="">(Chọn camera đã lưu)</option>';
      items.forEach((c) => {
        const opt = document.createElement('option');
        opt.value = c.camera_id;
        const details = [];
        if (c.has_stop_line) details.push('Vạch dừng');
        if (c.n_zones > 0) details.push(`${c.n_zones} vùng`);
        if (c.has_light_roi) details.push('Đèn');
        opt.textContent = `${c.camera_id} ${details.length ? '(' + details.join(', ') + ')' : ''}`;
        select.appendChild(opt);
      });
    } catch (err) {
      console.warn('Cannot load camera list', err);
    }
  }

  async function loadCamera(id) {
    setStatus(`Đang tải ${id}...`);
    try {
      const res = await fetch(`/api/cameras/${id}`);
      if (!res.ok) throw new Error(`Không tìm thấy cấu hình ${id}`);
      const data = await res.json();
      state.cameraId = id;
      state.config = data;
      $('cameraId').value = id;

      if (data.stop_line) {
        const dir = data.stop_line.allowed_direction || 'down';
        $('direction').value = dir;
        document.querySelectorAll('.dir-item').forEach((b) => {
          b.classList.toggle('active', b.getAttribute('data-dir') === dir);
        });
      }

      await loadReferenceImage(id);
      updateLayersList();
      updateJson();
      updateLightCropPreview();
      showToast(`Đã tải camera: ${id}`, 'success');
      setStatus('Sẵn sàng');
    } catch (err) {
      showToast(err.message, 'error');
      setStatus('Lỗi tải camera');
    }
  }

  async function saveCamera() {
    const id = ($('cameraId').value || '').trim();
    if (!id) {
      showToast('Vui lòng nhập mã camera', 'error');
      return;
    }
    state.cameraId = id;
    state.config.camera_id = id;

    setStatus(`Đang lưu ${id}...`);
    try {
      const res = await fetch(`/api/cameras/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.config),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Lưu thất bại');
      }
      showToast(`Đã lưu cấu hình '${id}' thành công!`, 'success');
      setStatus('Sẵn sàng');
      loadCameraList();
    } catch (err) {
      showToast(`Lỗi: ${err.message}`, 'error');
      setStatus('Lỗi lưu cấu hình');
    }
  }

  async function loadReferenceImage(id) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        state.img = img;
        $('frameInfoCard').style.display = 'block';
        $('metaRes').textContent = `${img.naturalWidth} x ${img.naturalHeight} px`;
        $('metaFile').textContent = `${id}_ref.jpg`;
        fitCanvas();
        resolve(true);
      };
      img.onerror = () => {
        state.img = null;
        $('frameInfoCard').style.display = 'none';
        fitCanvas();
        resolve(false);
      };
      img.src = `/api/cameras/${id}/reference.jpg?t=${Date.now()}`;
    });
  }

  // ------------------------------------------------------------- Event Listeners
  function initEvents() {
    $('btnFromCamera').addEventListener('click', async () => {
      const id = ($('cameraId').value || '').trim();
      if (!id) return showToast('Nhập mã camera trước', 'error');
      setStatus('Đang lấy ảnh từ camera...');
      try {
        const response = await fetch(`/api/managed-cameras/${encodeURIComponent(id)}/reference`, {method:'POST'});
        if (!response.ok) throw new Error((await response.json()).detail || 'Không lấy được ảnh');
        state.cameraId = id;
        await loadReferenceImage(id);
        updateLightCropPreview();
        setStatus('Sẵn sàng'); showToast('Đã lấy ảnh từ camera', 'success');
      } catch (err) { setStatus('Lỗi lấy ảnh'); showToast(err.message, 'error'); }
    });
    $('btnSave').addEventListener('click', saveCamera);
    $('btnLoad').addEventListener('click', () => {
      const id = ($('cameraId').value || '').trim();
      if (id) loadCamera(id);
    });

    $('cameraList').addEventListener('change', (e) => {
      const val = e.target.value;
      if (val) loadCamera(val);
    });

    $('btnRefreshList').addEventListener('click', loadCameraList);

    $('btnDownload').addEventListener('click', () => {
      const blob = new Blob([JSON.stringify(state.config, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${state.cameraId || 'camera'}.json`;
      a.click();
      URL.revokeObjectURL(url);
      showToast('Đã xuất file JSON thành công', 'success');
    });

    $('btnCopyJson').addEventListener('click', () => {
      navigator.clipboard.writeText(JSON.stringify(state.config, null, 2)).then(() => {
        showToast('Đã sao chép JSON vào bộ nhớ tạm', 'success');
      });
    });

    $('btnFinishZone').addEventListener('click', finishZone);
    $('btnUndo').addEventListener('click', undoLastPoint);

    $('btnClearLine').addEventListener('click', () => {
      state.config.stop_line = null;
      updateLayersList();
      updateJson();
      draw();
      showToast('Đã xóa vạch dừng', 'info');
    });

    $('btnClearRoi').addEventListener('click', () => {
      state.config.light_roi = null;
      updateLayersList();
      updateJson();
      updateLightCropPreview();
      draw();
      showToast('Đã xóa vùng đèn', 'info');
    });

    $('btnTestLight').addEventListener('click', testLightDetection);

    // Frame from Video
    $('btnFromVideo').addEventListener('click', async () => {
      const video = ($('videoPath').value || '').trim();
      const timeS = parseFloat($('videoTime').value || 0);
      if (!video) {
        showToast('Vui lòng nhập đường dẫn video', 'error');
        return;
      }
      setStatus('Đang lấy frame từ video...');
      try {
        const res = await fetch(`/api/cameras/${state.cameraId}/reference/from-video`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ video_path: video, time_s: timeS }),
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Không thể lấy frame');
        }
        await loadReferenceImage(state.cameraId);
        updateLightCropPreview();
        showToast('Lấy frame thành công', 'success');
        setStatus('Sẵn sàng');
      } catch (err) {
        showToast(err.message, 'error');
        setStatus('Lỗi lấy frame');
      }
    });

    // Image Upload
    const imageInput = $('imageFile');
    imageInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      await uploadReferenceFile(file);
    });

    const dropzone = $('dropzone');
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', async (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        await uploadReferenceFile(e.dataTransfer.files[0]);
      }
    });

    // Zoom Controls
    $('btnZoomIn').addEventListener('click', () => {
      state.zoom = Math.min(3.0, state.zoom + 0.25);
      $('zoomLevelText').textContent = `${Math.round(state.zoom * 100)}%`;
      fitCanvas();
    });

    $('btnZoomOut').addEventListener('click', () => {
      state.zoom = Math.max(0.5, state.zoom - 0.25);
      $('zoomLevelText').textContent = `${Math.round(state.zoom * 100)}%`;
      fitCanvas();
    });

    $('btnZoomReset').addEventListener('click', () => {
      state.zoom = 1.0;
      $('zoomLevelText').textContent = '100%';
      fitCanvas();
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key === 'Enter') {
        if (state.tool === 'zone') finishZone();
      } else if (e.key === 'Escape') {
        state.draft = [];
        state.drag = null;
        setTool('select');
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        undoLastPoint();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        saveCamera();
      }
    });

    window.addEventListener('resize', fitCanvas);
  }

  async function uploadReferenceFile(file) {
    const fd = new FormData();
    fd.append('file', file);
    setStatus('Đang tải ảnh lên...');
    try {
      const res = await fetch(`/api/cameras/${state.cameraId}/reference/upload`, {
        method: 'POST',
        body: fd,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Tải ảnh thất bại');
      }
      await loadReferenceImage(state.cameraId);
      updateLightCropPreview();
      showToast('Tải ảnh thành công', 'success');
      setStatus('Sẵn sàng');
    } catch (err) {
      showToast(err.message, 'error');
      setStatus('Lỗi tải ảnh');
    }
  }

  // ------------------------------------------------------------- Init
  function init() {
    initTools();
    initEvents();
    loadCameraList();
    $('cameraId').value = state.cameraId;
    loadCamera(state.cameraId);
    setTool('select');
    fitCanvas();
  }

  window.addEventListener('DOMContentLoaded', init);
})();
