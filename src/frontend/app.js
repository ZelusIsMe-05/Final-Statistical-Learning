/**
 * app.js — ViRAG Frontend Logic
 * Giao tiếp với Backend FastAPI (port 5000)
 */

// ============================================================
// CONFIG
// ============================================================
const BACKEND_URL = 'http://localhost:8000';  // URL backend Python
const HEALTH_INTERVAL = 30000;  // Kiểm tra trạng thái mỗi 30 giây

// ============================================================
// STATE
// ============================================================
const state = {
  hasDocument: false,
  isLoading: false,
  selectedFile: null,
  chatHistory: [],
  colabOnline: false,
};


// ============================================================
// KHỞI TẠO
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  // Theo dõi ký tự trong textarea
  const ta = document.getElementById('textInput');
  ta.addEventListener('input', updateCharCount);

  // Kéo thả PDF
  setupDropZone();

  // Kiểm tra trạng thái Colab API lúc đầu
  checkHealth();
  setInterval(checkHealth, HEALTH_INTERVAL);

  // Cập nhật UI ban đầu
  updateDocumentUI(false);
});


// ============================================================
// HEALTH CHECK
// ============================================================
async function checkHealth() {
  const dot = document.getElementById('colabStatus');
  const label = document.getElementById('colabLabel');
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { signal: AbortSignal.timeout(8000) });
    if (res.ok) {
      const data = await res.json();
      state.colabOnline = data.status === 'ok';
      dot.className = `status-dot ${state.colabOnline ? 'online' : 'offline'}`;
      label.textContent = state.colabOnline ? 'Server Online' : 'Server Offline';
    } else {
      throw new Error('Backend không phản hồi');
    }
  } catch (e) {
    dot.className = 'status-dot offline';
    label.textContent = 'Server Offline';
    state.colabOnline = false;
  }
}


// ============================================================
// TAB SWITCH
// ============================================================
function switchTab(type) {
  document.getElementById('tabText').classList.toggle('active', type === 'text');
  document.getElementById('tabPdf').classList.toggle('active', type === 'pdf');
  document.getElementById('sectionText').style.display = type === 'text' ? 'flex' : 'none';
  document.getElementById('sectionPdf').style.display = type === 'pdf' ? 'flex' : 'none';
}


// ============================================================
// CHAR COUNT
// ============================================================
function updateCharCount() {
  const ta = document.getElementById('textInput');
  const count = ta.value.length;
  const el = document.getElementById('charCount');
  el.textContent = `${count.toLocaleString('vi-VN')} ký tự`;
  el.style.color = count > 50000 ? 'var(--error)' : 'var(--text-muted)';
}


// ============================================================
// DROP ZONE
// ============================================================
function setupDropZone() {
  const zone = document.getElementById('dropZone');

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('dragover');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('dragover');
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (state.hasDocument) {
      showToast('Đã có tài liệu! Vui lòng xóa tài liệu hiện tại trước khi tải mới.', 'error');
      return;
    }
    const file = e.dataTransfer.files[0];
    if (file && file.name.toLowerCase().endsWith('.pdf')) {
      setSelectedFile(file);
    } else {
      showToast('Chỉ chấp nhận file PDF!', 'error');
    }
  });
}

function handleFileSelect(event) {
  if (state.hasDocument) {
    showToast('Đã có tài liệu! Vui lòng xóa tài liệu hiện tại trước khi tải mới.', 'error');
    event.target.value = '';
    return;
  }
  const file = event.target.files[0];
  if (file) setSelectedFile(file);
}

function setSelectedFile(file) {
  state.selectedFile = file;
  document.getElementById('fileName').textContent = file.name;
  document.getElementById('fileInfo').style.display = 'flex';
}


// ============================================================
// UPLOAD TEXT
// ============================================================
async function uploadText() {
  if (state.hasDocument) {
    showToast('Đã có tài liệu! Vui lòng xóa tài liệu hiện tại trước khi tải mới.', 'error');
    return;
  }
  const text = document.getElementById('textInput').value.trim();
  if (!text) {
    showToast('Vui lòng nhập văn bản trước!', 'error');
    return;
  }

  showLoading('Đang xử lý văn bản...', 'Chunking thông minh theo đề mục/câu...');

  try {
    const res = await fetch(`${BACKEND_URL}/upload/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Lỗi không xác định');

    updateDocumentUI(true, data);
    showToast(`Đã xử lý thành ${data.chunk_count} chunk!`, 'success');

  } catch (e) {
    showToast(`Lỗi: ${e.message}`, 'error');
  } finally {
    hideLoading();
  }
}


// ============================================================
// UPLOAD PDF
// ============================================================
async function uploadPdf() {
  if (state.hasDocument) {
    showToast('Đã có tài liệu! Vui lòng xóa tài liệu hiện tại trước khi tải mới.', 'error');
    return;
  }
  if (!state.selectedFile) {
    showToast('Chưa chọn file PDF!', 'error');
    return;
  }

  showLoading('Đang xử lý...');

  const formData = new FormData();
  formData.append('file', state.selectedFile);

  try {
    const res = await fetch(`${BACKEND_URL}/upload/pdf`, {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Lỗi đọc PDF');

    updateDocumentUI(true, data);
    showToast(`Đã tải file PDF thành công!`, 'success');

  } catch (e) {
    showToast(`Lỗi: ${e.message}`, 'error');
  } finally {
    hideLoading();
  }
}


// ============================================================
// SUMMARIZE
// ============================================================
async function summarizeDocument() {
  if (!state.hasDocument) {
    showToast('Chưa có tài liệu!', 'error');
    return;
  }

  const btn = document.getElementById('btnSummarize');
  btn.disabled = true;
  
  hideWelcome();
  addMessage('user', 'Hãy tóm tắt tài liệu này giúp tôi.');
  const typingId = addTypingIndicator();

  try {
    const res = await fetch(`${BACKEND_URL}/summarize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });

    const data = await res.json();
    removeTypingIndicator(typingId);
    
    if (!res.ok) throw new Error(data.detail || 'Lỗi tóm tắt');

    const summaryHtml = renderSummary(data.summary);

    addMessage('bot', summaryHtml, { isHtml: true });

    showToast('Tóm tắt hoàn thành!', 'success');

  } catch (e) {
    removeTypingIndicator(typingId);
    addMessage('bot', `Lỗi tóm tắt: ${e.message}`, { isError: true });
  } finally {
    btn.disabled = false;
  }
}


// ============================================================
// ASK QUESTION
// ============================================================
async function askQuestion() {
  const input = document.getElementById('questionInput');
  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  hideWelcome();

  // Hiện câu hỏi của user
  addMessage('user', question);

  // Hiện typing indicator
  const typingId = addTypingIndicator();

  const btn = document.getElementById('btnAsk');
  btn.disabled = true;
  input.disabled = true;

  try {
    const res = await fetch(`${BACKEND_URL}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: 3 })
    });

    const data = await res.json();
    removeTypingIndicator(typingId);

    if (!res.ok) throw new Error(data.detail || 'Lỗi server');

    addMessage('bot', data.answer, {
      hasContext: data.has_context,
      chunkCount: data.relevant_chunk_count
    });

  } catch (e) {
    removeTypingIndicator(typingId);
    addMessage('bot', `Lỗi: ${e.message}`, { isError: true });
  } finally {
    btn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}


// ============================================================
// CLEAR DOCUMENT
// ============================================================
async function clearDocument() {
  try {
    await fetch(`${BACKEND_URL}/document`, { method: 'DELETE' });
  } catch (_) { }

  updateDocumentUI(false);
  document.getElementById('textInput').value = '';
  document.getElementById('charCount').textContent = '0 ký tự';
  document.getElementById('fileInfo').style.display = 'none';
  document.getElementById('fileName').textContent = 'document.pdf';
  document.getElementById('pdfInput').value = '';
  state.selectedFile = null;
  showToast('Đã xóa tài liệu.', 'info');
}


// ============================================================
// CLEAR CHAT
// ============================================================
function clearChat() {
  const chatArea = document.getElementById('chatArea');
  chatArea.innerHTML = '';
  addWelcome();
  state.chatHistory = [];
}


// ============================================================
// UI HELPERS
// ============================================================
function updateDocumentUI(hasDoc, data = null) {
  state.hasDocument = hasDoc;

  const docInfo = document.getElementById('docInfo');
  const summarizeSection = document.getElementById('summarizeSection');
  const btnClear = document.getElementById('btnClearDoc');
  const noDocNotice = document.getElementById('noDocNotice');

  if (hasDoc && data) {
    docInfo.style.display = 'block';
    summarizeSection.style.display = 'block';
    btnClear.style.display = 'flex';
    noDocNotice.style.display = 'none';

    const sourceName = data.source === 'pdf' && state.selectedFile
      ? state.selectedFile.name
      : 'văn bản';
    document.getElementById('docSuccessMessage').textContent = `✓ Đã tải thành công ${sourceName}`;
  } else {
    docInfo.style.display = 'none';
    summarizeSection.style.display = 'none';
    btnClear.style.display = 'none';
    noDocNotice.style.display = 'flex';
  }
}

function addMessage(role, text, opts = {}) {
  const chatArea = document.getElementById('chatArea');
  const msg = document.createElement('div');
  msg.className = `msg msg-${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  if (opts.isError) {
    bubble.style.background = 'var(--error-bg)';
    bubble.style.color = 'var(--error)';
  } else if (role === 'bot' && !opts.hasContext && state.hasDocument === false) {
    bubble.classList.add('no-context');
  }

  if (opts.isHtml) {
    bubble.innerHTML = text;
  } else {
    bubble.textContent = text;
  }

  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  const now = new Date();
  meta.textContent = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;

  msg.appendChild(bubble);
  msg.appendChild(meta);
  chatArea.appendChild(msg);
  chatArea.scrollTop = chatArea.scrollHeight;

  state.chatHistory.push({ role, text, timestamp: now.toISOString() });
}

function escapeHtml(value) {
  const div = document.createElement('div');
  div.textContent = value;
  return div.innerHTML;
}

function renderSummary(summary) {
  return summary
    .split('\n\n')
    .map(section => {
      const lines = section
        .split('\n')
        .map(line => line.trim())
        .filter(Boolean);

      if (!lines.length) return '';
      if (lines.length === 1) {
        return `<span class="summary-para">${escapeHtml(lines[0])}</span>`;
      }

      const heading = escapeHtml(lines[0]);
      const chunkLines = lines
        .slice(1)
        .map(line => `<span class="summary-chunk">${escapeHtml(line)}</span>`)
        .join('');

      return `<span class="summary-para"><strong class="summary-heading">${heading}</strong>${chunkLines}</span>`;
    })
    .join('');
}

function addTypingIndicator() {
  const chatArea = document.getElementById('chatArea');
  const id = 'typing-' + Date.now();
  const div = document.createElement('div');
  div.id = id;
  div.className = 'msg msg-bot';
  div.innerHTML = `
    <div class="typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>`;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function hideWelcome() {
  const w = document.getElementById('chatWelcome');
  if (w) w.remove();
}

function addWelcome() {
  const chatArea = document.getElementById('chatArea');
  chatArea.innerHTML = `
    <div class="chat-welcome" id="chatWelcome">
      <div class="welcome-icon">💬</div>
      <p>Đặt câu hỏi về tài liệu của bạn</p>
      <p class="welcome-hint">Model sẽ tìm thông tin liên quan trong tài liệu đã tải</p>
    </div>`;
}

function showLoading(text, sub = '') {
  document.getElementById('loadingText').textContent = text;
  document.getElementById('loadingSub').textContent = sub;
  document.getElementById('loadingOverlay').style.display = 'flex';
  state.isLoading = true;
}

function hideLoading() {
  document.getElementById('loadingOverlay').style.display = 'none';
  state.isLoading = false;
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 3200);
}

function copyText(elementId) {
  const el = document.getElementById(elementId);
  if (!el) return;
  navigator.clipboard.writeText(el.textContent).then(() => {
    showToast('Đã sao chép!', 'success');
  });
}

// Removed toggleExpand as it's no longer needed


// Gắn nút clear document
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnClearDoc').addEventListener('click', clearDocument);
  initResizableDivider();
});


// ============================================================
// RESIZABLE DIVIDER — kéo thả phân chia tỉ lệ hai panel
// ============================================================
function initResizableDivider() {
  const divider = document.getElementById('resizeDivider');
  const left = document.getElementById('panelDocument');
  const right = document.getElementById('panelQA');
  const workspace = left.parentElement;

  if (!divider || !left || !right) return;

  let isDragging = false;
  let startX = 0;
  let startLeftW = 0;

  /** Chiều rộng khả dụng cho 2 panel (trừ padding workspace, gap và divider) */
  function getWorkspaceW() {
    const style = getComputedStyle(workspace);
    const gap = parseFloat(style.gap) || 16;
    const pl = parseFloat(style.paddingLeft) || 0;
    const pr = parseFloat(style.paddingRight) || 0;
    return workspace.clientWidth - pl - pr - divider.offsetWidth - gap * 2;
  }

  divider.addEventListener('mousedown', (e) => {
    isDragging = true;
    startX = e.clientX;
    startLeftW = left.getBoundingClientRect().width;
    divider.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;

    const dx = e.clientX - startX;
    const totalW = getWorkspaceW();

    // Giới hạn: tối thiểu 20%, tối đa 80%
    const newLeftPx = Math.min(Math.max(startLeftW + dx, totalW * 0.20), totalW * 0.80);
    const newRightPx = totalW - newLeftPx;

    // Dùng pixel thườ để tránh nhập nhằng với % relative to container
    left.style.flex = `0 0 ${newLeftPx.toFixed(1)}px`;
    right.style.flex = `0 0 ${newRightPx.toFixed(1)}px`;
  });

  document.addEventListener('mouseup', () => {
    if (!isDragging) return;
    isDragging = false;
    divider.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });

  // Touch support (mobile)
  divider.addEventListener('touchstart', (e) => {
    const t = e.touches[0];
    isDragging = true;
    startX = t.clientX;
    startLeftW = left.getBoundingClientRect().width;
    divider.classList.add('dragging');
    e.preventDefault();
  }, { passive: false });

  document.addEventListener('touchmove', (e) => {
    if (!isDragging) return;
    const t = e.touches[0];
    const totalW = getWorkspaceW();
    const newPx = Math.min(Math.max(startLeftW + t.clientX - startX, totalW * 0.20), totalW * 0.80);
    left.style.flex = `0 0 ${newPx.toFixed(1)}px`;
    right.style.flex = `0 0 ${(totalW - newPx).toFixed(1)}px`;
  }, { passive: true });

  document.addEventListener('touchend', () => {
    isDragging = false;
    divider.classList.remove('dragging');
  });
}
