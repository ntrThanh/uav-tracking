// LaTeX Studio - Client App
let monacoEditor = null;
let currentFilePath = "main.tex";
let isDirty = false;
let isCompiling = false;

// DOM Elements
const editorHost = document.getElementById("monaco-host");
const imageHost = document.getElementById("image-viewer-host");
const previewImage = document.getElementById("preview-image-elem");
const previewCaption = document.getElementById("preview-image-caption");
const activeFilename = document.getElementById("active-filename");
const unsavedBadge = document.getElementById("unsaved-badge");
const cursorPos = document.getElementById("cursor-pos");
const fileTreeContainer = document.getElementById("file-tree-container");
const pdfFrame = document.getElementById("pdf-frame");
const pdfEmptyState = document.getElementById("pdf-empty-state");
const statusPill = document.getElementById("status-pill");
const statusText = document.getElementById("status-text");
const btnCompile = document.getElementById("btn-compile");
const compileSpinner = document.getElementById("compile-spinner");
const compileText = document.getElementById("compile-text");
const selectPasses = document.getElementById("select-passes");
const btnSave = document.getElementById("btn-save");
const btnDownloadPdf = document.getElementById("btn-download-pdf");
const btnOpenPdfTab = document.getElementById("btn-open-pdf-tab");
const btnRefreshPdf = document.getElementById("btn-refresh-pdf");
const btnRefreshFiles = document.getElementById("btn-refresh-files");
const btnNewFile = document.getElementById("btn-new-file");
const btnUploadFile = document.getElementById("btn-upload-file");
const uploadInput = document.getElementById("upload-input");
const chkShowArtifacts = document.getElementById("chk-show-artifacts");
const btnToggleSidebar = document.getElementById("btn-toggle-sidebar");
const sidebar = document.getElementById("file-sidebar");
const btnThemeToggle = document.getElementById("btn-theme-toggle");
const resizer = document.getElementById("pane-resizer");
const editorPane = document.getElementById("editor-container-pane");
const previewPane = document.getElementById("preview-container-pane");

// Diagnostics Elements
const errorBadge = document.getElementById("error-badge");
const warningBadge = document.getElementById("warning-badge");
const diagSummaryText = document.getElementById("diag-summary-text");
const diagErrorsList = document.getElementById("diag-errors-list");
const diagWarningsList = document.getElementById("diag-warnings-list");
const diagErrCount = document.getElementById("diag-err-count");
const diagWarnCount = document.getElementById("diag-warn-count");
const rawLogView = document.getElementById("raw-log-view");
const btnFetchFullLog = document.getElementById("btn-fetch-full-log");

// Modal Elements
const modalNewFile = document.getElementById("modal-new-file");
const modalCloseBtn = document.getElementById("modal-close-btn");
const modalCancelBtn = document.getElementById("modal-cancel-btn");
const modalCreateBtn = document.getElementById("modal-create-btn");
const newFilenameInput = document.getElementById("new-filename-input");

// --- Initialize Monaco Editor ---
require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' } });

require(['vs/editor/editor.main'], function () {
  // 1. Register LaTeX language
  monaco.languages.register({ id: 'latex', extensions: ['.tex', '.sty', '.cls'] });

  monaco.languages.setLanguageConfiguration('latex', {
    comments: {
      lineComment: '%'
    },
    brackets: [
      ['{', '}'],
      ['[', ']'],
      ['(', ')']
    ],
    autoClosingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '$', close: '$' },
      { open: '"', close: '"' }
    ],
    surroundingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '$', close: '$' },
      { open: '"', close: '"' }
    ]
  });

  monaco.languages.setMonarchTokensProvider('latex', {
    defaultToken: '',
    tokenPostfix: '.latex',

    tokenizer: {
      root: [
        // Comments
        [/%[^\r\n]*/, 'comment'],

        // Display math $$ ... $$
        [/\$\$/, { token: 'string.math.delim', next: '@displayMath' }],
        [/\\\[/, { token: 'string.math.delim', next: '@displayMathBracket' }],

        // Inline math $ ... $
        [/\$((?:\\\$|[^$])*)\$/, 'string.math'],
        [/\\\(/, { token: 'string.math.delim', next: '@inlineMathParen' }],

        // Environments: \begin{figure}, \end{table}, etc.
        [/(\\begin|\\end)(\s*)(\{)([a-zA-Z0-9*_-]+)(\})/, [
          'keyword.control',
          '',
          'delimiter.curly',
          'type.environment',
          'delimiter.curly'
        ]],

        // Control / structural keywords
        [/\\(begin|end)\b/, 'keyword.control'],
        [/\\(documentclass|usepackage|input|include|bibliography|bibliographystyle|addbibresource)\b/, 'keyword.package'],
        [/\\(newcommand|renewcommand|providecommand|def|let|DeclareMathOperator)\b/, 'keyword.definition'],

        // Sections
        [/\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\b/, 'keyword.section'],

        // Citations & References
        [/\\(label|ref|pageref|eqref|cite|citep|citet|nocite|autoref|nameref|url|nolinkurl|href)\b/, 'keyword.ref'],

        // Generic commands: \textbf, \centering, \caption, \Repo, \CanDien, etc.
        [/\\([a-zA-Z@]+)/, 'keyword.command'],

        // Escaped symbols
        [/\\[%&$_#{}]/, 'keyword.command'],

        // Alignment and line breaks
        [/\\\\/, 'delimiter.special'],
        [/&/, 'delimiter.special'],

        // Brackets & Delimiters
        [/[{}]/, 'delimiter.curly'],
        [/[[\]]/, 'delimiter.bracket'],
        [/[()]/, 'delimiter.parenthesis'],

        // Numbers
        [/\b\d+([.,]\d+)?\b/, 'number'],
      ],

      displayMath: [
        [/\$\$/, { token: 'string.math.delim', next: '@pop' }],
        [/%[^\r\n]*/, 'comment'],
        [/\\([a-zA-Z]+)/, 'keyword.command'],
        [/./, 'string.math']
      ],

      displayMathBracket: [
        [/\\\]/, { token: 'string.math.delim', next: '@pop' }],
        [/%[^\r\n]*/, 'comment'],
        [/\\([a-zA-Z]+)/, 'keyword.command'],
        [/./, 'string.math']
      ],

      inlineMathParen: [
        [/\\\)/, { token: 'string.math.delim', next: '@pop' }],
        [/%[^\r\n]*/, 'comment'],
        [/\\([a-zA-Z]+)/, 'keyword.command'],
        [/./, 'string.math']
      ]
    }
  });

  // 2. Define VS Code Dark+ & Light+ themes for Monaco
  monaco.editor.defineTheme('vscode-dark-plus', {
    base: 'vs-dark',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '6A9955', fontStyle: 'italic' },
      { token: 'keyword.control', foreground: 'C586C0', fontStyle: 'bold' },
      { token: 'keyword.package', foreground: '569CD6', fontStyle: 'bold' },
      { token: 'keyword.section', foreground: '4EC9B0', fontStyle: 'bold' },
      { token: 'keyword.definition', foreground: 'C586C0' },
      { token: 'keyword.ref', foreground: '4FC1FF' },
      { token: 'type.environment', foreground: '4EC9B0', fontStyle: 'bold' },
      { token: 'keyword.command', foreground: 'DCDCAA' },
      { token: 'string.math', foreground: 'CE9178' },
      { token: 'string.math.delim', foreground: '569CD6', fontStyle: 'bold' },
      { token: 'delimiter.curly', foreground: 'FFD700' },
      { token: 'delimiter.bracket', foreground: 'DA70D6' },
      { token: 'delimiter.parenthesis', foreground: '179FFF' },
      { token: 'delimiter.special', foreground: '569CD6', fontStyle: 'bold' },
      { token: 'number', foreground: 'B5CEA8' },
    ],
    colors: {
      'editor.background': '#1e1e24',
      'editor.foreground': '#d4d4d4',
      'editorLineNumber.foreground': '#6e7681',
      'editorLineNumber.activeForeground': '#ffffff',
      'editor.selectionBackground': '#264f78',
      'editor.inactiveSelectionBackground': '#3a3d41',
    }
  });

  monaco.editor.defineTheme('vscode-light-plus', {
    base: 'vs',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '008000', fontStyle: 'italic' },
      { token: 'keyword.control', foreground: 'AF00DB', fontStyle: 'bold' },
      { token: 'keyword.package', foreground: '0000FF', fontStyle: 'bold' },
      { token: 'keyword.section', foreground: '001080', fontStyle: 'bold' },
      { token: 'keyword.definition', foreground: 'AF00DB' },
      { token: 'keyword.ref', foreground: '0070C1' },
      { token: 'type.environment', foreground: '267F99', fontStyle: 'bold' },
      { token: 'keyword.command', foreground: '795E26' },
      { token: 'string.math', foreground: 'A31515' },
      { token: 'string.math.delim', foreground: '0000FF', fontStyle: 'bold' },
      { token: 'delimiter.curly', foreground: '0431FA' },
      { token: 'delimiter.bracket', foreground: '319331' },
      { token: 'delimiter.parenthesis', foreground: '7B3814' },
      { token: 'delimiter.special', foreground: '0000FF', fontStyle: 'bold' },
      { token: 'number', foreground: '098658' },
    ],
    colors: {
      'editor.background': '#ffffff',
      'editor.foreground': '#000000',
    }
  });

  monacoEditor = monaco.editor.create(editorHost, {
    value: "% Đang tải mã nguồn...",
    language: "latex",
    theme: "vscode-dark-plus",
    automaticLayout: true,
    fontSize: 14,
    lineNumbers: "on",
    minimap: { enabled: true },
    wordWrap: "on",
    scrollBeyondLastLine: false,
    tabSize: 2,
    renderWhitespace: "selection",
    bracketPairColorization: { enabled: true },
    guides: { bracketPairs: true },
  });

  // Track cursor position
  monacoEditor.onDidChangeCursorPosition((e) => {
    cursorPos.textContent = `Dòng ${e.position.lineNumber}, Cột ${e.position.column}`;
  });

  // Track unsaved changes
  monacoEditor.onDidChangeModelContent(() => {
    if (!isDirty) {
      isDirty = true;
      unsavedBadge.classList.remove("hidden");
    }
  });

  // Keyboard shortcuts inside editor
  monacoEditor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
    saveAndCompile();
  });

  monacoEditor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
    compileLatex();
  });

  // Load initial file
  loadFile("main.tex");
  loadFileList();
  loadProjectInfo();
});

// --- File Handling ---
async function loadFileList() {
  try {
    const res = await fetch("/api/files");
    const data = await res.json();
    renderFileTree(data.files || []);
  } catch (err) {
    console.error("Lỗi tải danh sách tệp:", err);
  }
}

function renderFileTree(files) {
  fileTreeContainer.innerHTML = "";
  const showArtifacts = chkShowArtifacts.checked;

  files.forEach((f) => {
    if (f.is_dir) return; // Keep simple flat view of files with path
    if (f.is_build_artifact && !showArtifacts) return;

    const div = document.createElement("div");
    div.className = `file-item ${f.path === currentFilePath ? "active" : ""}`;
    div.title = f.path;

    const extBadge = (f.extension || "").replace(".", "").toUpperCase() || "FILE";

    div.innerHTML = `
      <div class="file-item-left">
        <span class="file-icon-badge">${extBadge}</span>
        <span class="file-item-name">${f.path}</span>
      </div>
      ${f.path !== "main.tex" ? `<button class="file-del-btn" title="Xóa">&times;</button>` : ""}
    `;

    div.querySelector(".file-item-left").addEventListener("click", () => {
      switchFile(f.path);
    });

    const delBtn = div.querySelector(".file-del-btn");
    if (delBtn) {
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteFile(f.path);
      });
    }

    fileTreeContainer.appendChild(div);
  });
}

async function switchFile(path) {
  if (isDirty) {
    const confirmLeave = confirm("Tệp hiện tại có thay đổi chưa lưu. Bạn có muốn lưu trước không?");
    if (confirmLeave) {
      await saveFile();
    }
  }
  loadFile(path);
}

async function loadFile(path) {
  try {
    currentFilePath = path;
    activeFilename.textContent = path;
    isDirty = false;
    unsavedBadge.classList.add("hidden");

    const isImage = /\.(png|jpg|jpeg|gif|webp|svg)$/i.test(path);

    if (isImage) {
      editorHost.classList.add("hidden");
      imageHost.classList.remove("hidden");
      previewImage.src = `/api/file?path=${encodeURIComponent(path)}&t=${Date.now()}`;
      previewCaption.textContent = path;
      return;
    }

    editorHost.classList.remove("hidden");
    imageHost.classList.add("hidden");

    const res = await fetch(`/api/file?path=${encodeURIComponent(path)}`);
    if (!res.ok) throw new Error("Không thể tải file");
    const data = await res.json();

    if (monacoEditor) {
      let lang = "latex";
      if (path.endsWith(".sh")) lang = "shell";
      else if (path.endsWith(".json")) lang = "json";
      else if (path.endsWith(".bib")) lang = "bibtex";
      else if (path.endsWith(".log") || path.endsWith(".txt")) lang = "plaintext";

      const model = monaco.editor.createModel(data.content, lang);
      monacoEditor.setModel(model);
      isDirty = false;
      unsavedBadge.classList.add("hidden");
    }

    loadFileList();
  } catch (err) {
    alert("Lỗi khi mở file: " + err.message);
  }
}

async function saveFile() {
  if (!monacoEditor) return;
  const content = monacoEditor.getValue();
  setStatus("Đang lưu...", "compiling");

  try {
    const res = await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: currentFilePath, content: content }),
    });

    if (!res.ok) throw new Error("Lỗi khi lưu tệp");

    isDirty = false;
    unsavedBadge.classList.add("hidden");
    setStatus("Đã lưu", "ready");
    return true;
  } catch (err) {
    setStatus("Lỗi lưu", "error");
    alert(err.message);
    return false;
  }
}

async function saveAndCompile() {
  const saved = await saveFile();
  if (saved) {
    await compileLatex();
  }
}

// --- Compilation ---
async function compileLatex() {
  if (isCompiling) return;
  isCompiling = true;

  btnCompile.disabled = true;
  compileSpinner.classList.remove("hidden");
  compileText.textContent = "Đang biên dịch...";
  setStatus("Đang biên dịch LaTeX...", "compiling");

  const passes = parseInt(selectPasses.value, 10) || 1;

  try {
    const res = await fetch("/api/compile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        passes: passes,
        engine: "pdflatex",
        main_file: "main.tex"
      }),
    });

    const data = await res.json();

    if (data.success) {
      setStatus(`Thành công (${data.page_count || "?"} trang, ${data.time_taken}s)`, "ready");
      updatePdfPreview(data.pdf_url);
    } else {
      setStatus(`Biên dịch có lỗi (${data.error_count} lỗi)`, "error");
    }

    updateDiagnostics(data);
    rawLogView.textContent = data.raw_output_tail || "Không có log output.";

  } catch (err) {
    setStatus("Lỗi kết nối biên dịch", "error");
    alert("Lỗi biên dịch: " + err.message);
  } finally {
    isCompiling = false;
    btnCompile.disabled = false;
    compileSpinner.classList.add("hidden");
    compileText.textContent = "Biên dịch";
  }
}

window.compileLatex = compileLatex;

function updatePdfPreview(pdfUrl) {
  if (!pdfUrl) {
    pdfEmptyState.classList.remove("hidden");
    pdfFrame.classList.add("hidden");
    return;
  }

  pdfEmptyState.classList.add("hidden");
  pdfFrame.classList.remove("hidden");
  pdfFrame.src = pdfUrl;
}

function updateDiagnostics(data) {
  const errCount = data.error_count || 0;
  const warnCount = data.warning_count || 0;

  diagErrCount.textContent = errCount;
  diagWarnCount.textContent = warnCount;

  if (errCount > 0) {
    errorBadge.textContent = errCount;
    errorBadge.classList.remove("hidden");
  } else {
    errorBadge.classList.add("hidden");
  }

  if (warnCount > 0) {
    warningBadge.textContent = warnCount;
    warningBadge.classList.remove("hidden");
  } else {
    warningBadge.classList.add("hidden");
  }

  diagSummaryText.innerHTML = `Biên dịch <b>${data.passes} pass</b> trong <b>${data.time_taken}s</b>. Số trang: <b>${data.page_count || "N/A"}</b>. Số lỗi: <b style="color:var(--accent-red)">${errCount}</b>, Cảnh báo: <b style="color:var(--accent-yellow)">${warnCount}</b>.`;

  // Render Errors
  diagErrorsList.innerHTML = "";
  if (!data.errors || data.errors.length === 0) {
    diagErrorsList.innerHTML = `<p class="diag-empty">Không phát hiện lỗi cú pháp nghiêm trọng.</p>`;
  } else {
    data.errors.forEach((err) => {
      const item = document.createElement("div");
      item.className = "diag-item";
      item.innerHTML = `
        <div class="diag-item-header">
          <span>${escapeHtml(err.message)}</span>
          ${err.line ? `<span class="diag-line-tag">Dòng ${err.line}</span>` : ""}
        </div>
        ${err.context ? `<pre class="diag-item-context">${escapeHtml(err.context)}</pre>` : ""}
      `;
      if (err.line) {
        item.addEventListener("click", () => {
          jumpToLine(err.line);
        });
      }
      diagErrorsList.appendChild(item);
    });
  }

  // Render Warnings
  diagWarningsList.innerHTML = "";
  if (!data.warnings || data.warnings.length === 0) {
    diagWarningsList.innerHTML = `<p class="diag-empty">Không có cảnh báo.</p>`;
  } else {
    data.warnings.forEach((warn) => {
      const item = document.createElement("div");
      item.className = "diag-warn-item";
      item.textContent = warn;
      diagWarningsList.appendChild(item);
    });
  }
}

function jumpToLine(lineNumber) {
  if (!monacoEditor) return;
  // If not currently in main.tex, switch first
  if (currentFilePath !== "main.tex") {
    switchFile("main.tex").then(() => {
      monacoEditor.revealLineInCenter(lineNumber);
      monacoEditor.setPosition({ lineNumber: lineNumber, column: 1 });
      monacoEditor.focus();
    });
  } else {
    monacoEditor.revealLineInCenter(lineNumber);
    monacoEditor.setPosition({ lineNumber: lineNumber, column: 1 });
    monacoEditor.focus();
  }
}

function setStatus(msg, type) {
  statusText.textContent = msg;
  statusPill.className = `status-pill status-${type}`;
}

async function loadProjectInfo() {
  try {
    const res = await fetch("/api/project");
    const data = await res.json();
    document.getElementById("project-name").textContent = data.project_name;
    if (data.has_pdf) {
      updatePdfPreview(`/api/pdf?t=${Date.now()}`);
    } else {
      updatePdfPreview(null);
    }
  } catch (err) {
    console.error("Lỗi tải thông tin dự án:", err);
  }
}

// --- Preview Tabs ---
document.querySelectorAll(".preview-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".preview-tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-body").forEach((b) => b.classList.remove("active"));

    tab.classList.add("active");
    const targetId = `tab-content-${tab.dataset.tab}`;
    const targetBody = document.getElementById(targetId);
    if (targetBody) targetBody.classList.add("active");
  });
});

// Full Log Button
btnFetchFullLog.addEventListener("click", async () => {
  try {
    const res = await fetch("/api/raw-log");
    const data = await res.json();
    rawLogView.textContent = data.content;
  } catch (err) {
    rawLogView.textContent = "Lỗi khi tải log: " + err.message;
  }
});

// --- Modal & File Creation ---
btnNewFile.addEventListener("click", () => {
  newFilenameInput.value = "";
  modalNewFile.classList.remove("hidden");
  newFilenameInput.focus();
});

modalCloseBtn.addEventListener("click", () => modalNewFile.classList.add("hidden"));
modalCancelBtn.addEventListener("click", () => modalNewFile.classList.add("hidden"));

modalCreateBtn.addEventListener("click", async () => {
  const filename = newFilenameInput.value.trim();
  if (!filename) return;

  try {
    const res = await fetch("/api/create-file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: filename, content: "" }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Không thể tạo file");
    }
    modalNewFile.classList.add("hidden");
    await loadFileList();
    switchFile(filename);
  } catch (err) {
    alert(err.message);
  }
});

async function deleteFile(path) {
  if (!confirm(`Bạn có chắc chắn muốn xóa file "${path}" không?`)) return;
  try {
    const res = await fetch("/api/delete-file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: path }),
    });
    if (!res.ok) throw new Error("Không thể xóa file");
    await loadFileList();
    if (currentFilePath === path) {
      switchFile("main.tex");
    }
  } catch (err) {
    alert(err.message);
  }
}

// Upload Figure
btnUploadFile.addEventListener("click", () => uploadInput.click());

uploadInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("target_dir", "figures");
  formData.append("file", file);

  try {
    setStatus("Đang tải ảnh...", "compiling");
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Tải tệp thất bại");
    const data = await res.json();
    setStatus("Đã tải ảnh lên figures/", "ready");
    await loadFileList();
    alert(`Đã tải tệp lên: ${data.path}`);
  } catch (err) {
    alert(err.message);
    setStatus("Lỗi upload", "error");
  } finally {
    uploadInput.value = "";
  }
});

// --- Resizer / Splitter Logic ---
let isResizing = false;

resizer.addEventListener("mousedown", (e) => {
  isResizing = true;
  resizer.classList.add("resizing");
  document.body.style.cursor = "col-resize";
  e.preventDefault();
});

document.addEventListener("mousemove", (e) => {
  if (!isResizing) return;
  const sidebarWidth = sidebar.classList.contains("collapsed") ? 0 : sidebar.offsetWidth;
  const availableWidth = window.innerWidth - sidebarWidth - 6; // minus resizer width
  const offsetLeft = e.clientX - sidebarWidth;
  const leftPercent = Math.max(20, Math.min(80, (offsetLeft / availableWidth) * 100));
  const rightPercent = 100 - leftPercent;

  editorPane.style.flex = `${leftPercent} 1 0%`;
  previewPane.style.flex = `${rightPercent} 1 0%`;
});

document.addEventListener("mouseup", () => {
  if (isResizing) {
    isResizing = false;
    resizer.classList.remove("resizing");
    document.body.style.cursor = "";
  }
});

// --- General UI Actions ---
btnSave.addEventListener("click", () => saveFile());
btnCompile.addEventListener("click", () => compileLatex());
btnRefreshPdf.addEventListener("click", () => {
  pdfFrame.src = `/api/pdf?t=${Date.now()}`;
});
btnRefreshFiles.addEventListener("click", () => loadFileList());
chkShowArtifacts.addEventListener("change", () => loadFileList());

btnDownloadPdf.addEventListener("click", () => {
  window.open("/api/download-pdf", "_blank");
});

btnOpenPdfTab.addEventListener("click", () => {
  window.open("/api/pdf", "_blank");
});

btnToggleSidebar.addEventListener("click", () => {
  sidebar.classList.toggle("collapsed");
});

// Theme Toggle
btnThemeToggle.addEventListener("click", () => {
  const isDark = document.body.classList.contains("theme-dark");
  if (isDark) {
    document.body.classList.remove("theme-dark");
    document.body.classList.add("theme-light");
    if (monacoEditor) monaco.editor.setTheme("vscode-light-plus");
  } else {
    document.body.classList.remove("theme-light");
    document.body.classList.add("theme-dark");
    if (monacoEditor) monaco.editor.setTheme("vscode-dark-plus");
  }
});

// Global Shortcuts
window.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "s") {
    e.preventDefault();
    saveAndCompile();
  } else if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    compileLatex();
  } else if ((e.ctrlKey || e.metaKey) && e.key === "b") {
    e.preventDefault();
    sidebar.classList.toggle("collapsed");
  } else if (e.key === "Escape") {
    modalNewFile.classList.add("hidden");
  }
});

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
