/* ═══════════════════════════════════════════════════════════════
   Deep Learning Inference Dashboard — Client-Side Logic
   Handles tab navigation, file uploads, API calls, result rendering.
   ═══════════════════════════════════════════════════════════════ */

;(function () {
    "use strict";

    // ── API Base URL ─────────────────────────────────────────────
    // Local dev  : same-origin (FastAPI serves both frontend & backend)
    // GitHub Pages: calls the deployed Render backend
    //
    // After deploying on Render, replace the URL below with your app URL,
    // e.g. "https://dl-inference-xxxx.onrender.com"
    const BACKEND_URL = "https://YOUR-APP.onrender.com";

    const isLocal = window.location.hostname === "localhost"
                 || window.location.hostname === "127.0.0.1";
    const API_BASE = isLocal ? "" : BACKEND_URL;

    // ── Helpers ─────────────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    function showToast(msg, type = "error") {
        const container = $("#toast-container");
        const el = document.createElement("div");
        el.className = `toast ${type}`;
        el.textContent = msg;
        container.appendChild(el);
        setTimeout(() => el.remove(), 3000);
    }

    function show(el)  { el.classList.remove("hidden"); }
    function hide(el)  { el.classList.add("hidden"); }

    // ── Tab Navigation ──────────────────────────────────────────
    const navBtns   = $$(".nav-btn");
    const tabPanels = $$(".tab-panel");
    const pageTitle = $("#page-title");

    const TAB_TITLES = {
        image:      "Image Classification",
        text:       "Text Classification",
        multimodal: "Multimodal Classification",
    };

    function switchTab(tabId) {
        navBtns.forEach(b => b.classList.toggle("active", b.dataset.tab === tabId));
        tabPanels.forEach(p => p.classList.toggle("active", p.id === `tab-${tabId}`));
        pageTitle.textContent = TAB_TITLES[tabId] || tabId;
    }

    navBtns.forEach(btn => {
        btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });

    // Mobile menu toggle
    const menuToggle = $("#menu-toggle");
    const sidebar = $("#sidebar");
    if (menuToggle) {
        menuToggle.addEventListener("click", () => sidebar.classList.toggle("open"));
        navBtns.forEach(btn => btn.addEventListener("click", () => sidebar.classList.remove("open")));
    }

    // ── Server Health Check ─────────────────────────────────────
    async function checkHealth() {
        const dot  = $(".status-dot");
        const text = $(".status-text");
        try {
            const res = await fetch(`${API_BASE}/api/health`);
            if (res.ok) {
                dot.classList.add("online");
                dot.classList.remove("offline");
                text.textContent = "Server online";
            } else {
                throw new Error("unhealthy");
            }
        } catch {
            dot.classList.add("offline");
            dot.classList.remove("online");
            text.textContent = "Server offline";
        }
    }
    checkHealth();
    setInterval(checkHealth, 15000);

    // ── Generic File Upload Setup ───────────────────────────────
    function setupUpload(dropZoneId, fileInputId, previewId, placeholderId, clearBtnId) {
        const zone        = $(`#${dropZoneId}`);
        const fileInput   = $(`#${fileInputId}`);
        const preview     = $(`#${previewId}`);
        const placeholder = $(`#${placeholderId}`);
        const clearBtn    = $(`#${clearBtnId}`);
        let currentFile   = null;

        zone.addEventListener("click", (e) => {
            if (e.target === clearBtn) return;
            fileInput.click();
        });

        ["dragenter", "dragover"].forEach(evt => {
            zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add("drag-over"); });
        });
        ["dragleave", "drop"].forEach(evt => {
            zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.remove("drag-over"); });
        });
        zone.addEventListener("drop", (e) => {
            const files = e.dataTransfer.files;
            if (files.length) handleFile(files[0]);
        });

        fileInput.addEventListener("change", () => {
            if (fileInput.files.length) handleFile(fileInput.files[0]);
        });

        clearBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            currentFile = null;
            fileInput.value = "";
            hide(preview);
            hide(clearBtn);
            show(placeholder);
        });

        function handleFile(file) {
            if (!file.type.startsWith("image/")) {
                showToast("Please upload an image file (PNG, JPG, JPEG).", "warning");
                return;
            }
            currentFile = file;
            const reader = new FileReader();
            reader.onload = (ev) => {
                preview.src = ev.target.result;
                show(preview);
                show(clearBtn);
                hide(placeholder);
            };
            reader.readAsDataURL(file);
        }

        return () => currentFile;
    }

    // ── Render Results ──────────────────────────────────────────
    function renderResults(data, prefix) {
        const preds = data.predictions;
        const top   = preds[0];

        $(`#${prefix}-top-prediction`).innerHTML = `
            <div class="top-label">Top Prediction</div>
            <div class="top-class">${top.label}</div>
            <div class="top-confidence">Confidence: <strong>${(top.confidence * 100).toFixed(1)}%</strong></div>
        `;

        const maxConf = top.confidence;
        let chartHTML = "";
        preds.forEach((p, i) => {
            const pct = (p.confidence * 100).toFixed(1);
            const w   = maxConf > 0 ? (p.confidence / maxConf * 100) : 0;
            chartHTML += `
                <div class="chart-bar-row">
                    <span class="chart-label" title="${p.label}">${p.label}</span>
                    <div class="chart-bar-track">
                        <div class="chart-bar-fill ${i === 0 ? 'top' : 'other'}" style="width: 0%;" data-width="${w}%"></div>
                    </div>
                    <span class="chart-value">${pct}%</span>
                </div>
            `;
        });
        $(`#${prefix}-chart`).innerHTML = chartHTML;

        requestAnimationFrame(() => {
            $$(`#${prefix}-chart .chart-bar-fill`).forEach(bar => {
                bar.style.width = bar.dataset.width;
            });
        });

        const tbody = $(`#${prefix}-table tbody`);
        tbody.innerHTML = preds.map((p, i) => `
            <tr>
                <td>${i + 1}</td>
                <td>${p.label}</td>
                <td>${(p.confidence * 100).toFixed(2)}%</td>
                <td>
                    <div class="conf-bar-bg">
                        <div class="conf-bar" style="width: ${(p.confidence * 100).toFixed(1)}%"></div>
                    </div>
                </td>
            </tr>
        `).join("");

        const latencyEl = $(`#${prefix}-latency`);
        latencyEl.textContent = `⏱ ${data.inference_time_ms} ms`;
        show(latencyEl);
    }

    function setTabState(prefix, state) {
        hide($(`#${prefix}-empty-state`));
        hide($(`#${prefix}-results`));
        hide($(`#${prefix}-loading`));
        if (state === "empty")   show($(`#${prefix}-empty-state`));
        if (state === "loading") show($(`#${prefix}-loading`));
        if (state === "results") show($(`#${prefix}-results`));
    }

    // ══════════════════════════════════════════════════════════════
    // IMAGE TAB
    // ══════════════════════════════════════════════════════════════
    const getImageFile = setupUpload(
        "image-drop-zone", "image-file-input", "image-preview",
        "image-upload-placeholder", "image-clear-btn"
    );

    $("#image-predict-btn").addEventListener("click", async () => {
        const file = getImageFile();
        if (!file) { showToast("Please upload an image first.", "warning"); return; }

        const formData = new FormData();
        formData.append("file", file);
        formData.append("model", $("#image-model-select").value);

        setTabState("image", "loading");
        try {
            const res = await fetch(`${API_BASE}/api/predict/image`, { method: "POST", body: formData });
            if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Server error"); }
            const data = await res.json();
            renderResults(data, "image");
            setTabState("image", "results");
            showToast(`Predicted using ${data.model}`, "success");
        } catch (err) {
            setTabState("image", "empty");
            showToast(err.message, "error");
        }
    });

    // ══════════════════════════════════════════════════════════════
    // TEXT TAB
    // ══════════════════════════════════════════════════════════════
    const textInput = $("#text-input");
    const charCount = $("#text-char-count");
    textInput.addEventListener("input", () => { charCount.textContent = textInput.value.length; });

    $("#text-predict-btn").addEventListener("click", async () => {
        const text = textInput.value.trim();
        if (!text) { showToast("Please enter text to classify.", "warning"); return; }

        const formData = new FormData();
        formData.append("text", text);
        formData.append("model", $("#text-model-select").value);

        setTabState("text", "loading");
        try {
            const res = await fetch(`${API_BASE}/api/predict/text`, { method: "POST", body: formData });
            if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Server error"); }
            const data = await res.json();
            renderResults(data, "text");
            setTabState("text", "results");
            showToast(`Predicted using ${data.model}`, "success");
        } catch (err) {
            setTabState("text", "empty");
            showToast(err.message, "error");
        }
    });

    // ══════════════════════════════════════════════════════════════
    // MULTIMODAL TAB
    // ══════════════════════════════════════════════════════════════
    const getMultiFile = setupUpload(
        "multi-drop-zone", "multi-file-input", "multi-preview",
        "multi-upload-placeholder", "multi-clear-btn"
    );

    $("#multi-predict-btn").addEventListener("click", async () => {
        const file = getMultiFile();
        const text = $("#multi-text-input").value.trim();
        if (!file) { showToast("Please upload an image.", "warning"); return; }
        if (!text) { showToast("Please enter a text query.", "warning"); return; }

        const formData = new FormData();
        formData.append("file", file);
        formData.append("text", text);
        formData.append("model", $("#multi-model-select").value);

        setTabState("multi", "loading");
        try {
            const res = await fetch(`${API_BASE}/api/predict/multimodal`, { method: "POST", body: formData });
            if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Server error"); }
            const data = await res.json();
            renderResults(data, "multi");
            setTabState("multi", "results");
            showToast(`Predicted using ${data.model} (${data.method})`, "success");
        } catch (err) {
            setTabState("multi", "empty");
            showToast(err.message, "error");
        }
    });

    // ── Ctrl+Enter to predict ────────────────────────────────────
    document.addEventListener("keydown", (e) => {
        if (e.ctrlKey && e.key === "Enter") {
            const btn = document.querySelector(".tab-panel.active .btn-primary");
            if (btn) btn.click();
        }
    });

})();
