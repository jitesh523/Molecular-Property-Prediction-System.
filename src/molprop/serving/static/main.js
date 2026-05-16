document.addEventListener("DOMContentLoaded", () => {
    // ── Elements: Navigation ──────────────────────────────────────────────────
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanels = document.querySelectorAll(".tab-panel");

    // ── Elements: Predict Tab ────────────────────────────────────────────────
    const predictBtn = document.getElementById("predict-btn");
    const smilesInput = document.getElementById("smiles-input");
    const explainCb = document.getElementById("explain-cb");
    const uncertaintyInput = document.getElementById("uncertainty-input");
    const resultsPanel = document.getElementById("results-panel");
    const predictLoading = document.getElementById("predict-loading");
    const predValue = document.getElementById("pred-value");
    const predUncertainty = document.getElementById("pred-uncertainty");
    const stdSmilesDisplay = document.getElementById("std-smiles-display");
    const svgContainer = document.getElementById("svg-container");
    const explanationPanel = document.getElementById("explanation-panel");
    const taskLabel = document.getElementById("task-label");
    const similarityPanel = document.getElementById("similarity-panel");
    const analogsContainer = document.getElementById("analogs-container");
    const exampleSelect = document.getElementById("example-select");
    const copySmilesBtn = document.getElementById("copy-smiles-btn");
    const latencyDisplay = document.getElementById("latency-display");

    // ── Elements: Generate Tab ───────────────────────────────────────────────
    const generateBtn = document.getElementById("generate-btn");
    const genNInput = document.getElementById("gen-n");
    const genTempInput = document.getElementById("gen-temperature");
    const genSeedInput = document.getElementById("gen-seed");
    const genLoading = document.getElementById("gen-loading");
    const genResultsPanel = document.getElementById("gen-results-panel");
    const genGrid = document.getElementById("gen-grid");
    const genStats = document.getElementById("gen-stats");
    const vaeStatusBanner = document.getElementById("vae-status-banner");

    // ── Elements: Optimize Tab ──────────────────────────────────────────────
    const optimizeBtn = document.getElementById("optimize-btn");
    const optMethod = document.getElementById("opt-method");
    const optCandidates = document.getElementById("opt-candidates");
    const optTempInput = document.getElementById("opt-temperature");
    const optLoading = document.getElementById("optimize-loading");
    const optResultsPanel = document.getElementById("optimize-results-panel");
    const optGrid = document.getElementById("optimize-grid");
    const optStats = document.getElementById("optimize-stats");
    const optStatusBanner = document.getElementById("optimize-status-banner");
    const optMwMin = document.getElementById("opt-mw-min");
    const optMwMax = document.getElementById("opt-mw-max");
    const optLogpMin = document.getElementById("opt-logp-min");
    const optLogpMax = document.getElementById("opt-logp-max");
    const optTpsaMin = document.getElementById("opt-tpsa-min");
    const optTpsaMax = document.getElementById("opt-tpsa-max");
    const optHbdMin = document.getElementById("opt-hbd-min");
    const optHbdMax = document.getElementById("opt-hbd-max");
    const optHbaMin = document.getElementById("opt-hba-min");
    const optHbaMax = document.getElementById("opt-hba-max");
    const optQedMin = document.getElementById("opt-qed-min");
    const optQedMax = document.getElementById("opt-qed-max");
    const optSasMin = document.getElementById("opt-sas-min");
    const optSasMax = document.getElementById("opt-sas-max");
    const optSeed = document.getElementById("opt-seed");
    const optExportCsv = document.getElementById("opt-export-csv");
    const paretoBtn = document.getElementById("pareto-btn");
    const paretoLoading = document.getElementById("pareto-loading");
    const paretoResultsPanel = document.getElementById("pareto-results-panel");
    const paretoGrid = document.getElementById("pareto-grid");
    const paretoStats = document.getElementById("pareto-stats");
    const paretoExportCsv = document.getElementById("pareto-export-csv");
    const paretoSamples = document.getElementById("pareto-samples");
    const findAnalogsBtn = document.getElementById("find-analogs-btn");

    // ── Elements: History Tab ────────────────────────────────────────────────
    const historyList = document.getElementById("history-list");
    const historyCount = document.getElementById("history-count");
    const historyCompareBtn = document.getElementById("history-compare-btn");
    const historyExportBtn = document.getElementById("history-export-btn");
    const historyClearBtn = document.getElementById("history-clear-btn");
    const historyComparePanel = document.getElementById("history-compare-panel");
    const historyCompareTable = document.getElementById("history-compare-table");

    // Store last optimization results for CSV export
    let lastOptimizeResults = [];
    let lastParetoResults = [];
    let lastPredictedSmiles = "";

    // ── History: localStorage helpers ────────────────────────────────────────
    const HISTORY_KEY = "molprop_history";
    const MAX_HISTORY = 50;

    function loadHistory() {
        try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); }
        catch { return []; }
    }

    function saveHistory(h) {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(h.slice(0, MAX_HISTORY)));
    }

    function addToHistory(entry) {
        const h = loadHistory();
        h.unshift({ ...entry, id: Date.now(), timestamp: new Date().toLocaleString() });
        saveHistory(h);
        if (document.getElementById("tab-history")?.classList.contains("active")) {
            renderHistory();
        }
        if (historyCount) historyCount.textContent = `${h.length} molecule${h.length !== 1 ? 's' : ''} saved`;
    }

    function renderHistory() {
        const h = loadHistory();
        if (!historyList) return;
        historyList.innerHTML = '';
        if (!h.length) {
            historyList.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:2rem;">No predictions yet. Predict a molecule to see history here.</p>';
            if (historyCount) historyCount.textContent = "0 molecules";
            if (historyCompareBtn) historyCompareBtn.disabled = true;
            return;
        }
        if (historyCount) historyCount.textContent = `${h.length} molecule${h.length !== 1 ? 's' : ''} saved`;
        h.forEach(entry => {
            const row = document.createElement("div");
            row.className = "history-row";
            row.dataset.id = entry.id;

            const propChips = Object.entries(entry.properties || {})
                .map(([k, v]) => `<span class="history-prop-chip">${k}: ${typeof v === 'number' ? v.toFixed(2) : v}</span>`)
                .join('');

            row.innerHTML = `
                <input type="checkbox" class="history-checkbox" data-id="${entry.id}" style="width:1.1rem;height:1.1rem;accent-color:var(--primary);">
                <div style="flex:1;min-width:0;">
                    <div class="history-smiles">${entry.smiles}</div>
                    <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.2rem;">${entry.timestamp}</div>
                </div>
                <div class="history-pred">${typeof entry.prediction === 'number' ? entry.prediction.toFixed(4) : entry.prediction ?? '—'}</div>
                <div class="history-props">${propChips}</div>
                <div style="display:flex;gap:0.5rem;">
                    <button class="btn-secondary" style="padding:0.3rem 0.6rem;font-size:0.75rem;"
                            onclick="window.historyPredict('${entry.smiles}')">Predict</button>
                    <button class="btn-secondary" style="padding:0.3rem 0.6rem;font-size:0.75rem;"
                            onclick="window.historyOptimize('${entry.smiles}')">Optimize</button>
                </div>
            `;
            historyList.appendChild(row);
        });

        // Update compare button based on checkbox state
        historyList.querySelectorAll(".history-checkbox").forEach(cb => {
            cb.addEventListener("change", () => {
                const checked = historyList.querySelectorAll(".history-checkbox:checked").length;
                if (historyCompareBtn) historyCompareBtn.disabled = checked < 2;
            });
        });
    }

    // Bridge: history -> predict tab
    window.historyPredict = (smiles) => {
        smilesInput.value = smiles;
        document.querySelector('[data-tab="predict"]').click();
        predictBtn.click();
    };

    // Bridge: history -> optimize tab
    window.historyOptimize = (smiles) => {
        if (optSeed) optSeed.value = smiles;
        document.querySelector('[data-tab="optimize"]').click();
        if (optSeed) optSeed.scrollIntoView({ behavior: "smooth", block: "center" });
    };

    // Render on history tab click
    document.querySelector('[data-tab="history"]')?.addEventListener("click", renderHistory);

    // ── Elements: Visualize Tab ──────────────────────────────────────────────
    const vizBtn = document.getElementById("viz-btn");
    const vizCanvas = document.getElementById("viz-canvas");
    const vizResults = document.getElementById("viz-results");
    const vizStats = document.getElementById("viz-stats");
    const vizTooltip = document.getElementById("viz-tooltip");
    const vizSelected = document.getElementById("viz-selected");
    const vizSelectedSmiles = document.getElementById("viz-selected-smiles");
    const vizSelectedProps = document.getElementById("viz-selected-props");
    const vizPredictBtn = document.getElementById("viz-predict-btn");
    const vizOptimizeBtn = document.getElementById("viz-optimize-btn");
    const vizSamplesInput = document.getElementById("viz-samples");
    const vizTempInput = document.getElementById("viz-temperature");
    const vizSeedInput = document.getElementById("viz-seed");

    let vizPoints = [];  // raw points from API
    let vizSelectedPoint = null;

    // Helper: map QED 0→1 to a color (purple → green)
    function qedColor(qed) {
        const r = Math.round(99 + (16 - 99) * qed);
        const g = Math.round(102 + (185 - 102) * qed);
        const b = Math.round(241 + (129 - 241) * qed);
        return `rgb(${r},${g},${b})`;
    }

    // Helper: project points to canvas pixel coords
    function projectPoints(points, canvas) {
        if (!points.length) return [];
        const xs = points.map(p => p.x);
        const ys = points.map(p => p.y);
        const xMin = Math.min(...xs), xMax = Math.max(...xs);
        const yMin = Math.min(...ys), yMax = Math.max(...ys);
        const pad = 40;
        const W = canvas.width, H = canvas.height;
        return points.map(p => ({
            ...p,
            px: pad + (p.x - xMin) / (xMax - xMin + 1e-9) * (W - 2 * pad),
            py: H - pad - (p.y - yMin) / (yMax - yMin + 1e-9) * (H - 2 * pad),
        }));
    }

    function drawVizCanvas(projected) {
        if (!vizCanvas) return;
        const ctx = vizCanvas.getContext("2d");
        ctx.clearRect(0, 0, vizCanvas.width, vizCanvas.height);

        // Draw regular points first, seeds last
        const regular = projected.filter(p => !p.is_seed);
        const seeds = projected.filter(p => p.is_seed);

        for (const p of regular) {
            ctx.beginPath();
            ctx.arc(p.px, p.py, 5, 0, Math.PI * 2);
            ctx.fillStyle = qedColor(p.qed);
            ctx.globalAlpha = 0.75;
            ctx.fill();
        }

        // Highlight selected
        if (vizSelectedPoint) {
            ctx.beginPath();
            ctx.arc(vizSelectedPoint.px, vizSelectedPoint.py, 8, 0, Math.PI * 2);
            ctx.fillStyle = "#fbbf24";
            ctx.globalAlpha = 1;
            ctx.fill();
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Draw seed stars
        for (const p of seeds) {
            ctx.globalAlpha = 1;
            ctx.font = "18px serif";
            ctx.fillText("⭐", p.px - 9, p.py + 7);
        }

        ctx.globalAlpha = 1;
    }

    function findNearestPoint(projected, mx, my, threshold = 12) {
        let best = null, bestDist = Infinity;
        for (const p of projected) {
            const d = Math.hypot(p.px - mx, p.py - my);
            if (d < threshold && d < bestDist) { best = p; bestDist = d; }
        }
        return best;
    }

    // ── Action: Generate Map ─────────────────────────────────────────────────
    if (vizBtn) {
        vizBtn.addEventListener("click", async () => {
            vizBtn.disabled = true;
            vizBtn.textContent = "⏳ Computing…";
            vizResults.classList.add("hidden");

            try {
                const resp = await fetch("/latent_map", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        n_samples: parseInt(vizSamplesInput?.value || 300),
                        temperature: parseFloat(vizTempInput?.value || 0.8),
                        seed_smiles: vizSeedInput?.value.trim() || null
                    })
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || "Latent map failed");

                vizPoints = projectPoints(data.points, vizCanvas);
                drawVizCanvas(vizPoints);
                vizStats.textContent = `${data.total} molecules • PCA projection • colored by QED`;
                vizResults.classList.remove("hidden");
            } catch (err) {
                alert(err.message);
            } finally {
                vizBtn.disabled = false;
                vizBtn.textContent = "🌐 Generate Map";
            }
        });
    }

    // ── Visualize: hover tooltip ─────────────────────────────────────────────
    if (vizCanvas) {
        vizCanvas.addEventListener("mousemove", (e) => {
            const rect = vizCanvas.getBoundingClientRect();
            const scaleX = vizCanvas.width / rect.width;
            const scaleY = vizCanvas.height / rect.height;
            const mx = (e.clientX - rect.left) * scaleX;
            const my = (e.clientY - rect.top) * scaleY;

            const nearest = findNearestPoint(vizPoints, mx, my, 14);
            if (nearest) {
                vizTooltip.classList.remove("hidden");
                vizTooltip.style.left = (e.clientX + 14) + "px";
                vizTooltip.style.top = (e.clientY - 10) + "px";
                vizTooltip.innerHTML = `
                    <div style="font-weight:600;color:#e2e8f0;margin-bottom:0.3rem;">QED: ${nearest.qed.toFixed(3)}</div>
                    <div style="color:#94a3b8;font-size:0.75rem;">MW: ${nearest.mw.toFixed(1)}</div>
                    <div style="font-family:monospace;font-size:0.72rem;color:#cbd5e1;margin-top:0.3rem;word-break:break-all;">${nearest.smiles}</div>
                `;
                vizCanvas.style.cursor = "pointer";
            } else {
                vizTooltip.classList.add("hidden");
                vizCanvas.style.cursor = "crosshair";
            }
        });

        vizCanvas.addEventListener("mouseleave", () => {
            vizTooltip.classList.add("hidden");
        });

        // ── Visualize: click to select ────────────────────────────────────────
        vizCanvas.addEventListener("click", (e) => {
            const rect = vizCanvas.getBoundingClientRect();
            const scaleX = vizCanvas.width / rect.width;
            const scaleY = vizCanvas.height / rect.height;
            const mx = (e.clientX - rect.left) * scaleX;
            const my = (e.clientY - rect.top) * scaleY;

            const nearest = findNearestPoint(vizPoints, mx, my, 14);
            if (nearest) {
                vizSelectedPoint = nearest;
                drawVizCanvas(vizPoints);
                vizSelectedSmiles.textContent = nearest.smiles;
                vizSelectedProps.innerHTML = `
                    <span class="history-prop-chip">QED: ${nearest.qed.toFixed(3)}</span>
                    <span class="history-prop-chip">MW: ${nearest.mw.toFixed(1)}</span>
                `;
                vizSelected.classList.remove("hidden");
            }
        });
    }

    // ── Visualize: selected mol actions ────────────────────────────────────
    if (vizPredictBtn) {
        vizPredictBtn.addEventListener("click", () => {
            if (vizSelectedPoint) {
                smilesInput.value = vizSelectedPoint.smiles;
                document.querySelector('[data-tab="predict"]').click();
                predictBtn.click();
            }
        });
    }

    if (vizOptimizeBtn) {
        vizOptimizeBtn.addEventListener("click", () => {
            if (vizSelectedPoint) {
                if (optSeed) optSeed.value = vizSelectedPoint.smiles;
                document.querySelector('[data-tab="optimize"]').click();
                if (optSeed) optSeed.scrollIntoView({ behavior: "smooth", block: "center" });
            }
        });
    }

    // Enable viz button when VAE is loaded
    document.addEventListener("vae-ready", () => {
        if (vizBtn) vizBtn.disabled = false;
    });

    // ── Elements: Global ─────────────────────────────────────────────────────
    const appTitle = document.getElementById("app-title");
    const badgeModel = document.getElementById("badge-model");
    const badgeVae = document.getElementById("badge-vae");
    const apiBadges = document.getElementById("api-badges");

    // ── Tab Switching Logic ─────────────────────────────────────────────────
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const tabId = btn.getAttribute("data-tab");
            
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanels.forEach(p => p.classList.remove("active", "hidden"));
            tabPanels.forEach(p => {
                if(p.id !== `tab-${tabId}`) p.classList.add("hidden");
            });

            btn.classList.add("active");
            document.getElementById(`tab-${tabId}`).classList.add("active");
        });
    });

    // ── Initialization: Model & VAE Status ──────────────────────────────────
    async function initSystem() {
        try {
            // Check GNN Model
            const mResp = await fetch("/model/info");
            const mData = await mResp.json();
            if (mData.status === "loaded") {
                appTitle.textContent = `Environment: ${mData.dataset.toUpperCase()} Workspace`;
                badgeModel.textContent = `${mData.model_type.toUpperCase()} Active`;
                taskLabel.textContent = mData.task === "regression" ? "Predicted Value" : "Probability (Active)";
                predictBtn.disabled = false;
                document.dispatchEvent(new Event("model-ready"));
            } else {
                badgeModel.textContent = "GNN Not Loaded";
                badgeModel.className = "badge badge-red";
                predictBtn.disabled = true;
            }

            // Check VAE Status
            const vResp = await fetch("/generate/status");
            const vData = await vResp.json();
            if (vData.vae_loaded) {
                badgeVae.textContent = `VAE Ready (z=${vData.latent_dim})`;
                vaeStatusBanner.textContent = `✅ VAE Decoder Active: Latent dimension ${vData.latent_dim}, Vocab size ${vData.vocab_size}`;
                vaeStatusBanner.className = "status-banner status-ready";
                generateBtn.disabled = false;
                if (optStatusBanner) {
                    optStatusBanner.textContent = `✅ VAE Optimizer Ready: Latent dimension ${vData.latent_dim}`;
                    optStatusBanner.className = "status-banner status-ready";
                }
                if (optimizeBtn) optimizeBtn.disabled = false;
                if (paretoBtn) paretoBtn.disabled = false;
                if (vizBtn) vizBtn.disabled = false;
                document.dispatchEvent(new Event("vae-ready"));
            } else {
                badgeVae.textContent = "VAE Inactive";
                badgeVae.style.opacity = "0.5";
                vaeStatusBanner.textContent = "⚠️ VAE Checkpoint not found on server. Start training with scripts/train_vae.py";
                if (optStatusBanner) {
                    optStatusBanner.textContent = "⚠️ VAE not loaded. Optimization requires a trained VAE model.";
                }
            }
            apiBadges.classList.remove("hidden");
        } catch (err) {
            console.error("Connectivity error:", err);
            appTitle.textContent = "⚠️ Disconnected from API";
            appTitle.style.color = "#ef4444";
        }
    }

    initSystem();

    // ── Action: Predict ─────────────────────────────────────────────────────
    predictBtn.addEventListener("click", async () => {
        const smiles = smilesInput.value.trim();
        if (!smiles) return;

        // Reset UI
        resultsPanel.classList.add("hidden");
        explanationPanel.classList.add("hidden");
        similarityPanel.classList.add("hidden");
        predictLoading.classList.remove("hidden");
        predictBtn.disabled = true;

        try {
            const resp = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    smiles: smiles,
                    explain: explainCb.checked,
                    uncertainty_samples: parseInt(uncertaintyInput.value) || 0
                })
            });

            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || "Inference failed");

            // Server-reported latency from middleware
            const timing = resp.headers.get("X-Process-Time");
            if (timing && latencyDisplay) latencyDisplay.textContent = timing;

            // Populate Metrics
            const val = data.predictions.task_1;
            predValue.textContent = Number.isInteger(val) ? val : val.toFixed(4);
            stdSmilesDisplay.textContent = data.standardized_smiles || "-";
            predUncertainty.textContent = data.uncertainty_std ? `± ${data.uncertainty_std.task_1.toFixed(4)}` : "N/A";

            // Track SMILES and show Find Analogs button
            lastPredictedSmiles = data.standardized_smiles || smilesInput.value;
            if (findAnalogsBtn) findAnalogsBtn.style.display = "inline-block";

            // Save to history
            addToHistory({
                smiles: data.standardized_smiles || smilesInput.value,
                prediction: data.predictions?.task_1,
                uncertainty: data.uncertainty_std?.task_1,
                properties: {}
            });

            // Explanations
            if (data.explanation && data.explanation.svg) {
                svgContainer.innerHTML = data.explanation.svg;
                explanationPanel.classList.remove("hidden");
            }

            // Similarity
            if (data.similar_molecules && data.similar_molecules.length > 0) {
                analogsContainer.innerHTML = '';
                data.similar_molecules.forEach(mol => {
                    const item = document.createElement("div");
                    item.className = "analog-item slide-up";
                    item.innerHTML = `
                        <div class="analog-score">${(mol.score * 100).toFixed(1)}% Similar</div>
                        <div class="analog-smiles">${mol.smiles}</div>
                    `;
                    analogsContainer.appendChild(item);
                });
                similarityPanel.classList.remove("hidden");
            }

            resultsPanel.classList.remove("hidden");
        } catch (err) {
            alert(err.message);
        } finally {
            predictLoading.classList.add("hidden");
            predictBtn.disabled = false;
        }
    });

    // ── Action: Generate ────────────────────────────────────────────────────
    generateBtn.addEventListener("click", async () => {
        // UI Reset
        genGrid.innerHTML = '';
        genResultsPanel.classList.add("hidden");
        genLoading.classList.remove("hidden");
        generateBtn.disabled = true;

        try {
            const resp = await fetch("/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    n: parseInt(genNInput.value),
                    temperature: parseFloat(genTempInput.value),
                    seed: genSeedInput.value ? parseInt(genSeedInput.value) : null
                })
            });

            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || "Generation failed");

            let validCount = 0;
            data.forEach(mol => {
                const card = document.createElement("div");
                card.className = "gen-mol-card slide-up";
                const badgeClass = mol.valid ? "gen-badge-valid" : "gen-badge-invalid";
                const badgeText = mol.valid ? "VALID" : "INVALID";
                if(mol.valid) validCount++;

                card.innerHTML = `
                    <span class="gen-badge ${badgeClass}">${badgeText}</span>
                    <div class="analog-smiles" style="font-size: 0.85rem; color: white;">${mol.smiles}</div>
                    ${mol.valid ? `<button class="btn-primary" style="padding: 0.4rem; font-size: 0.7rem; margin-top: auto;" onclick="useGenerated('${mol.smiles}')">Use for Prediction</button>` : ''}
                `;
                genGrid.appendChild(card);
            });

            genStats.textContent = `Validity Rate: ${((validCount / data.length) * 100).toFixed(0)}% (${validCount}/${data.length})`;
            genResultsPanel.classList.remove("hidden");
        } catch (err) {
            alert(err.message);
        } finally {
            genLoading.classList.add("hidden");
            generateBtn.disabled = false;
        }
    });

    // Function to bridge Generate -> Predict tab
    window.useGenerated = (smiles) => {
        smilesInput.value = smiles;
        document.querySelector('[data-tab="predict"]').click();
        predictBtn.click();
    };

    // ── Action: Optimize ────────────────────────────────────────────────────
    if (optimizeBtn) {
        optimizeBtn.addEventListener("click", async () => {
            optGrid.innerHTML = '';
            optResultsPanel.classList.add("hidden");
            optLoading.classList.remove("hidden");
            optimizeBtn.disabled = true;

            const targets = {};
            if (optMwMin && optMwMax) targets.mw = [parseFloat(optMwMin.value), parseFloat(optMwMax.value)];
            if (optLogpMin && optLogpMax) targets.logp = [parseFloat(optLogpMin.value), parseFloat(optLogpMax.value)];
            if (optTpsaMin && optTpsaMax) targets.tpsa = [parseFloat(optTpsaMin.value), parseFloat(optTpsaMax.value)];
            if (optHbdMin && optHbdMax) targets.hbd = [parseInt(optHbdMin.value), parseInt(optHbdMax.value)];
            if (optHbaMin && optHbaMax) targets.hba = [parseInt(optHbaMin.value), parseInt(optHbaMax.value)];
            if (optQedMin && optQedMax) targets.qed = [parseFloat(optQedMin.value), parseFloat(optQedMax.value)];
            if (optSasMin && optSasMax) targets.sas = [parseFloat(optSasMin.value), parseFloat(optSasMax.value)];

            const seedValue = optSeed && optSeed.value.trim() ? optSeed.value.trim() : null;

            try {
                const resp = await fetch("/optimize", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        targets: targets,
                        method: optMethod ? optMethod.value : "gradient_ascent",
                        n_candidates: parseInt(optCandidates ? optCandidates.value : 10),
                        temperature: parseFloat(optTempInput ? optTempInput.value : 0.8),
                        seed_smiles: seedValue
                    })
                });

                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || "Optimization failed");

                // Store results for CSV export
                lastOptimizeResults = data.candidates || [];

                data.candidates.forEach((mol, idx) => {
                    const card = document.createElement("div");
                    card.className = "gen-mol-card slide-up";
                    const props = Object.entries(mol.properties || {})
                        .map(([k, v]) => `<div style="font-size:0.7rem;color:#94a3b8;">${k}: ${typeof v === 'number' ? v.toFixed(2) : v}</div>`)
                        .join('');

                    card.innerHTML = `
                        <span class="gen-badge gen-badge-valid">Score: ${mol.score.toFixed(1)}</span>
                        <div class="analog-smiles" style="font-size: 0.8rem; color: white; margin: 0.5rem 0;">${mol.smiles}</div>
                        <div style="margin-top: auto;">${props}</div>
                        <button class="btn-primary" style="padding: 0.4rem; font-size: 0.7rem; margin-top: 0.5rem;"
                                onclick="useGenerated('${mol.smiles}')">Predict Properties</button>
                    `;
                    optGrid.appendChild(card);
                });

                optStats.textContent = `Valid: ${data.valid_count}/${data.total_attempts} via ${data.method}${seedValue ? ' (seeded)' : ''}`;
                optResultsPanel.classList.remove("hidden");
            } catch (err) {
                alert(err.message);
            } finally {
                optLoading.classList.add("hidden");
                optimizeBtn.disabled = false;
            }
        });
    }

    // ── CSV Export ─────────────────────────────────────────────────────────
    if (optExportCsv) {
        optExportCsv.addEventListener("click", () => {
            if (!lastOptimizeResults.length) {
                alert("No optimization results to export");
                return;
            }

            // Build CSV
            const headers = ["SMILES", "Score", "MW", "LogP", "TPSA", "HBD", "HBA", "QED", "SAS"];
            const rows = lastOptimizeResults.map(mol => [
                mol.smiles,
                mol.score.toFixed(4),
                mol.properties?.mw?.toFixed(2) || "",
                mol.properties?.logp?.toFixed(2) || "",
                mol.properties?.tpsa?.toFixed(2) || "",
                mol.properties?.hbd?.toFixed(0) || "",
                mol.properties?.hba?.toFixed(0) || "",
                mol.properties?.qed?.toFixed(4) || "",
                mol.properties?.sas?.toFixed(2) || ""
            ]);

            const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
            const blob = new Blob([csv], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `optimized_molecules_${new Date().toISOString().slice(0, 10)}.csv`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }

    // Examples dropdown -> populate input
    if (exampleSelect) {
        exampleSelect.addEventListener("change", (e) => {
            const val = e.target.value;
            if (val) {
                smilesInput.value = val;
                smilesInput.focus();
            }
        });
    }

    // Copy standardized SMILES to clipboard
    if (copySmilesBtn) {
        copySmilesBtn.addEventListener("click", async () => {
            const text = stdSmilesDisplay.textContent || "";
            if (!text || text === "—" || text === "-") return;
            try {
                await navigator.clipboard.writeText(text);
                const prev = copySmilesBtn.textContent;
                copySmilesBtn.textContent = "Copied ✓";
                setTimeout(() => { copySmilesBtn.textContent = prev; }, 1200);
            } catch (err) {
                console.error("Clipboard write failed:", err);
            }
        });
    }

    // ── History: Compare ────────────────────────────────────────────────────
    if (historyCompareBtn) {
        historyCompareBtn.addEventListener("click", () => {
            const checked = [...historyList.querySelectorAll(".history-checkbox:checked")];
            const ids = checked.map(cb => parseInt(cb.dataset.id));
            const h = loadHistory();
            const selected = h.filter(e => ids.includes(e.id));
            if (selected.length < 2) return;

            const propKeys = [...new Set(selected.flatMap(e => Object.keys(e.properties || {})))];
            const rows = [
                ["Property", ...selected.map(e => e.smiles.length > 20 ? e.smiles.slice(0, 20) + "…" : e.smiles)],
                ["Prediction", ...selected.map(e => typeof e.prediction === 'number' ? e.prediction.toFixed(4) : '—')],
                ["Uncertainty", ...selected.map(e => e.uncertainty ? `±${e.uncertainty.toFixed(4)}` : '—')],
                ...propKeys.map(k => [k, ...selected.map(e => e.properties?.[k] != null ? Number(e.properties[k]).toFixed(2) : '—')])
            ];

            historyCompareTable.innerHTML = rows.map((row, i) => `
                <tr>${row.map(cell => i === 0 ? `<th>${cell}</th>` : `<td>${cell}</td>`).join('')}</tr>
            `).join('');
            historyComparePanel.classList.remove("hidden");
            historyComparePanel.scrollIntoView({ behavior: "smooth" });
        });
    }

    // ── History: Export CSV ─────────────────────────────────────────────────
    if (historyExportBtn) {
        historyExportBtn.addEventListener("click", () => {
            const h = loadHistory();
            if (!h.length) { alert("No history to export"); return; }
            const headers = ["SMILES", "Prediction", "Uncertainty", "Timestamp"];
            const rows = h.map(e => [e.smiles, e.prediction ?? "", e.uncertainty ?? "", e.timestamp]);
            const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
            const blob = new Blob([csv], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `prediction_history_${new Date().toISOString().slice(0, 10)}.csv`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }

    // ── History: Clear ──────────────────────────────────────────────────────
    if (historyClearBtn) {
        historyClearBtn.addEventListener("click", () => {
            if (!confirm("Clear all prediction history?")) return;
            localStorage.removeItem(HISTORY_KEY);
            renderHistory();
            if (historyComparePanel) historyComparePanel.classList.add("hidden");
        });
    }

    // ── Find Analogs → bridge to Optimize tab ───────────────────────────────
    if (findAnalogsBtn) {
        findAnalogsBtn.addEventListener("click", () => {
            if (!lastPredictedSmiles) return;
            // Pre-fill seed molecule and switch to Optimize tab
            if (optSeed) optSeed.value = lastPredictedSmiles;
            document.querySelector('[data-tab="optimize"]').click();
            // Scroll to seed input
            if (optSeed) optSeed.scrollIntoView({ behavior: "smooth", block: "center" });
        });
    }

    // ── Action: Pareto Optimization ─────────────────────────────────────────
    if (paretoBtn) {
        paretoBtn.addEventListener("click", async () => {
            paretoGrid.innerHTML = '';
            paretoResultsPanel.classList.add("hidden");
            paretoLoading.classList.remove("hidden");
            paretoBtn.disabled = true;

            // Gather selected objectives
            const objectives = [];
            if (document.getElementById("pareto-qed")?.checked) objectives.push("qed");
            if (document.getElementById("pareto-negsas")?.checked) objectives.push("neg_sas");
            if (document.getElementById("pareto-logp")?.checked) objectives.push("logp_norm");
            if (document.getElementById("pareto-mw")?.checked) objectives.push("mw_norm");
            if (document.getElementById("pareto-tpsa")?.checked) objectives.push("tpsa_norm");

            if (objectives.length < 2) {
                alert("Select at least 2 objectives for Pareto optimization");
                paretoLoading.classList.add("hidden");
                paretoBtn.disabled = false;
                return;
            }

            const seedValue = optSeed && optSeed.value.trim() ? optSeed.value.trim() : null;

            try {
                const resp = await fetch("/optimize/pareto", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        objectives: objectives,
                        n_samples: parseInt(paretoSamples ? paretoSamples.value : 200),
                        temperature: parseFloat(optTempInput ? optTempInput.value : 0.8),
                        seed_smiles: seedValue
                    })
                });

                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || "Pareto optimization failed");

                lastParetoResults = data.pareto_front || [];

                data.pareto_front.forEach(mol => {
                    const card = document.createElement("div");
                    card.className = "gen-mol-card slide-up";
                    const objEntries = Object.entries(mol.objectives || {})
                        .map(([k, v]) => `<div style="font-size:0.7rem;color:#d946ef;">${k}: ${v.toFixed(3)}</div>`)
                        .join('');
                    const propEntries = Object.entries(mol.properties || {})
                        .filter(([k]) => ["mw","logp","tpsa","qed","sas"].includes(k))
                        .map(([k, v]) => `<div style="font-size:0.7rem;color:#94a3b8;">${k}: ${typeof v === 'number' ? v.toFixed(2) : v}</div>`)
                        .join('');
                    const cd = isFinite(mol.crowding_distance) ? mol.crowding_distance.toFixed(3) : "∞";

                    card.innerHTML = `
                        <span class="gen-badge" style="background:linear-gradient(135deg,#d946ef,#8b5cf6);">Pareto (cd=${cd})</span>
                        <div class="analog-smiles" style="font-size:0.8rem;color:white;margin:0.5rem 0;">${mol.smiles}</div>
                        <div>${objEntries}</div>
                        <div>${propEntries}</div>
                        <button class="btn-primary" style="padding:0.4rem;font-size:0.7rem;margin-top:0.5rem;"
                                onclick="useGenerated('${mol.smiles}')">Predict Properties</button>
                    `;
                    paretoGrid.appendChild(card);
                });

                paretoStats.textContent = `Pareto front: ${data.pareto_count} / ${data.total_sampled} sampled (${data.dominated_count} dominated)`;
                paretoResultsPanel.classList.remove("hidden");
            } catch (err) {
                alert(err.message);
            } finally {
                paretoLoading.classList.add("hidden");
                paretoBtn.disabled = false;
            }
        });
    }

    // ── Pareto CSV Export ───────────────────────────────────────────────────
    if (paretoExportCsv) {
        paretoExportCsv.addEventListener("click", () => {
            if (!lastParetoResults.length) {
                alert("No Pareto results to export");
                return;
            }
            const headers = ["SMILES", "Crowding Distance", "QED", "MW", "LogP", "TPSA", "SAS"];
            const rows = lastParetoResults.map(mol => [
                mol.smiles,
                isFinite(mol.crowding_distance) ? mol.crowding_distance.toFixed(4) : "inf",
                mol.properties?.qed?.toFixed(4) || "",
                mol.properties?.mw?.toFixed(2) || "",
                mol.properties?.logp?.toFixed(2) || "",
                mol.properties?.tpsa?.toFixed(2) || "",
                mol.properties?.sas?.toFixed(2) || ""
            ]);
            const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
            const blob = new Blob([csv], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `pareto_front_${new Date().toISOString().slice(0, 10)}.csv`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }

    // ── ADMET Tab ────────────────────────────────────────────────────────────
    const admetBtn = document.getElementById("admet-btn");
    const admetSmilesInput = document.getElementById("admet-smiles");
    const admetResults = document.getElementById("admet-results");
    const admetScore = document.getElementById("admet-score");
    const admetPassBadge = document.getElementById("admet-pass-badge");
    const admetAlertsSummary = document.getElementById("admet-alerts-summary");
    const admetAlertsPanel = document.getElementById("admet-alerts-panel");
    const admetAlertsList = document.getElementById("admet-alerts-list");
    const admetUseBtn = document.getElementById("admet-use-btn");

    function admetRow(label, value, colorClass = "") {
        return `<div class="admet-row"><span class="admet-label">${label}</span><span class="admet-value ${colorClass}">${value}</span></div>`;
    }

    function boolBadge(val) {
        return val ? '<span class="admet-pass">✓ Pass</span>' : '<span class="admet-fail">✗ Fail</span>';
    }

    function riskColor(risk) {
        if (risk === "High") return "admet-fail";
        if (risk === "Moderate" || risk === "Possible") return "admet-warn";
        return "admet-pass";
    }

    function fillAdmetCard(id, title, rows) {
        const el = document.getElementById(id);
        if (!el) return;
        el.innerHTML = `<h4>${title}</h4>` + rows.join('');
    }

    if (admetBtn) {
        admetBtn.addEventListener("click", async () => {
            const smiles = admetSmilesInput?.value.trim();
            if (!smiles) { alert("Enter a SMILES string"); return; }

            admetBtn.disabled = true;
            admetBtn.textContent = "⏳ Analysing…";
            admetResults.classList.add("hidden");

            try {
                const resp = await fetch("/admet", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ smiles })
                });
                const d = await resp.json();
                if (!resp.ok) throw new Error(d.detail || "ADMET failed");

                // Overall score + badge
                const score = d.overall_score;
                admetScore.textContent = `${score}/100`;
                admetScore.style.color = score >= 70 ? "#10b981" : score >= 50 ? "#fbbf24" : "#f87171";

                if (d.pass_admet) {
                    admetPassBadge.textContent = "✅ ADMET Pass";
                    admetPassBadge.style.background = "rgba(16,185,129,0.15)";
                    admetPassBadge.style.color = "#10b981";
                    admetPassBadge.style.border = "1px solid #10b981";
                } else {
                    admetPassBadge.textContent = "❌ ADMET Concern";
                    admetPassBadge.style.background = "rgba(248,113,113,0.15)";
                    admetPassBadge.style.color = "#f87171";
                    admetPassBadge.style.border = "1px solid #f87171";
                }
                admetAlertsSummary.textContent = d.alerts?.length
                    ? `${d.alerts.length} alert${d.alerts.length > 1 ? 's' : ''} found`
                    : "No alerts";

                // Absorption
                const abs = d.absorption || {};
                fillAdmetCard("admet-absorption", "🫁 Absorption", [
                    admetRow("MW", `${abs.mw} Da`),
                    admetRow("LogP", abs.logp),
                    admetRow("TPSA", `${abs.tpsa} Ų`),
                    admetRow("HBD / HBA", `${abs.hbd} / ${abs.hba}`),
                    admetRow("Rotatable bonds", abs.rotatable_bonds),
                    admetRow("Lipinski Ro5", boolBadge(abs.lipinski_pass)),
                    admetRow("Veber rules", boolBadge(abs.veber_pass)),
                    admetRow("Oral bioavailability", abs.oral_bioavailability,
                        abs.oral_bioavailability === "High" ? "admet-pass" : abs.oral_bioavailability === "Low" ? "admet-fail" : "admet-warn"),
                ]);

                // Distribution
                const dist = d.distribution || {};
                fillAdmetCard("admet-distribution", "🔄 Distribution", [
                    admetRow("BBB permeability", dist.bbb_permeability, riskColor(dist.bbb_permeability === "Low" ? "High" : "Low")),
                    admetRow("TPSA", `${dist.tpsa} Ų`),
                    admetRow("Frac Csp3", dist.frac_csp3),
                    admetRow("Aromatic rings", dist.aromatic_rings),
                ]);

                // Metabolism
                const met = d.metabolism || {};
                const cyp = met.cyp_inhibition || {};
                fillAdmetCard("admet-metabolism", "⚗️ Metabolism", [
                    admetRow("CYP3A4", cyp.CYP3A4, riskColor(cyp.CYP3A4)),
                    admetRow("CYP1A2", cyp.CYP1A2, riskColor(cyp.CYP1A2)),
                    admetRow("CYP2D6", cyp.CYP2D6, riskColor(cyp.CYP2D6)),
                    admetRow("Rings", met.rings),
                ]);

                // Excretion
                const exc = d.excretion || {};
                fillAdmetCard("admet-excretion", "🚽 Excretion", [
                    admetRow("LogP", exc.logp),
                    admetRow("Renal clearance", exc.renal_clearance_estimate),
                ]);

                // Toxicity
                const tox = d.toxicity || {};
                fillAdmetCard("admet-toxicity", "☠️ Toxicity", [
                    admetRow("hERG risk", tox.herg_risk, riskColor(tox.herg_risk)),
                    admetRow("Ames mutagenicity", tox.ames_mutagenicity, riskColor(tox.ames_mutagenicity)),
                    admetRow("PAINS alerts", tox.pains_alerts, tox.pains_alerts > 0 ? "admet-fail" : "admet-pass"),
                    admetRow("Brenk alerts", tox.brenk_alerts, tox.brenk_alerts > 0 ? "admet-warn" : "admet-pass"),
                ]);

                // Alerts list
                if (d.alerts?.length) {
                    admetAlertsList.innerHTML = d.alerts.map(a =>
                        `<li style="font-size:0.82rem;color:#fbbf24;padding:0.2rem 0;">⚠️ ${a}</li>`
                    ).join('');
                    admetAlertsPanel.classList.remove("hidden");
                } else {
                    admetAlertsPanel.classList.add("hidden");
                }

                admetResults.classList.remove("hidden");

                // Use button pre-fills predict
                if (admetUseBtn) {
                    admetUseBtn.onclick = () => {
                        smilesInput.value = smiles;
                        document.querySelector('[data-tab="predict"]').click();
                        predictBtn.click();
                    };
                }
            } catch (err) {
                alert(err.message);
            } finally {
                admetBtn.disabled = false;
                admetBtn.textContent = "🧪 Analyse";
            }
        });

        // Also allow pre-fill from predict tab's SMILES
        document.querySelector('[data-tab="admet"]')?.addEventListener("click", () => {
            if (lastPredictedSmiles && admetSmilesInput && !admetSmilesInput.value) {
                admetSmilesInput.value = lastPredictedSmiles;
            }
        });
    }

    // ── Batch Prediction Tab ────────────────────────────────────────────────
    const batchTextarea = document.getElementById("batch-textarea");
    const batchFile = document.getElementById("batch-file");
    const batchBtn = document.getElementById("batch-btn");
    const batchExplain = document.getElementById("batch-explain");
    const batchResults = document.getElementById("batch-results");
    const batchStats = document.getElementById("batch-stats");
    const batchTable = document.getElementById("batch-table");
    const batchExportBtn = document.getElementById("batch-export-btn");
    const batchClearBtn = document.getElementById("batch-clear-btn");

    let lastBatchResults = [];

    function parseCSV(text) {
        const lines = text.split(/\r?\n/).filter(l => l.trim());
        // Try to find SMILES column
        const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
        const smilesIdx = headers.findIndex(h => h.includes('smiles') || h.includes('molecule'));
        const idx = smilesIdx >= 0 ? smilesIdx : 0;
        return lines.slice(1).map(line => {
            const cols = line.split(',').map(c => c.trim());
            return cols[idx] || '';
        }).filter(s => s);
    }

    function updateBatchButton() {
        const hasText = batchTextarea?.value.trim().length > 0;
        if (batchBtn) batchBtn.disabled = !hasText;
    }

    if (batchTextarea) {
        batchTextarea.addEventListener("input", updateBatchButton);
    }

    if (batchFile) {
        batchFile.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (ev) => {
                try {
                    const text = ev.target.result;
                    const smiles = file.name.endsWith('.csv') ? parseCSV(text) : text.split(/\r?\n/).filter(l => l.trim());
                    if (batchTextarea) batchTextarea.value = smiles.join('\n');
                    updateBatchButton();
                } catch (err) {
                    alert("Failed to parse file: " + err.message);
                }
            };
            reader.readAsText(file);
        });
    }

    if (batchBtn) {
        batchBtn.addEventListener("click", async () => {
            const raw = batchTextarea?.value || '';
            const smilesList = raw.split(/\r?\n/).map(s => s.trim()).filter(s => s);
            if (smilesList.length === 0) { alert("Enter at least one SMILES"); return; }
            if (smilesList.length > 100) { alert("Maximum 100 molecules per batch"); return; }

            batchBtn.disabled = true;
            batchBtn.textContent = "⏳ Predicting…";
            batchResults.classList.add("hidden");

            try {
                const resp = await fetch("/predict/batch", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        smiles_list: smilesList,
                        explain: batchExplain?.checked || false
                    })
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || "Batch prediction failed");

                lastBatchResults = data;
                const tbody = batchTable?.querySelector("tbody");
                if (tbody) {
                    tbody.innerHTML = '';
                    data.forEach((row, i) => {
                        const tr = document.createElement("tr");
                        const status = row.error
                            ? '<span style="color:#f87171;">Error</span>'
                            : '<span style="color:#10b981;">OK</span>';
                        const pred = row.error ? '—' : (Number.isInteger(row.predictions?.task_1) ? row.predictions.task_1 : row.predictions?.task_1?.toFixed(4) || '—');
                        const unc = row.error ? '—' : (row.uncertainty_std?.task_1 ? `±${row.uncertainty_std.task_1.toFixed(4)}` : '—');
                        tr.innerHTML = `
                            <td>${i + 1}</td>
                            <td style="font-family:monospace;font-size:0.8rem;max-width:300px;overflow:hidden;text-overflow:ellipsis;">${row.smiles}</td>
                            <td>${pred}</td>
                            <td>${unc}</td>
                            <td>${status}</td>
                            <td>
                                <button class="btn-secondary" style="padding:0.3rem 0.6rem;font-size:0.75rem;"
                                        onclick="useBatchSmiles('${row.smiles.replace(/'/g, "\\'")}')">Use</button>
                            </td>
                        `;
                        tbody.appendChild(tr);
                    });
                }

                const successCount = data.filter(r => !r.error).length;
                batchStats.textContent = `${successCount}/${data.length} successful predictions`;
                batchResults.classList.remove("hidden");
            } catch (err) {
                alert(err.message);
            } finally {
                batchBtn.disabled = false;
                batchBtn.textContent = "🔮 Predict Batch";
            }
        });
    }

    // Global handler for "Use" button in batch table
    window.useBatchSmiles = (smiles) => {
        smilesInput.value = smiles;
        document.querySelector('[data-tab="predict"]').click();
        predictBtn.click();
    };

    if (batchExportBtn) {
        batchExportBtn.addEventListener("click", () => {
            if (!lastBatchResults.length) { alert("No results to export"); return; }
            const headers = ["SMILES", "Prediction", "Uncertainty", "Standardized_SMILES", "Error"];
            const rows = lastBatchResults.map(r => [
                r.smiles,
                r.predictions?.task_1 ?? '',
                r.uncertainty_std?.task_1 ?? '',
                r.standardized_smiles ?? '',
                r.error ?? ''
            ]);
            const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
            const blob = new Blob([csv], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `batch_predictions_${new Date().toISOString().slice(0,10)}.csv`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }

    if (batchClearBtn) {
        batchClearBtn.addEventListener("click", () => {
            if (batchTextarea) batchTextarea.value = '';
            if (batchFile) batchFile.value = '';
            batchResults.classList.add("hidden");
            lastBatchResults = [];
            updateBatchButton();
        });
    }

    // Enable batch button when model is loaded
    document.addEventListener("model-ready", () => {
        if (batchBtn) batchBtn.disabled = false;
    });

    // ── Similarity Search Tab ────────────────────────────────────────────────
    const searchBtn = document.getElementById("search-btn");
    const searchSmiles = document.getElementById("search-smiles");
    const searchFp = document.getElementById("search-fp");
    const searchTopk = document.getElementById("search-topk");
    const searchThreshold = document.getElementById("search-threshold");
    const searchResults = document.getElementById("search-results");
    const searchStats = document.getElementById("search-stats");
    const searchGrid = document.getElementById("search-grid");

    function updateSearchButton() {
        if (searchBtn) searchBtn.disabled = !(searchSmiles?.value.trim());
    }

    if (searchSmiles) {
        searchSmiles.addEventListener("input", updateSearchButton);
    }

    if (searchBtn) {
        searchBtn.addEventListener("click", async () => {
            const smiles = searchSmiles?.value.trim();
            if (!smiles) return;

            searchBtn.disabled = true;
            searchBtn.textContent = "⏳ Searching…";
            searchResults.classList.add("hidden");

            try {
                const resp = await fetch("/search/similar", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        smiles: smiles,
                        fingerprint_type: searchFp?.value || "morgan",
                        top_k: parseInt(searchTopk?.value || 10),
                        threshold: parseFloat(searchThreshold?.value || 0.5)
                    })
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || "Search failed");

                searchStats.textContent = `${data.results.length} matches from ${data.total_indexed} indexed molecules`;

                searchGrid.innerHTML = '';
                if (data.results.length === 0) {
                    searchGrid.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:2rem;">No similar molecules found above threshold.</p>';
                } else {
                    data.results.forEach((r, i) => {
                        const card = document.createElement("div");
                        card.className = "gen-mol-card slide-up";
                        const simColor = r.score > 0.7 ? "#10b981" : r.score > 0.5 ? "#fbbf24" : "#94a3b8";
                        card.innerHTML = `
                            <span class="gen-badge" style="background:${simColor}20;color:${simColor};border:1px solid ${simColor};">Rank #${r.rank} • Tanimoto ${r.score.toFixed(3)}</span>
                            <div class="analog-smiles" style="font-size:0.8rem;color:white;margin:0.75rem 0;">${r.smiles}</div>
                            <button class="btn-primary" style="padding:0.4rem;font-size:0.75rem;"
                                    onclick="useBatchSmiles('${r.smiles.replace(/'/g, "\\'")}')">Predict</button>
                            <button class="btn-secondary" style="padding:0.4rem;font-size:0.75rem;margin-left:0.3rem;"
                                    onclick="window.admetFromSearch('${r.smiles.replace(/'/g, "\\'")}')">ADMET</button>
                        `;
                        searchGrid.appendChild(card);
                    });
                }

                searchResults.classList.remove("hidden");
            } catch (err) {
                alert(err.message);
            } finally {
                searchBtn.disabled = false;
                searchBtn.textContent = "🔍 Search";
            }
        });
    }

    window.admetFromSearch = (smiles) => {
        if (admetSmilesInput) admetSmilesInput.value = smiles;
        document.querySelector('[data-tab="admet"]').click();
        if (admetBtn) admetBtn.click();
    };

    // Prefill search from last predicted
    document.querySelector('[data-tab="search"]')?.addEventListener("click", () => {
        if (lastPredictedSmiles && searchSmiles && !searchSmiles.value) {
            searchSmiles.value = lastPredictedSmiles;
            updateSearchButton();
        }
    });

    // ── 3D Structure Viewer Tab ──────────────────────────────────────────────
    const viewerBtn = document.getElementById("viewer-btn");
    const viewerSmiles = document.getElementById("viewer-smiles");
    const viewerStyle = document.getElementById("viewer-style");
    const viewerResults = document.getElementById("viewer-results");
    const viewerStats = document.getElementById("viewer-stats");
    const viewerContainer = document.getElementById("viewer-3d-container");
    const viewerSpinBtn = document.getElementById("viewer-spin-btn");
    const viewerResetBtn = document.getElementById("viewer-reset-btn");
    const viewerDownloadBtn = document.getElementById("viewer-download-btn");

    let viewer3D = null;
    let lastPdbBlock = null;
    let lastViewerSmiles = null;
    let isSpinning = false;

    function applyViewerStyle(style) {
        if (!viewer3D) return;
        viewer3D.removeAllShapes();
        let styleObj;
        if (style === "stick") {
            styleObj = { stick: { radius: 0.15, colorscheme: "Jmol" } };
        } else if (style === "ball") {
            styleObj = { stick: { radius: 0.15, colorscheme: "Jmol" }, sphere: { scale: 0.25, colorscheme: "Jmol" } };
        } else if (style === "sphere") {
            styleObj = { sphere: { scale: 0.85, colorscheme: "Jmol" } };
        } else {
            styleObj = { line: { linewidth: 2, colorscheme: "Jmol" } };
        }
        viewer3D.setStyle({}, styleObj);
        viewer3D.render();
    }

    if (viewerBtn) {
        viewerBtn.addEventListener("click", async () => {
            const smiles = viewerSmiles?.value.trim();
            if (!smiles) { alert("Enter a SMILES string"); return; }
            if (typeof $3Dmol === "undefined") {
                alert("3Dmol.js failed to load. Check internet connection.");
                return;
            }

            viewerBtn.disabled = true;
            viewerBtn.textContent = "⏳ Generating…";
            viewerResults.classList.add("hidden");

            try {
                const resp = await fetch("/conformer", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ smiles })
                });
                const data = await resp.json();
                if (!resp.ok || data.error) throw new Error(data.error || data.detail || "Conformer generation failed");
                if (!data.pdb_block) throw new Error("No PDB block returned");

                lastPdbBlock = data.pdb_block;
                lastViewerSmiles = smiles;
                viewerStats.textContent = `${data.num_atoms} atoms • ETKDGv3 + MMFF94 optimized`;

                // Initialize 3Dmol viewer
                viewerContainer.innerHTML = '';
                viewer3D = $3Dmol.createViewer(viewerContainer, {
                    backgroundColor: "rgb(15, 15, 31)"
                });
                viewer3D.addModel(data.pdb_block, "pdb");
                applyViewerStyle(viewerStyle?.value || "stick");
                viewer3D.zoomTo();
                viewer3D.render();
                isSpinning = false;
                viewerSpinBtn.textContent = "⟳ Spin";

                viewerResults.classList.remove("hidden");
            } catch (err) {
                alert(err.message);
            } finally {
                viewerBtn.disabled = false;
                viewerBtn.textContent = "🧬 Generate 3D";
            }
        });
    }

    if (viewerStyle) {
        viewerStyle.addEventListener("change", () => applyViewerStyle(viewerStyle.value));
    }

    if (viewerSpinBtn) {
        viewerSpinBtn.addEventListener("click", () => {
            if (!viewer3D) return;
            if (isSpinning) {
                viewer3D.spin(false);
                viewerSpinBtn.textContent = "⟳ Spin";
            } else {
                viewer3D.spin("y", 1);
                viewerSpinBtn.textContent = "⏹ Stop";
            }
            isSpinning = !isSpinning;
        });
    }

    if (viewerResetBtn) {
        viewerResetBtn.addEventListener("click", () => {
            if (viewer3D) {
                viewer3D.zoomTo();
                viewer3D.render();
            }
        });
    }

    if (viewerDownloadBtn) {
        viewerDownloadBtn.addEventListener("click", () => {
            if (!lastPdbBlock) { alert("No structure to download"); return; }
            const blob = new Blob([lastPdbBlock], { type: "chemical/x-pdb" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            const safeName = (lastViewerSmiles || "molecule").replace(/[^a-zA-Z0-9]/g, "_").slice(0, 30);
            a.href = url;
            a.download = `${safeName}.pdb`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }

    // Prefill from last predicted SMILES
    document.querySelector('[data-tab="viewer3d"]')?.addEventListener("click", () => {
        if (lastPredictedSmiles && viewerSmiles && !viewerSmiles.value) {
            viewerSmiles.value = lastPredictedSmiles;
        }
    });

    // ── Smart Generate (Property-Targeted) ──────────────────────────────────
    const smartGenBtn = document.getElementById("smart-gen-btn");
    const smartResultsPanel = document.getElementById("smart-results-panel");
    const smartGrid = document.getElementById("smart-grid");
    const smartStats = document.getElementById("smart-stats");

    function readSmartTargets() {
        const targets = {};
        const map = [
            { key: "mw", on: "smart-mw-on", min: "smart-mw-min", max: "smart-mw-max" },
            { key: "logp", on: "smart-logp-on", min: "smart-logp-min", max: "smart-logp-max" },
            { key: "qed", on: "smart-qed-on", min: "smart-qed-min", max: "smart-qed-max" },
            { key: "tpsa", on: "smart-tpsa-on", min: "smart-tpsa-min", max: "smart-tpsa-max" },
        ];
        for (const m of map) {
            const onEl = document.getElementById(m.on);
            if (!onEl?.checked) continue;
            const lo = parseFloat(document.getElementById(m.min).value);
            const hi = parseFloat(document.getElementById(m.max).value);
            if (!isNaN(lo) && !isNaN(hi)) targets[m.key] = [lo, hi];
        }
        return targets;
    }

    if (smartGenBtn) {
        smartGenBtn.addEventListener("click", async () => {
            const targets = readSmartTargets();
            if (Object.keys(targets).length === 0) {
                alert("Enable at least one property target");
                return;
            }

            smartGenBtn.disabled = true;
            smartGenBtn.textContent = "⏳ Sampling…";
            smartResultsPanel.classList.add("hidden");

            try {
                const resp = await fetch("/generate/smart", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        n_target: parseInt(document.getElementById("smart-n-target").value),
                        max_attempts: parseInt(document.getElementById("smart-max-attempts").value),
                        temperature: 0.9,
                        targets: targets
                    })
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || "Smart generation failed");

                smartStats.textContent =
                    `${data.n_accepted} accepted from ${data.n_attempts} samples ` +
                    `(${(data.acceptance_rate * 100).toFixed(1)}% acceptance)`;

                smartGrid.innerHTML = '';
                if (data.molecules.length === 0) {
                    smartGrid.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:2rem;">No molecules matched criteria. Try relaxing constraints or increasing max attempts.</p>';
                } else {
                    data.molecules.forEach(mol => {
                        const card = document.createElement("div");
                        card.className = "gen-mol-card slide-up";
                        const propEntries = Object.entries(mol.properties || {})
                            .filter(([k]) => ["mw","logp","tpsa","qed","hbd","hba"].includes(k))
                            .map(([k, v]) => `<div style="font-size:0.7rem;color:#94a3b8;">${k}: ${v}</div>`)
                            .join('');
                        card.innerHTML = `
                            <span class="gen-badge" style="background:linear-gradient(135deg,#10b981,#06b6d4);">✓ Match</span>
                            <div class="analog-smiles" style="font-size:0.8rem;color:white;margin:0.5rem 0;">${mol.smiles}</div>
                            <div>${propEntries}</div>
                            <button class="btn-primary" style="padding:0.4rem;font-size:0.7rem;margin-top:0.5rem;"
                                    onclick="useBatchSmiles('${mol.smiles.replace(/'/g, "\\'")}')">Predict</button>
                        `;
                        smartGrid.appendChild(card);
                    });
                }
                smartResultsPanel.classList.remove("hidden");
            } catch (err) {
                alert(err.message);
            } finally {
                smartGenBtn.disabled = false;
                smartGenBtn.textContent = "🎯 Smart Generate";
            }
        });
    }

    // Enable smart generate when VAE is ready
    document.addEventListener("vae-ready", () => {
        if (smartGenBtn) smartGenBtn.disabled = false;
    });

    // ── Dashboard Tab ────────────────────────────────────────────────────────
    const kpiTotal = document.getElementById("kpi-total");
    const kpiMean = document.getElementById("kpi-mean");
    const kpiRange = document.getElementById("kpi-range");
    const kpiUncertainty = document.getElementById("kpi-uncertainty");
    const histCanvas = document.getElementById("dashboard-histogram");
    const timelineCanvas = document.getElementById("dashboard-timeline");

    function drawHistogram(canvas, values) {
        if (!canvas || !values.length) return;
        const ctx = canvas.getContext("2d");
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        const numBins = 12;
        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min || 1;
        const binSize = range / numBins;
        const bins = new Array(numBins).fill(0);
        values.forEach(v => {
            const idx = Math.min(Math.floor((v - min) / binSize), numBins - 1);
            bins[idx]++;
        });
        const maxCount = Math.max(...bins);

        const padL = 50, padR = 20, padT = 20, padB = 40;
        const plotW = W - padL - padR;
        const plotH = H - padT - padB;
        const barW = plotW / numBins;

        // Axes
        ctx.strokeStyle = "rgba(255,255,255,0.2)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, padT);
        ctx.lineTo(padL, H - padB);
        ctx.lineTo(W - padR, H - padB);
        ctx.stroke();

        // Bars
        bins.forEach((count, i) => {
            const h = (count / maxCount) * plotH;
            const x = padL + i * barW;
            const y = H - padB - h;
            const grad = ctx.createLinearGradient(0, y, 0, H - padB);
            grad.addColorStop(0, "#8b5cf6");
            grad.addColorStop(1, "#6366f1");
            ctx.fillStyle = grad;
            ctx.fillRect(x + 2, y, barW - 4, h);

            // Count label
            if (count > 0) {
                ctx.fillStyle = "#cbd5e1";
                ctx.font = "11px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText(count, x + barW / 2, y - 4);
            }
        });

        // X axis labels (min, mid, max)
        ctx.fillStyle = "#94a3b8";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(min.toFixed(2), padL, H - padB + 18);
        ctx.fillText(((min + max) / 2).toFixed(2), padL + plotW / 2, H - padB + 18);
        ctx.fillText(max.toFixed(2), W - padR, H - padB + 18);

        ctx.textAlign = "left";
        ctx.fillStyle = "#cbd5e1";
        ctx.font = "12px sans-serif";
        ctx.fillText("Frequency", 10, padT + 10);
        ctx.fillText("Predicted Value", padL + plotW / 2 - 40, H - 8);
    }

    function drawTimeline(canvas, points) {
        if (!canvas || !points.length) return;
        const ctx = canvas.getContext("2d");
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        const padL = 50, padR = 20, padT = 20, padB = 30;
        const plotW = W - padL - padR;
        const plotH = H - padT - padB;

        const values = points.map(p => p.value);
        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min || 1;

        // Axes
        ctx.strokeStyle = "rgba(255,255,255,0.2)";
        ctx.beginPath();
        ctx.moveTo(padL, padT);
        ctx.lineTo(padL, H - padB);
        ctx.lineTo(W - padR, H - padB);
        ctx.stroke();

        // Line
        ctx.strokeStyle = "#8b5cf6";
        ctx.lineWidth = 2;
        ctx.beginPath();
        points.forEach((p, i) => {
            const x = padL + (i / Math.max(1, points.length - 1)) * plotW;
            const y = H - padB - ((p.value - min) / range) * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Dots
        points.forEach((p, i) => {
            const x = padL + (i / Math.max(1, points.length - 1)) * plotW;
            const y = H - padB - ((p.value - min) / range) * plotH;
            ctx.fillStyle = "#a78bfa";
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();
        });

        // Y labels
        ctx.fillStyle = "#94a3b8";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(max.toFixed(2), padL - 6, padT + 4);
        ctx.fillText(min.toFixed(2), padL - 6, H - padB);

        ctx.textAlign = "center";
        ctx.fillText(`Earliest`, padL + 20, H - padB + 18);
        ctx.fillText(`Latest`, W - padR - 20, H - padB + 18);
    }

    function refreshDashboard() {
        const h = loadHistory();
        const validPredictions = h.filter(e => typeof e.prediction === 'number');

        kpiTotal.textContent = h.length;

        if (validPredictions.length === 0) {
            kpiMean.textContent = "—";
            kpiRange.textContent = "—";
            kpiUncertainty.textContent = "—";
            const hctx = histCanvas?.getContext("2d");
            if (hctx) hctx.clearRect(0, 0, histCanvas.width, histCanvas.height);
            const tctx = timelineCanvas?.getContext("2d");
            if (tctx) tctx.clearRect(0, 0, timelineCanvas.width, timelineCanvas.height);
            return;
        }

        const values = validPredictions.map(e => e.prediction);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const minV = Math.min(...values);
        const maxV = Math.max(...values);

        kpiMean.textContent = mean.toFixed(3);
        kpiRange.textContent = `${minV.toFixed(2)} / ${maxV.toFixed(2)}`;

        const uncertainties = validPredictions
            .map(e => e.uncertainty)
            .filter(u => typeof u === 'number');
        if (uncertainties.length) {
            const meanU = uncertainties.reduce((a, b) => a + b, 0) / uncertainties.length;
            kpiUncertainty.textContent = `±${meanU.toFixed(3)}`;
        } else {
            kpiUncertainty.textContent = "—";
        }

        drawHistogram(histCanvas, values);
        // Reverse history so earliest is first for timeline
        const timelinePoints = [...validPredictions].reverse().map(e => ({ value: e.prediction }));
        drawTimeline(timelineCanvas, timelinePoints);
    }

    // Refresh dashboard when its tab is opened
    document.querySelector('[data-tab="dashboard"]')?.addEventListener("click", refreshDashboard);

    // ── Compound Library Tab ────────────────────────────────────────────────
    const libSmiles = document.getElementById("lib-smiles");
    const libName = document.getElementById("lib-name");
    const libProject = document.getElementById("lib-project");
    const libTags = document.getElementById("lib-tags");
    const libNotes = document.getElementById("lib-notes");
    const libSaveBtn = document.getElementById("lib-save-btn");
    const libSearch = document.getElementById("lib-search");
    const libFilterProject = document.getElementById("lib-filter-project");
    const libFilterTag = document.getElementById("lib-filter-tag");
    const libRefreshBtn = document.getElementById("lib-refresh-btn");
    const libTbody = document.getElementById("lib-tbody");
    const libEmpty = document.getElementById("lib-empty");

    function libEscape(s) {
        return String(s ?? "").replace(/[&<>"']/g, c => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
        }[c]));
    }

    async function libLoadFilters() {
        try {
            const resp = await fetch("/library/projects");
            const data = await resp.json();
            const curP = libFilterProject.value;
            libFilterProject.innerHTML = '<option value="">All projects</option>' +
                (data.projects || []).map(p => `<option value="${libEscape(p)}">${libEscape(p)}</option>`).join('');
            libFilterProject.value = curP;
            const curT = libFilterTag.value;
            libFilterTag.innerHTML = '<option value="">All tags</option>' +
                (data.tags || []).map(t => `<option value="${libEscape(t)}">${libEscape(t)}</option>`).join('');
            libFilterTag.value = curT;
        } catch (err) {
            console.warn("Failed to load library filters:", err);
        }
    }

    async function libRefresh() {
        try {
            const params = new URLSearchParams();
            if (libFilterProject.value) params.set("project", libFilterProject.value);
            if (libFilterTag.value) params.set("tag", libFilterTag.value);
            if (libSearch.value.trim()) params.set("search", libSearch.value.trim());
            params.set("limit", "200");
            const resp = await fetch(`/library?${params.toString()}`);
            const data = await resp.json();
            const compounds = data.compounds || [];

            libTbody.innerHTML = "";
            if (compounds.length === 0) {
                libEmpty.classList.remove("hidden");
                return;
            }
            libEmpty.classList.add("hidden");

            compounds.forEach(c => {
                const tagsHtml = (c.tags || [])
                    .map(t => `<span class="history-prop-chip">${libEscape(t)}</span>`)
                    .join(' ');
                const updated = c.updated_at ? new Date(c.updated_at).toLocaleString() : "";
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td>${c.id}</td>
                    <td style="font-family:monospace;font-size:0.78rem;max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${libEscape(c.smiles)}</td>
                    <td>${libEscape(c.name || "—")}</td>
                    <td>${libEscape(c.project)}</td>
                    <td>${tagsHtml || '<span style="color:var(--text-muted);">—</span>'}</td>
                    <td style="font-size:0.75rem;color:var(--text-muted);">${updated}</td>
                    <td style="white-space:nowrap;">
                        <button class="btn-secondary" style="padding:0.3rem 0.6rem;font-size:0.75rem;"
                                onclick="window.libLoadInto('${libEscape(c.smiles)}')">Predict</button>
                        <button class="btn-secondary" style="padding:0.3rem 0.6rem;font-size:0.75rem;"
                                onclick="window.libDelete(${c.id})" style="color:var(--danger);">🗑</button>
                    </td>
                `;
                libTbody.appendChild(tr);
            });
        } catch (err) {
            alert("Failed to load library: " + err.message);
        }
    }

    if (libSaveBtn) {
        libSaveBtn.addEventListener("click", async () => {
            const smiles = libSmiles.value.trim();
            if (!smiles) { alert("Enter a SMILES string"); return; }

            const tags = libTags.value
                .split(",").map(t => t.trim()).filter(t => t);
            try {
                const resp = await fetch("/library", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        smiles,
                        name: libName.value.trim() || null,
                        project: libProject.value.trim() || "default",
                        tags,
                        notes: libNotes.value.trim() || null,
                        properties: {}
                    })
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || "Save failed");

                // Clear form (keep project)
                libSmiles.value = "";
                libName.value = "";
                libTags.value = "";
                libNotes.value = "";
                await libLoadFilters();
                await libRefresh();
            } catch (err) {
                alert(err.message);
            }
        });
    }

    window.libLoadInto = (smiles) => {
        smilesInput.value = smiles;
        document.querySelector('[data-tab="predict"]').click();
        predictBtn.click();
    };

    window.libDelete = async (id) => {
        if (!confirm(`Delete compound #${id}?`)) return;
        try {
            const resp = await fetch(`/library/${id}`, { method: "DELETE" });
            if (!resp.ok) throw new Error("Delete failed");
            await libRefresh();
        } catch (err) {
            alert(err.message);
        }
    };

    if (libRefreshBtn) libRefreshBtn.addEventListener("click", libRefresh);
    if (libFilterProject) libFilterProject.addEventListener("change", libRefresh);
    if (libFilterTag) libFilterTag.addEventListener("change", libRefresh);
    if (libSearch) {
        let searchTimer;
        libSearch.addEventListener("input", () => {
            clearTimeout(searchTimer);
            searchTimer = setTimeout(libRefresh, 300);
        });
    }

    // Auto-load library tab on first click
    document.querySelector('[data-tab="library"]')?.addEventListener("click", async () => {
        if (lastPredictedSmiles && libSmiles && !libSmiles.value) {
            libSmiles.value = lastPredictedSmiles;
        }
        await libLoadFilters();
        await libRefresh();
    });

    // Add "Save to Library" buttons to the predict result panel
    const predictSaveBtnHook = () => {
        // Inject save-to-library button if a result is shown
        const result = document.getElementById("result");
        if (!result || result.classList.contains("hidden")) return;
        if (document.getElementById("predict-save-lib-btn")) return;
        const btn = document.createElement("button");
        btn.id = "predict-save-lib-btn";
        btn.className = "btn-secondary";
        btn.textContent = "📚 Save to Library";
        btn.style.cssText = "padding:0.4rem 0.9rem;font-size:0.85rem;margin-left:0.5rem;";
        btn.addEventListener("click", async () => {
            if (!lastPredictedSmiles) return;
            const tag = prompt("Optional tags (comma-separated):", "predicted") || "";
            try {
                const resp = await fetch("/library", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        smiles: lastPredictedSmiles,
                        project: "default",
                        tags: tag.split(",").map(t => t.trim()).filter(t => t),
                        properties: {}
                    })
                });
                if (!resp.ok) {
                    const d = await resp.json();
                    throw new Error(d.detail || "Save failed");
                }
                btn.textContent = "✅ Saved!";
                setTimeout(() => { btn.textContent = "📚 Save to Library"; }, 1500);
            } catch (err) {
                alert(err.message);
            }
        });
        // Find a place to put it
        const predictedSmiles = document.getElementById("predicted-smiles");
        if (predictedSmiles && predictedSmiles.parentElement) {
            predictedSmiles.parentElement.appendChild(btn);
        }
    };

    // Hook into prediction completion
    const origObserver = new MutationObserver(predictSaveBtnHook);
    const resultEl = document.getElementById("result");
    if (resultEl) origObserver.observe(resultEl, { attributes: true, attributeFilter: ["class"] });

    // ── Scaffold Analysis Tab ───────────────────────────────────────────────
    const scaffoldBtn = document.getElementById("scaffold-btn");
    const scaffoldSmilesIn = document.getElementById("scaffold-smiles");
    const scaffoldResults = document.getElementById("scaffold-results");

    if (scaffoldBtn) {
        scaffoldBtn.addEventListener("click", async () => {
            const smiles = scaffoldSmilesIn.value.trim();
            if (!smiles) { alert("Enter a SMILES string"); return; }

            scaffoldBtn.disabled = true;
            scaffoldBtn.textContent = "⏳ Analyzing…";
            scaffoldResults.classList.add("hidden");

            try {
                const resp = await fetch("/scaffold", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ smiles })
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || "Analysis failed");

                // SA score
                const saScore = data.sa_score;
                document.getElementById("sa-score-value").textContent = saScore.toFixed(2);

                const colors = {
                    easy: { bg: "rgba(16,185,129,0.2)", fg: "#10b981" },
                    moderate: { bg: "rgba(251,191,36,0.2)", fg: "#fbbf24" },
                    hard: { bg: "rgba(248,113,113,0.2)", fg: "#f87171" }
                };
                const c = colors[data.sa_class] || colors.moderate;
                const badge = document.getElementById("sa-class-badge");
                badge.textContent = data.sa_class.toUpperCase();
                badge.style.background = c.bg;
                badge.style.color = c.fg;
                document.getElementById("sa-score-value").style.color = c.fg;

                // Marker (1-10 mapped to 0-100%)
                const markerPct = ((saScore - 1) / 9) * 100;
                document.getElementById("sa-marker").style.left = `calc(${Math.max(0, Math.min(100, markerPct))}% - 2px)`;

                // Scaffolds
                document.getElementById("scaffold-murcko").textContent = data.murcko_smiles || "(no rings)";
                document.getElementById("scaffold-generic").textContent = data.generic_murcko_smiles || "(no rings)";

                // Ring metrics
                document.getElementById("ring-total").textContent = data.num_rings;
                document.getElementById("ring-arom").textContent = data.num_aromatic_rings;
                document.getElementById("ring-aliph").textContent = data.num_aliphatic_rings;
                document.getElementById("ring-largest").textContent = data.largest_ring_size;
                document.getElementById("ring-spiro").textContent = data.num_spiro_atoms;
                document.getElementById("ring-bridge").textContent = data.num_bridgehead_atoms;
                document.getElementById("ring-macro").textContent = data.has_macrocycle ? "Yes" : "No";

                scaffoldResults.classList.remove("hidden");

                // Also fetch functional groups
                try {
                    const fgResp = await fetch("/functional_groups", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ smiles })
                    });
                    const fg = await fgResp.json();
                    const fgListEl = document.getElementById("fg-list");
                    const fgEmpty = document.getElementById("fg-empty");
                    const fgCats = document.getElementById("fg-categories");
                    const fgSummary = document.getElementById("fg-summary");

                    fgSummary.textContent = `${fg.num_groups_found} group${fg.num_groups_found === 1 ? "" : "s"} detected`;

                    // Category chips
                    const catColors = {
                        Carbonyl: "#f59e0b", Amine: "#8b5cf6", Nitrogen: "#a78bfa",
                        Oxygen: "#06b6d4", Sulfur: "#fbbf24", Phosphorus: "#fb923c",
                        Halogen: "#10b981", Aromatic: "#ec4899", Heteroaromatic: "#f472b6",
                        Aliphatic: "#94a3b8", Cyclic: "#6366f1"
                    };
                    fgCats.innerHTML = Object.entries(fg.categories || {})
                        .sort((a, b) => b[1] - a[1])
                        .map(([cat, n]) => {
                            const c = catColors[cat] || "#64748b";
                            return `<span style="padding:0.3rem 0.7rem;border-radius:14px;background:${c}20;color:${c};border:1px solid ${c};font-size:0.78rem;font-weight:500;">${cat}: ${n}</span>`;
                        }).join("");

                    // Group cards
                    fgListEl.innerHTML = "";
                    if (fg.groups.length === 0) {
                        fgEmpty.classList.remove("hidden");
                    } else {
                        fgEmpty.classList.add("hidden");
                        fg.groups
                            .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
                            .forEach(g => {
                                const c = catColors[g.category] || "#64748b";
                                const card = document.createElement("div");
                                card.style.cssText = `padding:0.6rem 0.8rem;border-left:3px solid ${c};background:rgba(255,255,255,0.03);border-radius:6px;`;
                                card.innerHTML = `
                                    <div style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;">
                                        <span style="font-weight:600;color:white;">${g.name}</span>
                                        <span style="font-size:0.78rem;color:${c};font-weight:700;">×${g.count}</span>
                                    </div>
                                    <div style="font-family:monospace;font-size:0.7rem;color:var(--text-muted);margin-top:0.2rem;">${g.smarts}</div>
                                `;
                                fgListEl.appendChild(card);
                            });
                    }
                } catch (fgErr) {
                    console.warn("Functional group detection failed:", fgErr);
                }
            } catch (err) {
                alert(err.message);
            } finally {
                scaffoldBtn.disabled = false;
                scaffoldBtn.textContent = "🦴 Analyze";
            }
        });
    }

    // Prefill scaffold SMILES from last predicted
    document.querySelector('[data-tab="scaffold"]')?.addEventListener("click", () => {
        if (lastPredictedSmiles && scaffoldSmilesIn && !scaffoldSmilesIn.value) {
            scaffoldSmilesIn.value = lastPredictedSmiles;
        }
    });

    // ── Markdown Report Download ────────────────────────────────────────────
    const reportBtn = document.getElementById("report-btn");
    if (reportBtn) {
        reportBtn.addEventListener("click", async () => {
            const smiles = (scaffoldSmilesIn?.value || lastPredictedSmiles || "").trim();
            if (!smiles) { alert("Enter a SMILES string"); return; }

            reportBtn.disabled = true;
            reportBtn.textContent = "⏳ Generating…";
            try {
                const resp = await fetch("/report", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ smiles })
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || "Report generation failed");

                const blob = new Blob([data.markdown], { type: "text/markdown" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                const stamp = new Date().toISOString().slice(0, 10);
                const safe = smiles.replace(/[^a-zA-Z0-9]/g, "_").slice(0, 30);
                a.href = url;
                a.download = `report_${safe}_${stamp}.md`;
                a.click();
                URL.revokeObjectURL(url);
            } catch (err) {
                alert(err.message);
            } finally {
                reportBtn.disabled = false;
                reportBtn.textContent = "📄 Download Report";
            }
        });
    }

    // Enter key support
    smilesInput.addEventListener("keypress", (e) => { if (e.key === "Enter") predictBtn.click(); });
});
