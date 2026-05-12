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

            try {
                const resp = await fetch("/optimize", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        targets: targets,
                        method: optMethod ? optMethod.value : "gradient_ascent",
                        n_candidates: parseInt(optCandidates ? optCandidates.value : 10),
                        temperature: parseFloat(optTempInput ? optTempInput.value : 0.8)
                    })
                });

                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || "Optimization failed");

                data.candidates.forEach((mol, idx) => {
                    const card = document.createElement("div");
                    card.className = "gen-mol-card slide-up";
                    const props = Object.entries(mol.properties || {})
                        .map(([k, v]) => `<div style="font-size:0.7rem;color:#94a3b8;">${k}: ${typeof v === 'number' ? v.toFixed(1) : v}</div>`)
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

                optStats.textContent = `Valid candidates: ${data.valid_count}/${data.total_attempts} attempts using ${data.method}`;
                optResultsPanel.classList.remove("hidden");
            } catch (err) {
                alert(err.message);
            } finally {
                optLoading.classList.add("hidden");
                optimizeBtn.disabled = false;
            }
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

    // Enter key support
    smilesInput.addEventListener("keypress", (e) => { if (e.key === "Enter") predictBtn.click(); });
});
