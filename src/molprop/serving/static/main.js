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

    // Store last optimization results for CSV export
    let lastOptimizeResults = [];
    let lastParetoResults = [];
    let lastPredictedSmiles = "";

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
                if (paretoBtn) paretoBtn.disabled = false;
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

    // Enter key support
    smilesInput.addEventListener("keypress", (e) => { if (e.key === "Enter") predictBtn.click(); });
});
