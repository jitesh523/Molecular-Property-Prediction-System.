document.addEventListener("DOMContentLoaded", () => {
    const predictBtn = document.getElementById("predict-btn");
    const smilesInput = document.getElementById("smiles-input");
    const explainCb = document.getElementById("explain-cb");
    const uncertaintyInput = document.getElementById("uncertainty-input");
    const resultsPanel = document.getElementById("results-panel");
    const loading = document.getElementById("loading");
    const predValue = document.getElementById("pred-value");
    const predUncertainty = document.getElementById("pred-uncertainty");
    const svgContainer = document.getElementById("svg-container");
    const explanationPanel = document.getElementById("explanation-panel");
    const taskLabel = document.getElementById("task-label");
    const appTitle = document.getElementById("app-title");
    const similarityPanel = document.getElementById("similarity-panel");
    const analogsContainer = document.getElementById("analogs-container");

    // Fetch active model info on load
    fetch("/model/info")
        .then(r => r.json())
        .then(data => {
            if (data.status === "loaded") {
                appTitle.textContent = `Active Model: ${data.model_type.toUpperCase()} • Dataset: ${data.dataset.toUpperCase()}`;
                
                if (data.task === "regression") {
                    taskLabel.textContent = "Predicted Value (LogS / pIC50)";
                } else {
                    taskLabel.textContent = "Probability (Active)";
                }
            } else {
                appTitle.textContent = "⚠️ No active model loaded in API";
                appTitle.style.color = "#ef4444"; // red
                predictBtn.disabled = true;
            }
        }).catch(err => {
            console.error("Error fetching model info:", err);
            appTitle.textContent = "⚠️ Cannot connect to API";
            appTitle.style.color = "#ef4444";
        });

    predictBtn.addEventListener("click", async () => {
        const smiles = smilesInput.value.trim();
        if (!smiles) {
            alert("Please enter a SMILES string to predict. Example: CC(=O)OC1=CC=CC=C1C(=O)O");
            return;
        }

        const explain = explainCb.checked;
        const uncertainty = parseInt(uncertaintyInput.value) || 0;

        // Reset UI state
        resultsPanel.classList.add("hidden");
        explanationPanel.classList.add("hidden");
        similarityPanel.classList.add("hidden");
        loading.classList.remove("hidden");
        predictBtn.disabled = true;
        predictBtn.textContent = "Predicting...";
        svgContainer.innerHTML = ''; // clear previous explanations

        try {
            const resp = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    smiles: smiles,
                    explain: explain,
                    uncertainty_samples: uncertainty
                })
            });

            const data = await resp.json();
            
            if (!resp.ok) {
                alert(`Error: ${data.detail || 'Prediction failed'}`);
                return;
            }

            // Set metric values
            let val = data.predictions.task_1 !== undefined ? data.predictions.task_1 : null;
            if (val !== null) {
                // If it's probability mode, maybe format as percentage, but taskLabel specifies it
                predValue.textContent = Number.isInteger(val) ? val : parseFloat(val).toFixed(4);
            } else {
                predValue.textContent = "-";
            }

            if (data.uncertainty_std && data.uncertainty_std.task_1 !== undefined) {
                predUncertainty.textContent = `± ${parseFloat(data.uncertainty_std.task_1).toFixed(4)}`;
            } else {
                predUncertainty.textContent = "N/A";
            }

            // Display Structural SVG if requested
            if (data.explanation && data.explanation.svg) {
                svgContainer.innerHTML = data.explanation.svg;
                explanationPanel.classList.remove("hidden");
                
                // Slight hack to give the SVG full width cleanly
                const svg = svgContainer.querySelector("svg");
                if (svg) {
                    svg.style.width = "100%";
                    svg.style.height = "auto";
                }
            }

            // Display Similar Molecules (Vector Search Analogs)
            if (data.similar_molecules && data.similar_molecules.length > 0) {
                analogsContainer.innerHTML = '';
                data.similar_molecules.forEach(mol => {
                    const item = document.createElement("div");
                    item.className = "analog-item animate-fade-in";
                    item.innerHTML = `
                        <div class="analog-score">Similarity: ${(mol.score * 100).toFixed(1)}%</div>
                        <span class="analog-smiles">${mol.smiles}</span>
                    `;
                    analogsContainer.appendChild(item);
                });
                similarityPanel.classList.remove("hidden");
            }

            // Reflow and animated display
            resultsPanel.classList.remove("hidden");
        } catch (error) {
            console.error(error);
            alert(`Network error: ${error.message}`);
        } finally {
            loading.classList.add("hidden");
            predictBtn.disabled = false;
            predictBtn.textContent = "Predict →";
        }
    });

    smilesInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            predictBtn.click();
        }
    });
});
