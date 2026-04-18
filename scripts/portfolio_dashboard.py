import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import py3Dmol
import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from stmol import showmol

from molprop.data.standardize import standardize_smiles
from molprop.features.conformers import generate_3d_conformer, mol_to_pdb
from molprop.features.graphs import smiles_to_graph
from molprop.models.explain import explain_graph, get_explainer
from molprop.models.visualize_explanations import get_explanation_image
from molprop.serving.load_model import load_gnn_model

st.set_page_config(
    page_title="Molecular Property Prediction Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown(
    """
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""",
    unsafe_allow_html=True,
)


# Cache model loading
@st.cache_resource
def get_model(model_type, dataset_name):
    weights_map = {
        "BBBP (Blood-Brain Barrier)": "best_gat_bbbp.pt",
        "Delaney (Solubility)": "best_gat_delaney.pt",
    }
    weights_path = weights_map.get(dataset_name)
    if not weights_path or not os.path.exists(weights_path):
        return None
    return load_gnn_model("gat", weights_path, in_dim=9, hidden_dim=128, out_dim=1)


def run_prediction(model, smiles, n_samples=10):
    std_smiles = standardize_smiles(smiles)
    if not std_smiles:
        return None, "Invalid SMILES"
    data = smiles_to_graph(std_smiles)
    if not data:
        return None, "Graph conversion failed"

    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(data)
            preds.append(out.item())
    return np.mean(preds), np.std(preds)


st.title("🧬 Molecular Property Prediction System")
st.markdown("""
Welcome to the interactive explorer for the **Molecular Property Prediction System**.
This dashboard allows you to predict physicochemical and ADMET properties and explore chemical space.
""")

st.sidebar.header("Configuration")
dataset_choice = st.sidebar.selectbox(
    "Select Model/Dataset", ["BBBP (Blood-Brain Barrier)", "Delaney (Solubility)"]
)

model = get_model("gat", dataset_choice)

# Tabs for different views
tab_pred, tab_viz, tab_bench, tab_3d, tab_about = st.tabs(
    [
        "🔍 Prediction Explorer",
        "📊 Chemical Space",
        "🏆 Benchmarks",
        "🧊 3D Inspector",
        "ℹ️ About the Project",
    ]
)

with tab_pred:
    st.header("Single Molecule Prediction")
    smiles_input = st.text_input("Enter SMILES string:", "CC(=O)OC1=CC=CC=C1C(=O)O")

    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            st.success(f"Valid SMILES: {smiles_input}")
            col1, col2 = st.columns([1, 1])
            with col1:
                img = Draw.MolToImage(mol, size=(400, 400))
                st.image(img, caption="Standardized Structure", use_column_width=True)
            with col2:
                st.subheader("Model Predictions")
                if model:
                    mean_val, std_val = run_prediction(model, smiles_input)
                    if mean_val is not None:
                        if "BBBP" in dataset_choice:
                            prob = 1 / (1 + np.exp(-mean_val))
                            st.metric("Probability (BBB+)", f"{prob:.2%}")
                        else:
                            st.metric("LogS (Solubility)", f"{mean_val:.2f}")
                        st.write(f"**Uncertainty (MC Dropout σ):** {std_val:.4f}")

                    st.divider()
                    if st.button("✨ Explain Decision"):
                        with st.spinner("Generating attribution map..."):
                            # Setup explainer (using Captum/IntegratedGradients for speed)
                            task_type = (
                                "binary_classification"
                                if "BBBP" in dataset_choice
                                else "regression"
                            )
                            explainer = get_explainer(
                                model, task_type=task_type, algorithm="captum"
                            )

                            std_smiles = standardize_smiles(smiles_input)
                            data = smiles_to_graph(std_smiles)

                            explanation = explain_graph(
                                explainer, data.x, data.edge_index, data.edge_attr
                            )
                            svg_data = get_explanation_image(std_smiles, explanation)

                            if svg_data:
                                st.subheader("Structural Evidence")
                                st.write(
                                    "The highlighted atoms contribute most to this prediction."
                                )
                                st.image(io.BytesIO(svg_data.encode()), use_column_width=True)
                            else:
                                st.error("Failed to generate explanation.")
                else:
                    st.warning(f"Model weights for {dataset_choice} not found.")
        else:
            st.error("Invalid SMILES.")

with tab_viz:
    st.header("Interactive Chemical Space Explorer")
    st.markdown(
        "Explore the UMAP projection of the dataset. Each point represents a molecule, colored by its property value."
    )

    # Try to load pre-computed UMAP data
    dataset_name = "bbbp" if "BBBP" in dataset_choice else "delaney"
    csv_path = f"results/plots/chemical_space/umap_{dataset_name}_morgan.csv"

    if os.path.exists(csv_path):
        df_umap = pd.read_csv(csv_path)
        target_col = [c for c in df_umap.columns if c not in ["std_smiles", "UMAP1", "UMAP2"]][0]

        fig = px.scatter(
            df_umap,
            x="UMAP1",
            y="UMAP2",
            color=target_col,
            hover_data=["std_smiles"],
            color_continuous_scale="Viridis",
            title=f"UMAP Projection of {dataset_name.upper()} Dataset",
            template="plotly_white",
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            f"UMAP data for {dataset_name} not found. Run `python scripts/visualize_chemical_space.py --dataset {dataset_name}` to generate it."
        )

with tab_bench:
    st.header("Model Performance Benchmarks")
    st.markdown("Automated comparison of all implemented architectures across multiple datasets.")

    whitepaper_path = "results/WHITE-PAPER.md"
    if os.path.exists(whitepaper_path):
        with open(whitepaper_path, "r") as f:
            st.markdown(f.read())
    else:
        st.info(
            "Technical whitepaper not found. Run `python scripts/generate_whitepaper.py` to generate the full report."
        )

    csv_path = "results/benchmark_table.csv"
    if os.path.exists(csv_path):
        st.subheader("Raw Metric Data")
        df_bench = pd.read_csv(csv_path)
        st.dataframe(
            df_bench.style.highlight_max(
                axis=0, subset=["R²", "ROC-AUC", "PR-AUC"], color="lightgreen"
            )
        )
    else:
        st.info("Benchmark table not found. Run `python scripts/generate_benchmark.py` first.")

with tab_3d:
    st.header("Interactive 3D Structure Inspector")
    st.markdown("Generate and explore the 3D optimized geometry (MMFF94) of the molecule.")

    smiles_3d = st.text_input("Enter SMILES for 3D generation:", smiles_input, key="3d_input")

    if smiles_3d:
        with st.spinner("Embedding and optimizing conformer..."):
            mol_3d = generate_3d_conformer(smiles_3d)
            if mol_3d:
                pdb_block = mol_to_pdb(mol_3d)

                col_ctrl, col_view = st.columns([1, 3])
                with col_ctrl:
                    style = st.selectbox("Style", ["stick", "sphere", "line", "cartoon"])
                    spin = st.checkbox("Spin molecule", value=False)

                with col_view:
                    view = py3Dmol.view()
                    view.addModel(pdb_block, "pdb")
                    view.setStyle({style: {"colorscheme": "CPK"}})
                    if spin:
                        view.spin(True)
                    view.setBackgroundColor("#f8f9fa")
                    showmol(view, height=500, width=800)
            else:
                st.error("3D Conformer generation failed.")

with tab_about:
    st.header("Technical Implementation")
    st.markdown("""
    This project demonstrates a production-grade machine learning pipeline for molecular property prediction:

    1. **Data Pipeline:** Automated ingestion from MoleculeNet, ChEMBL, and PubChem with rigorous chemical standardization using RDKit.
    2. **Modeling:** Graph Neural Networks (GCN, GAT, MPNN) implemented in PyTorch Geometric with uncertainty quantification.
    3. **Evaluation:** Scaffold-based cross-validation and benchmarking against Random Forest/XGBoost baselines.
    4. **MLOps:** Experiment tracking with MLflow, data versioning with DVC, and containerized deployment with FastAPI and Docker.

    **Built for the Eli Lilly Small Molecule AI Engineer Intern Portfolio.**
    """)

st.divider()
st.caption(
    "© 2026 Molecular Property Prediction System · Built with Streamlit, RDKit, and PyTorch Geometric"
)
