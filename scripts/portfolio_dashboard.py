import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io
import os
from molprop.data.standardize import standardize_smiles
from molprop.features.graphs import smiles_to_graph
from molprop.serving.load_model import load_gnn_model

st.set_page_config(
    page_title="Molecular Property Prediction Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
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
""", unsafe_allow_html=True)

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
    "Select Model/Dataset",
    ["BBBP (Blood-Brain Barrier)", "Delaney (Solubility)"]
)

model = get_model("gat", dataset_choice)

# Tabs for different views
tab_pred, tab_viz, tab_about = st.tabs(["🔍 Prediction Explorer", "📊 Chemical Space", "ℹ️ About the Project"])

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
                    else:
                        st.error("Prediction failed.")
                else:
                    st.warning(f"Model weights for {dataset_choice} not found.")
        else:
            st.error("Invalid SMILES.")

with tab_viz:
    st.header("Interactive Chemical Space Explorer")
    st.markdown("Explore the UMAP projection of the dataset. Each point represents a molecule, colored by its property value.")
    
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
            template="plotly_white"
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"UMAP data for {dataset_name} not found. Run `python scripts/visualize_chemical_space.py --dataset {dataset_name}` to generate it.")

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
st.caption("© 2026 Molecular Property Prediction System · Built with Streamlit, RDKit, and PyTorch Geometric")
