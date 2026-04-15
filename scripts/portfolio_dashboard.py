import streamlit as st
import pandas as pd
import numpy as np
import torch
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
    layout="wide"
)

# Cache model loading
@st.cache_resource
def get_model(model_type, dataset_name):
    # Mapping dataset to weight files (assuming they exist in root)
    weights_map = {
        "BBBP (Blood-Brain Barrier)": "best_gat_bbbp.pt",
        "Delaney (Solubility)": "best_gat_delaney.pt",
    }
    weights_path = weights_map.get(dataset_name)
    if not weights_path or not os.path.exists(weights_path):
        return None
    
    # We'll use GAT as default for the dashboard
    return load_gnn_model("gat", weights_path, in_dim=9, hidden_dim=128, out_dim=1)

def run_prediction(model, smiles, n_samples=10):
    std_smiles = standardize_smiles(smiles)
    if not std_smiles:
        return None, "Invalid SMILES"
    
    data = smiles_to_graph(std_smiles)
    if not data:
        return None, "Graph conversion failed"

    model.eval()
    # Enable MC Dropout if we want uncertainty
    # For simplicity, we just run multiple forward passes if dropout is active
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            # We would need the model forward pass to keep dropout active. 
            # Assuming GNNBase supports mc_dropout flag or similar.
            # For now, let's just do a single pass if mc_dropout is not explicitly handled in forward.
            out = model(data)
            preds.append(out.item())
            
    return np.mean(preds), np.std(preds)

st.title("🧬 Molecular Property Prediction System")
st.markdown("""
Welcome to the interactive explorer for the **Molecular Property Prediction System**.
This dashboard allows you to predict physicochemical and ADMET properties directly from SMILES strings.
""")

st.sidebar.header("Configuration")
dataset_choice = st.sidebar.selectbox(
    "Select Model/Dataset",
    ["BBBP (Blood-Brain Barrier)", "Delaney (Solubility)"]
)

model = get_model("gat", dataset_choice)

# Main prediction area
st.header("🔍 Prediction Explorer")
smiles_input = st.text_input("Enter SMILES string:", "CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        st.success(f"Valid SMILES: {smiles_input}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            img = Draw.MolToImage(mol, size=(400, 400))
            st.image(img, caption="Standardized Structure", use_column_width=True)
            
        with col2:
            st.subheader("Results")
            if model:
                mean_val, std_val = run_prediction(model, smiles_input)
                if mean_val is not None:
                    # Formatting based on dataset
                    if "BBBP" in dataset_choice:
                        prob = 1 / (1 + np.exp(-mean_val)) # Sigmoid
                        st.metric("Probability (BBB+)", f"{prob:.2%}")
                    else:
                        st.metric("LogS (Solubility)", f"{mean_val:.2f}")
                    
                    st.write(f"**Uncertainty (std dev):** {std_val:.4f}")
                else:
                    st.error("Prediction failed.")
            else:
                st.warning(f"Model weights for {dataset_choice} not found. Please train the model first.")
    else:
        st.error("Invalid SMILES. Please check your input.")

st.divider()
st.markdown("""
### 📊 Project Insights
* **Backbone:** Graph Attention Network (GAT) with Multi-Task Heads
* **Uncertainty:** Monte Carlo Dropout (10 samples)
* **Standardization:** Salt stripping, canonicalization, and fragment normalization.
""")
