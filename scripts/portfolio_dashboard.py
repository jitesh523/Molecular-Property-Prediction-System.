import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io

st.set_page_config(
    page_title="Molecular Property Prediction Explorer",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Molecular Property Prediction System")
st.markdown("""
Welcome to the interactive explorer for the **Molecular Property Prediction System**.
This dashboard allows you to predict physicochemical and ADMET properties directly from SMILES strings.
""")

st.sidebar.header("Configuration")
dataset_choice = st.sidebar.selectbox(
    "Select Model/Dataset",
    ["BBBP (Blood-Brain Barrier)", "Delaney (Solubility)", "Lipo (Lipophilicity)"]
)

# Main prediction area
st.header("🔍 Prediction Explorer")
smiles_input = st.text_input("Enter SMILES string:", "CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin as default

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        st.success("Valid SMILES detected!")
        # Render Molecule
        img = Draw.MolToImage(mol, size=(300, 300))
        st.image(img, caption="Chemical Structure")
        
        # Placeholder for predictions
        st.info("Model inference integrated in next step...")
    else:
        st.error("Invalid SMILES. Please check your input.")

st.divider()
st.markdown("Built for the **Eli Lilly Small Molecule AI Engineer Intern** Portfolio.")
