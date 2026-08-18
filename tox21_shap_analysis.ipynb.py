#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Machine Learning-Based Prediction of NR-AhR Toxicity Using Molecular Fingerprints
# 
# **Author:** Chukwudum Hillary Uzoh  
# **Dataset:** Tox21 (NR-AhR endpoint)  
# **Objective:** Train and evaluate a Random Forest classifier to predict Aryl 
# hydrocarbon Receptor (NR-AhR) toxicity from molecular structure, using ECFP 
# fingerprints, with SHAP-based interpretability to identify structurally 
# meaningful predictors of toxicity.

# In[ ]:





# ### Stage 1: Environment Setup and Imports

# In[18]:


# Fix for a known Windows OpenMP conflict between MKL, numba, and XGBoost.
# Must be set BEFORE any other imports to take effect.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Use inline backend so plots render directly below each cell
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Suppress DeepChem's verbose import-time messages (optional backend
# notices for TensorFlow/JAX/PyTorch Geometric, which we don't use)
import logging
from contextlib import redirect_stdout, redirect_stderr

logging.getLogger('deepchem').setLevel(logging.ERROR)

with open(os.devnull, 'w') as devnull:
    with redirect_stdout(devnull), redirect_stderr(devnull):
        import deepchem as dc          # Tox21 dataset loading

# Core libraries
import numpy as np                                      # numerical operations
import pandas as pd                                      # tabular data handling
from sklearn.ensemble import RandomForestClassifier      # baseline ML model
from sklearn.metrics import roc_auc_score                 # evaluation metric
import shap                                                # model interpretability

# RDKit, with its internal logger suppressed too
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

print("All libraries imported successfully.")


# In[ ]:





# ### Stage 2: Load the Tox21 Datase

# In[2]:


# Load Tox21 with ECFP fingerprint features and a scaffold-based split
tasks, datasets, transformers = dc.molnet.load_tox21(
    featurizer='ECFP',
    splitter='scaffold'
)
train_dataset, valid_dataset, test_dataset = datasets

# Confirm the data loaded correctly
print("Number of toxicity endpoints (tasks):", len(tasks))
print("Task names:", tasks)
print("Train set shape:", train_dataset.X.shape)
print("Test set shape:", test_dataset.X.shape)


# In[ ]:





# ### Stage 2b: Full 12-Task Benchmark — Random Forest vs. XGBoost

# Trains and evaluates both Random Forest and XGBoost classifiers across  all 12 Tox21 toxicity endpoints (not just NR-AhR), producing the benchmark comparison

# In[3]:


from xgboost import XGBClassifier

# Dictionaries to store ROC-AUC results per task, for each model
rf_results = {}
xgb_results = {}

# Loop through all 12 toxicity endpoints
for i, task in enumerate(tasks):
    # --- Extract labels and weights for this specific task ---
    y_train_t, w_train_t = train_dataset.y[:, i], train_dataset.w[:, i]
    y_test_t, w_test_t = test_dataset.y[:, i], test_dataset.w[:, i]

    # --- Mask out compounds with no label for this task ---
    train_mask_t = w_train_t != 0
    test_mask_t = w_test_t != 0

    X_train_t = train_dataset.X[train_mask_t]
    y_train_t = y_train_t[train_mask_t]
    X_test_t = test_dataset.X[test_mask_t]
    y_test_t = y_test_t[test_mask_t]

    # --- Train Random Forest on this task ---
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_t, y_train_t)
    rf_auc = roc_auc_score(y_test_t, rf.predict_proba(X_test_t)[:, 1])
    rf_results[task] = rf_auc

    # --- Train XGBoost on this task ---
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        eval_metric='logloss', random_state=42, n_jobs=-1
    )
    xgb.fit(X_train_t, y_train_t)
    xgb_auc = roc_auc_score(y_test_t, xgb.predict_proba(X_test_t)[:, 1])
    xgb_results[task] = xgb_auc

    print(f"{task}: RF={rf_auc:.3f}  XGB={xgb_auc:.3f}")

# --- Compile results into a comparison table ---
benchmark_df = pd.DataFrame({
    'Task': tasks,
    'RF_ROC_AUC': [rf_results[t] for t in tasks],
    'XGB_ROC_AUC': [xgb_results[t] for t in tasks]
})
benchmark_df['Difference'] = benchmark_df['XGB_ROC_AUC'] - benchmark_df['RF_ROC_AUC']
benchmark_df.loc['Mean'] = ['Mean', benchmark_df['RF_ROC_AUC'].mean(),
                              benchmark_df['XGB_ROC_AUC'].mean(),
                              benchmark_df['Difference'].mean()]

# Save to CSV for easy reuse in the paper
benchmark_df.to_csv('tox21_12task_benchmark.csv', index=False)

print("\n" + benchmark_df.to_string(index=False))


# In[ ]:





# In[ ]:





# ### Stage 3: Isolate the NR-AhR Task

# 
# Tox21 is multi-task; not every compound has a valid label for every one  of the 12 toxicity endpoints. This stage extracts only the NR-AhR column and removes compounds with no valid label for it (marked by a weight of 0),  leaving a clean, task-specific dataset.

# In[4]:


# Locate NR-AhR's position among the 12 tasks
task_name = 'NR-AhR'
task_idx = tasks.index(task_name)
print(f"'{task_name}' is task index {task_idx} of {len(tasks)}")

# Extract labels (y) and weights (w) for this task only
# w = 0 means "no label available" -> must be excluded
# w = 1 means "valid label" -> keep it
y_train_full = train_dataset.y[:, task_idx]
w_train_full = train_dataset.w[:, task_idx]
y_test_full = test_dataset.y[:, task_idx]
w_test_full = test_dataset.w[:, task_idx]

# Build masks that keep only compounds with a valid label
train_mask = w_train_full != 0
test_mask = w_test_full != 0

# Apply masks to both features (X) and labels (y)
X_train = train_dataset.X[train_mask]
y_train = y_train_full[train_mask]
X_test = test_dataset.X[test_mask]
y_test = y_test_full[test_mask]

print("Training samples with valid NR-AhR label:", X_train.shape[0])
print("Test samples with valid NR-AhR label:", X_test.shape[0])


# In[ ]:





# ### Stage 4: Train and Evaluate Random Forest Baseline

# Trains a Random Forest classifier on NR-AhR-labeled compounds using their ECFP fingerprints as input features. Evaluated with ROC-AUC — the standard metric for Tox21, chosen because toxic compounds are a minority class  within the dataset.

# In[5]:


# Create the Random Forest model
# n_estimators=200 -> number of decision trees in the forest
# random_state=42 -> ensures reproducible results
# n_jobs=-1 -> uses all available CPU cores for faster training
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

# Train on the NR-AhR training data
clf.fit(X_train, y_train)
print("Random Forest training complete.")

# Get predicted probabilities for the "toxic" class (column index 1)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# ROC-AUC measures how well the model ranks toxic compounds above 
# non-toxic ones, independent of any specific classification threshold
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"NR-AhR Random Forest ROC-AUC: {auc_score:.3f}")


# In[ ]:





# ### Stage 5: Compute SHAP Values

# Computes SHAP (SHapley Additive exPlanations) values, which quantify how  much each fingerprint bit (molecular substructure) contributes to each individual prediction. A small sample size (50 compounds) is used to keep memory usage low and avoid crashes on limited-RAM systems.

# In[6]:


# Build the SHAP explainer for tree-based models
explainer = shap.TreeExplainer(clf)

# Use a small sample of the test set to keep computation light
sample_size = min(50, X_test.shape[0])
X_sample = X_test[:sample_size]

# Compute SHAP values
# check_additivity=False -> disables a strict internal check that fails
# due to floating-point rounding in sklearn's RF, not an actual error
shap_values = explainer.shap_values(X_sample, check_additivity=False)

# Array shape is (samples, features, classes) -> select class 1 ("toxic")
shap_values_toxic = shap_values[:, :, 1]

print("SHAP values computed. Shape:", shap_values_toxic.shape)


# In[ ]:





# ### Stage 6: Visualize Feature Importance

# In[7]:


# Save a backup file before attempting inline display
shap.summary_plot(shap_values_toxic, X_sample, show=False)
plt.savefig('shap_summary_nrahr.png', dpi=150, bbox_inches='tight')
print("Backup saved to shap_summary_nrahr.png")

# Display inline
plt.show()


# In[ ]:





# ### Stage 7a: Check Compound Identifier Format

# Before we can map SHAP-important fingerprint bits back to actual chemical substructures, we need the original SMILES strings for each compound. DeepChem stores these as `.ids` on the dataset object, this cell confirms their exact format before we build the substructure-mapping code around it.

# In[8]:


# Check the format of compound identifiers stored alongside the fingerprint data
sample_id = test_dataset.ids[0]
print("Sample identifier:", sample_id)
print("Type:", type(sample_id))

# Confirm the same identifiers line up with our NR-AhR-filtered test set
test_ids_filtered = test_dataset.ids[test_mask]
print("\nNumber of filtered NR-AhR test identifiers:", len(test_ids_filtered))
print("First filtered identifier:", test_ids_filtered[0])


# In[ ]:





# ### Stage 7b: Map Top SHAP Bits to Chemical Substructures

# Regenerates ECFP fingerprints using RDKit directly (rather than DeepChem's featurizer), with bit-tracking enabled — this records exactly which atoms in each molecule activate each fingerprint bit. For compounds with multiple components (e.g., salts), only the largest fragment is used, since it represents the primary active molecule. The top SHAP-ranked bits are then drawn as highlighted substructures on real molecules from the test set.

# In[9]:


from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D


# Get the top 5 most important fingerprint bits from Stage 6's ranking
mean_abs_shap = np.abs(shap_values_toxic).mean(axis=0)
top_5_bits = np.argsort(mean_abs_shap)[::-1][:5]
print("Top 5 fingerprint bits to visualize:", top_5_bits)

# For each top bit, find a compound in our sample that actually contains it,
# then draw the substructure that activates that bit
sample_ids = test_ids_filtered[:sample_size]  # identifiers matching X_sample

for bit_id in top_5_bits:
    found = False
    for smiles in sample_ids:
        # Multi-component SMILES (e.g. salts) - keep only the largest fragment
        fragments = smiles.split('.')
        largest_fragment = max(fragments, key=len)
        mol = Chem.MolFromSmiles(largest_fragment)

        if mol is None:
            continue  # skip if RDKit can't parse this fragment

        # Generate ECFP with bitInfo tracking - records which atoms activate which bits
        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, bitInfo=bit_info)

        # Check if this molecule activates our target bit
        if bit_id in bit_info:
            # Draw the substructure that activates this bit
            drawing = Draw.DrawMorganBit(mol, bit_id, bit_info)
            drawing.save(f'substructure_bit_{bit_id}.png')
            print(f"Bit_{bit_id}: substructure found and saved -> substructure_bit_{bit_id}.png")
            found = True
            break

    if not found:
        print(f"Bit_{bit_id}: no activating compound found in this sample")


# In[ ]:





# ### Stage 7c: Display Substructures Inline

# In[10]:


from IPython.display import Image, display
import matplotlib.image as mpimg

# List the saved substructure images in order of SHAP importance
bit_files = [
    ('Bit_910', 'substructure_bit_910.png'),
    ('Bit_446', 'substructure_bit_446.png'),
    ('Bit_783', 'substructure_bit_783.png'),
    ('Bit_787', 'substructure_bit_787.png'),
    ('Bit_830', 'substructure_bit_830.png'),
]

# Display all 5 in a single grid figure
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, (bit_name, filename) in zip(axes, bit_files):
    img = mpimg.imread(filename)
    ax.imshow(img)
    ax.set_title(bit_name)
    ax.axis('off')  # hide axis ticks for a cleaner image grid

plt.tight_layout()
plt.savefig('all_substructures_grid.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:





# ### Stage 8: Repeat Full Pipeline for NR-AR

# In[ ]:





# In[ ]:





# In[11]:


# --- Step 1: Isolate NR-AR task data ---
task_name = 'NR-AR'
task_idx = tasks.index(task_name)

y_train_full = train_dataset.y[:, task_idx]
w_train_full = train_dataset.w[:, task_idx]
y_test_full = test_dataset.y[:, task_idx]
w_test_full = test_dataset.w[:, task_idx]

train_mask = w_train_full != 0
test_mask = w_test_full != 0

X_train = train_dataset.X[train_mask]
y_train = y_train_full[train_mask]
X_test = test_dataset.X[test_mask]
y_test = y_test_full[test_mask]
test_ids_filtered = test_dataset.ids[test_mask]

print(f"'{task_name}' — Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# --- Step 2: Train Random Forest baseline ---
clf_ar = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_ar.fit(X_train, y_train)

y_pred_proba = clf_ar.predict_proba(X_test)[:, 1]
auc_score_ar = roc_auc_score(y_test, y_pred_proba)
print(f"{task_name} Random Forest ROC-AUC: {auc_score_ar:.3f}")

# --- Step 3: Compute SHAP values (small sample, crash-safe) ---
explainer_ar = shap.TreeExplainer(clf_ar)
sample_size = min(50, X_test.shape[0])
X_sample_ar = X_test[:sample_size]
sample_ids_ar = test_ids_filtered[:sample_size]

shap_values_ar = explainer_ar.shap_values(X_sample_ar, check_additivity=False)
shap_values_toxic_ar = shap_values_ar[:, :, 1]

print("SHAP values computed. Shape:", shap_values_toxic_ar.shape)

# --- Step 4: Identify top 5 important bits ---
mean_abs_shap_ar = np.abs(shap_values_toxic_ar).mean(axis=0)
top_5_bits_ar = np.argsort(mean_abs_shap_ar)[::-1][:5]
print(f"\nTop 5 fingerprint bits for {task_name}:", top_5_bits_ar)

# --- Step 5: Draw substructures for each top bit ---
for bit_id in top_5_bits_ar:
    found = False
    for smiles in sample_ids_ar:
        fragments = smiles.split('.')
        largest_fragment = max(fragments, key=len)
        mol = Chem.MolFromSmiles(largest_fragment)

        if mol is None:
            continue

        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, bitInfo=bit_info)

        if bit_id in bit_info:
            drawing = Draw.DrawMorganBit(mol, bit_id, bit_info)
            drawing.save(f'substructure_NRAR_bit_{bit_id}.png')
            print(f"Bit_{bit_id}: substructure saved -> substructure_NRAR_bit_{bit_id}.png")
            found = True
            break

    if not found:
        print(f"Bit_{bit_id}: no activating compound found in this sample")


# In[ ]:





# ### Stage 8b: Display NR-AR Substructures Inline

# In[12]:


# List the saved NR-AR substructure images, in order of SHAP importance
bit_files_ar = [(f'Bit_{b}', f'substructure_NRAR_bit_{b}.png') for b in top_5_bits_ar]

# Display all 5 in a single grid figure
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, (bit_name, filename) in zip(axes, bit_files_ar):
    img = mpimg.imread(filename)
    ax.imshow(img)
    ax.set_title(f"{bit_name} (NR-AR)")
    ax.axis('off')

plt.tight_layout()
plt.savefig('all_substructures_grid_NRAR.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:





# ### Stage 9: Repeat Full Pipeline for SR-p53

# Applies the same pipeline to SR-p53, a stress-response pathway task (DNA damage response), mechanistically distinct from the nuclear 
# receptor tasks (NR-AhR, NR-AR) analyzed so far. This tests whether a different toxicity mechanism produces yet another distinct set of predictive substructures.

# In[13]:


# --- Step 1: Isolate SR-p53 task data ---
task_name = 'SR-p53'
task_idx = tasks.index(task_name)

y_train_full = train_dataset.y[:, task_idx]
w_train_full = train_dataset.w[:, task_idx]
y_test_full = test_dataset.y[:, task_idx]
w_test_full = test_dataset.w[:, task_idx]

train_mask = w_train_full != 0
test_mask = w_test_full != 0

X_train = train_dataset.X[train_mask]
y_train = y_train_full[train_mask]
X_test = test_dataset.X[test_mask]
y_test = y_test_full[test_mask]
test_ids_filtered = test_dataset.ids[test_mask]

print(f"'{task_name}' — Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# --- Step 2: Train Random Forest baseline ---
clf_p53 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_p53.fit(X_train, y_train)

y_pred_proba = clf_p53.predict_proba(X_test)[:, 1]
auc_score_p53 = roc_auc_score(y_test, y_pred_proba)
print(f"{task_name} Random Forest ROC-AUC: {auc_score_p53:.3f}")

# --- Step 3: Compute SHAP values (small sample, crash-safe) ---
explainer_p53 = shap.TreeExplainer(clf_p53)
sample_size = min(50, X_test.shape[0])
X_sample_p53 = X_test[:sample_size]
sample_ids_p53 = test_ids_filtered[:sample_size]

shap_values_p53 = explainer_p53.shap_values(X_sample_p53, check_additivity=False)
shap_values_toxic_p53 = shap_values_p53[:, :, 1]

print("SHAP values computed. Shape:", shap_values_toxic_p53.shape)

# --- Step 4: Identify top 5 important bits ---
mean_abs_shap_p53 = np.abs(shap_values_toxic_p53).mean(axis=0)
top_5_bits_p53 = np.argsort(mean_abs_shap_p53)[::-1][:5]
print(f"\nTop 5 fingerprint bits for {task_name}:", top_5_bits_p53)

# --- Step 5: Draw substructures for each top bit ---
for bit_id in top_5_bits_p53:
    found = False
    for smiles in sample_ids_p53:
        fragments = smiles.split('.')
        largest_fragment = max(fragments, key=len)
        mol = Chem.MolFromSmiles(largest_fragment)

        if mol is None:
            continue

        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, bitInfo=bit_info)

        if bit_id in bit_info:
            drawing = Draw.DrawMorganBit(mol, bit_id, bit_info)
            drawing.save(f'substructure_SRp53_bit_{bit_id}.png')
            print(f"Bit_{bit_id}: substructure saved -> substructure_SRp53_bit_{bit_id}.png")
            found = True
            break

    if not found:
        print(f"Bit_{bit_id}: no activating compound found in this sample")


# In[ ]:





# ### Stage 9b: Display SR-p53 Substructures Inline

# In[14]:


# Loads and displays the saved SR-p53 substructure images in a grid

# List the saved SR-p53 substructure images, in order of SHAP importance
bit_files_p53 = [(f'Bit_{b}', f'substructure_SRp53_bit_{b}.png') for b in top_5_bits_p53]

# Display all 5 in a single grid figure
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, (bit_name, filename) in zip(axes, bit_files_p53):
    img = mpimg.imread(filename)
    ax.imshow(img)
    ax.set_title(f"{bit_name} (SR-p53)")
    ax.axis('off')

plt.tight_layout()
plt.savefig('all_substructures_grid_SRp53.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:





# ### Stage 10: Repeat Full Pipeline for NR-ER-LBD

# Applies the same pipeline to NR-ER-LBD (Estrogen Receptor Ligand Binding Domain), the final task in this comparison set. This task is notable because XGBoost outperformed Random Forest here by the largest margin (+0.095 ROC-AUC) in the original 12-task benchmark. worth keeping in mind when interpreting these results.

# In[15]:


# --- Step 1: Isolate NR-ER-LBD task data ---
task_name = 'NR-ER-LBD'
task_idx = tasks.index(task_name)

y_train_full = train_dataset.y[:, task_idx]
w_train_full = train_dataset.w[:, task_idx]
y_test_full = test_dataset.y[:, task_idx]
w_test_full = test_dataset.w[:, task_idx]

train_mask = w_train_full != 0
test_mask = w_test_full != 0

X_train = train_dataset.X[train_mask]
y_train = y_train_full[train_mask]
X_test = test_dataset.X[test_mask]
y_test = y_test_full[test_mask]
test_ids_filtered = test_dataset.ids[test_mask]

print(f"'{task_name}' — Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# --- Step 2: Train Random Forest baseline ---
clf_erlbd = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_erlbd.fit(X_train, y_train)

y_pred_proba = clf_erlbd.predict_proba(X_test)[:, 1]
auc_score_erlbd = roc_auc_score(y_test, y_pred_proba)
print(f"{task_name} Random Forest ROC-AUC: {auc_score_erlbd:.3f}")

# --- Step 3: Compute SHAP values (small sample, crash-safe) ---
explainer_erlbd = shap.TreeExplainer(clf_erlbd)
sample_size = min(50, X_test.shape[0])
X_sample_erlbd = X_test[:sample_size]
sample_ids_erlbd = test_ids_filtered[:sample_size]

shap_values_erlbd = explainer_erlbd.shap_values(X_sample_erlbd, check_additivity=False)
shap_values_toxic_erlbd = shap_values_erlbd[:, :, 1]

print("SHAP values computed. Shape:", shap_values_toxic_erlbd.shape)

# --- Step 4: Identify top 5 important bits ---
mean_abs_shap_erlbd = np.abs(shap_values_toxic_erlbd).mean(axis=0)
top_5_bits_erlbd = np.argsort(mean_abs_shap_erlbd)[::-1][:5]
print(f"\nTop 5 fingerprint bits for {task_name}:", top_5_bits_erlbd)

# --- Step 5: Draw substructures for each top bit ---
for bit_id in top_5_bits_erlbd:
    found = False
    for smiles in sample_ids_erlbd:
        fragments = smiles.split('.')
        largest_fragment = max(fragments, key=len)
        mol = Chem.MolFromSmiles(largest_fragment)

        if mol is None:
            continue

        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, bitInfo=bit_info)

        if bit_id in bit_info:
            drawing = Draw.DrawMorganBit(mol, bit_id, bit_info)
            drawing.save(f'substructure_NRERLBD_bit_{bit_id}.png')
            print(f"Bit_{bit_id}: substructure saved -> substructure_NRERLBD_bit_{bit_id}.png")
            found = True
            break

    if not found:
        print(f"Bit_{bit_id}: no activating compound found in this sample")


# In[ ]:





# In[16]:


## Stage 10b: Display NR-ER-LBD Substructures Inline


# List the saved NR-ER-LBD substructure images, in order of SHAP importance
bit_files_erlbd = [(f'Bit_{b}', f'substructure_NRERLBD_bit_{b}.png') for b in top_5_bits_erlbd]

# Display all 5 in a single grid figure
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, (bit_name, filename) in zip(axes, bit_files_erlbd):
    img = mpimg.imread(filename)
    ax.imshow(img)
    ax.set_title(f"{bit_name} (NR-ER-LBD)")
    ax.axis('off')

plt.tight_layout()
plt.savefig('all_substructures_grid_NRERLBD.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:





# ### Stage 11: Compile Results Summary

# Combines the Random Forest ROC-AUC scores and top SHAP-identified substructures across all 4 deep-dive tasks (NR-AhR, NR-AR, SR-p53, NR-ER-LBD) into a single summary table, for direct use in the paper's 
# Results section.

# In[17]:


# --- Compile a summary table across all 4 deep-dive tasks ---
summary_data = {
    'Task': ['NR-AhR', 'NR-AR', 'SR-p53', 'NR-ER-LBD'],
    'Biological_Category': ['Nuclear Receptor', 'Nuclear Receptor', 
                              'Stress Response', 'Nuclear Receptor'],
    'RF_ROC_AUC': [0.796, 0.734, 0.752, 0.626],
    'Top_5_SHAP_Bits': [
        str(list(top_5_bits)),        # NR-AhR (from earlier variable)
        str(list(top_5_bits_ar)),     # NR-AR
        str(list(top_5_bits_p53)),    # SR-p53
        str(list(top_5_bits_erlbd))   # NR-ER-LBD
    ],
    'Dominant_Substructure_Theme': [
        'Nitrogen-rich groups (amines, sulfonamide, lactone)',
        'N-heterocycles and carbonyls (imidazole, hydantoin)',
        'Aromatic/electrophilic motifs (nitrile-aromatic, O-heterocycles)',
        'Mixed/noisy (includes spurious organometallic outlier)'
    ]
}

summary_df = pd.DataFrame(summary_data)

# Save as CSV for easy import into a paper draft or supplementary materials
summary_df.to_csv('tox21_4task_summary.csv', index=False)

print(summary_df.to_string(index=False))


# In[ ]:




