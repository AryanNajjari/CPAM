#This script allows you to test pre-trained machine learning models using your own input values. 
#Please update the example_input section with your experimental parameters. 
#Ensure that the input categories (e.g., Target Tissue, Piezoelectric Material, etc.) are consistent with the provided Dataset. 

import joblib
import pandas as pd

# ==== Define the input example ====
example = pd.DataFrame([{
    'Target Tissue': 'Bone',
    'Piezoelectric Material': 'PVDF',
    'Additional Materials': 'Ceramics and Inorganics',
    'Fabrication Method': 'Mixing',
    'Cell Type': 'osteoblasts'
}])

# ==== Load models ====
mlp = joblib.load('MLP_best_model.pkl')
rf = joblib.load('RandomForest_best_model.pkl')
et = joblib.load('ExtraTrees_best_model.pkl')

# ==== Predict with each model ====
mlp_pred = mlp.predict(example)[0]
rf_pred = rf.predict(example)[0]
et_pred = et.predict(example)[0]

# ==== Display results ====
print("Predictions on the example input:")
print(f"MLP Prediction          : {mlp_pred}")
print(f"RandomForest Prediction : {rf_pred}")
print(f"ExtraTrees Prediction   : {et_pred}")
  