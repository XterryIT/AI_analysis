import pandas as pd
import joblib
import sys
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = 'rf_model_multiclass.joblib'
FEATURES_PATH = 'model_features.joblib'

# PUT YOUR NEW FILE NAME HERE
NEW_DATA_PATH = 'ultimate_sample.csv' 
OUTPUT_PATH = 'final_sample.csv'

def main():
    print("### STARTING TESTING PIPELINE ###")

    # 1. Load Model and Features
    print(f"[1] Loading model and features...")
    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        print("    Model and feature list loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        return

    # 2. Load New Data
    print(f"[2] Loading new data from '{NEW_DATA_PATH}'...")
    try:
        if NEW_DATA_PATH.endswith('.csv'):
            df = pd.read_csv(NEW_DATA_PATH)
        else:
            df = pd.read_excel(NEW_DATA_PATH)
        
        # Fill NaN just in case
        df.fillna(0, inplace=True)
        
    except FileNotFoundError:
        print(f"ERROR: Data file '{NEW_DATA_PATH}' not found.")
        return

    # 3. Align Columns (Crucial for ML)
    # The model expects specific columns in a specific order.
    # We filter the new dataframe to match the training features exactly.
    print("[3] Aligning columns...")
    
    try:
        # Create a new DF with only the required columns, fill missing with 0
        X_new = pd.DataFrame()
        for col in feature_names:
            if col in df.columns:
                X_new[col] = df[col]
            else:
                # If a column from training is missing in new data, fill with 0
                X_new[col] = 0
        
        # Ensure order matches exactly
        X_new = X_new[feature_names]
        
    except Exception as e:
        print(f"Error preparing columns: {e}")
        return

    # 4. Predict
    print("[4] Running prediction...")
    predictions = model.predict(X_new)
    
    # If possible, get probabilities (confidence)
    try:
        probs = model.predict_proba(X_new)
        max_probs = np.max(probs, axis=1) # Confidence of the prediction
    except:
        max_probs = None

    # 5. Save Results
    print(f"[5] Saving results to '{OUTPUT_PATH}'...")
    
    # We attach predictions to the ORIGINAL dataframe so you can see the context
    results = df.copy()
    results['Predicted_Label'] = predictions
    
    # Map back to names for readability if you want
    label_map = {0: 'Non-DoH', 1: 'Benign', 2: 'Malicious'}
    results['Label_Name'] = results['Predicted_Label'].map(label_map)

    if max_probs is not None:
        results['Confidence'] = max_probs

    results.to_csv(OUTPUT_PATH, index=False)
    
    print("\n" + "="*40)
    print("PREDICTION SUMMARY:")
    print(results['Label_Name'].value_counts())
    print("="*40)
    print("Process finished successfully.")

if __name__ == "__main__":
    main()