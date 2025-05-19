import os
import pandas as pd
import numpy as np
import re # For extracting info from filenames
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier # Use Classifier for all classification tasks 
from sklearn.metrics import classification_report, confusion_matrix 
import joblib # To save/load models and encoders

# --- Configuration ---
DATA_DIR = r"C:\Users\siddh\OneDrive\Desktop\Train_Data" # Use your actual path
LINE_FOLDERS = [
    'L4_5 combined faults',
    'L4_6 combined faults',
    'L5_7 combined faults',
    'L6_9 combined faults',
    'L7_8 combined faults'
]
FAULT_START_TIME = 3.0 # Time in seconds when the fault is initiated
# *** NEW: Define a window duration after FAULT_START_TIME to label as "fault" ***
FAULT_WINDOW_DURATION = 0.1 # e.g., 0.1 seconds (3.0s to 3.1s is fault) - Adjust as needed

# Define clear column names for easier access 
COLUMN_MAPPING = {
    'Time in s': 'Time_s',
    'Phase Voltage A/Terminal i in kV': 'Va_i_kV',
    'Phase Voltage B/Terminal i in kV': 'Vb_i_kV',
    'Phase Voltage C/Terminal i in kV': 'Vc_i_kV',
    'Phase Voltage A/Terminal j in kV': 'Va_j_kV',
    'Phase Voltage B/Terminal j in kV': 'Vb_j_kV',
    'Phase Voltage C/Terminal j in kV': 'Vc_j_kV',
    'Phase Current A/Terminal i in kA': 'Ia_i_kA',
    'Phase Current B/Terminal i in kA': 'Ib_i_kA',
    'Phase Current C/Terminal i in kA': 'Ic_i_kA',
    'Phase Current A/Terminal j in kA': 'Ia_j_kA',
    'Phase Current B/Terminal j in kA': 'Ib_j_kA',
    'Phase Current C/Terminal j in kA': 'Ic_j_kA' 
}


INPUT_FEATURES = [
    'Va_i_kV', 'Vb_i_kV', 'Vc_i_kV', 'Ia_i_kA', 'Ib_i_kA', 'Ic_i_kA',
    'Va_j_kV', 'Vb_j_kV', 'Vc_j_kV', 'Ia_j_kA', 'Ib_j_kA', 'Ic_j_kA'
]

TARGET_FAULT_PRESENCE = 'Target_Fault_Presence' # Binary (0 or 1)
TARGET_LINE = 'Target_Line'                     # Categorical (e.g., '4_5')
TARGET_FAULT_TYPE = 'Target_Fault_Type'         # Categorical (e.g., 'LG', 'LLG', 'LLL')
TARGET_LOCATION = 'Target_Fault_Location_Percent' # Categorical (1, 10, 50, 90, 99)

# --- Step 1: Data Loading and Preprocessing ---

def load_and_process_data(data_dir, line_folders, fault_start_time, fault_window_duration):
    """Loads data from CSVs, extracts targets, defines fault window, and combines."""
    all_data = []
    processed_files = 0
    error_files = []
    fault_end_time = fault_start_time + fault_window_duration # Calculate end of fault window

    for line_folder_name in line_folders:
        line_folder_path = os.path.join(data_dir, line_folder_name)
        if not os.path.isdir(line_folder_path):
            print(f"Warning: Directory not found {line_folder_path}")
            continue

        line_match = re.match(r'L(\d+_\d+)', line_folder_name)
        if not line_match:
            print(f"Warning: Could not extract line number from folder '{line_folder_name}'")
            line_number = 'Unknown'
        else:
            line_number = line_match.group(1)

        print(f"Processing folder: {line_folder_name} (Line: {line_number})")

        for filename in os.listdir(line_folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(line_folder_path, filename)

                file_match = re.match(r'([A-Z]+)_(\d+)\.csv', filename, re.IGNORECASE)
                if not file_match:
                    print(f"  Warning: Could not parse filename '{filename}'. Skipping.")
                    error_files.append(file_path)
                    continue

                fault_type = file_match.group(1).upper()
                try:
                    # Location is now treated as a category, but still read as int initially
                    fault_location = int(file_match.group(2))
                    # Ensuring only expected locations are processed if needed (optional)
                    # if fault_location not in [1, 10, 50, 90, 99]:
                    #    print(f"  Warning: Unexpected fault location {fault_location} in '{filename}'. Skipping.")
                    #    error_files.append(file_path)
                    #    continue
                except ValueError:
                    print(f"  Warning: Could not parse fault location from '{filename}'. Skipping.")
                    error_files.append(file_path)
                    continue

                try:
                    df = pd.read_csv(file_path)

                    if len(df.columns) >= 13:
                        df = df.iloc[:, :13]
                        df.columns = list(COLUMN_MAPPING.keys())
                        df = df.rename(columns=COLUMN_MAPPING)
                    else:
                        print(f"  Warning: File '{filename}' has fewer than 13 columns ({len(df.columns)}). Skipping.")
                        error_files.append(file_path)
                        continue

                    # *** Fault Presence Logic: Define fault only within the window ***
                    df[TARGET_FAULT_PRESENCE] = (
                        (df['Time_s'] >= fault_start_time) &
                        (df['Time_s'] < fault_end_time) # Use '<' to make window exclusive of end time? Or <= inclusive. Let's use <=
                        # (df['Time_s'] <= fault_end_time)
                    ).astype(int)


                    df[TARGET_LINE] = line_number
                    df[TARGET_FAULT_TYPE] = fault_type
                    # Store location as is for now, will encode later
                    df[TARGET_LOCATION] = fault_location

                    all_data.append(df)
                    processed_files += 1

                except Exception as e:
                    print(f"  Error reading or processing file {filename}: {e}")
                    error_files.append(file_path)

    if not all_data:
        raise ValueError("No data files were successfully processed. Check paths and file formats.")

    print(f"\nSuccessfully processed {processed_files} files.")
    if error_files:
        print("Files with errors or skipped:")
        for f in error_files:
            print(f"  - {f}")

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined_df)}")
    print(f"Columns in combined DataFrame: {combined_df.columns.tolist()}")

    # Check distribution of the fault presence label
    print("\nFault Presence Label Distribution (Window Definition):")
    print(combined_df[TARGET_FAULT_PRESENCE].value_counts(normalize=True) * 100)


    # Basic data cleaning
    # Checks for NaNs before dropping
    nan_rows = combined_df.isnull().any(axis=1).sum()
    if nan_rows > 0:
        print(f"Warning: Found {nan_rows} rows with NaN values. Dropping them.")
        combined_df = combined_df.dropna()
        print(f"Rows after dropping NA: {len(combined_df)}")
    else:
        print("No NaN values found.")


    return combined_df

# Load the data with the new fault window logic
raw_data = load_and_process_data(DATA_DIR, LINE_FOLDERS, FAULT_START_TIME, FAULT_WINDOW_DURATION)

# --- Step 2: Feature Engineering and Data Splitting ---

# Define features (X) and targets (y)
X = raw_data[INPUT_FEATURES]
y_presence = raw_data[TARGET_FAULT_PRESENCE]
y_line = raw_data[TARGET_LINE]
y_type = raw_data[TARGET_FAULT_TYPE]
y_location = raw_data[TARGET_LOCATION] # Keep original location values here for encoding

# Encode categorical targets
line_encoder = LabelEncoder()
type_encoder = LabelEncoder()
location_encoder = LabelEncoder() # *** Encoder for location ***

# Fit encoders on the entire dataset's target columns before splitting
y_line_encoded = line_encoder.fit_transform(y_line)
y_type_encoded = type_encoder.fit_transform(y_type)
# *** Encode location categories (e.g., 1, 10, 50, 90, 99 -> 0, 1, 2, 3, 4) ***
y_location_encoded = location_encoder.fit_transform(y_location)

# Save encoders for later use (decoding predictions)
joblib.dump(line_encoder, 'line_encoder.joblib')
joblib.dump(type_encoder, 'type_encoder.joblib')
joblib.dump(location_encoder, 'location_encoder.joblib') # *** Save location encoder ***

# Split data into training and testing sets
# Stratify by fault presence is now even more important due to imbalance
X_train, X_test, \
y_presence_train, y_presence_test, \
y_line_train, y_line_test, \
y_type_train, y_type_test, \
y_location_train, y_location_test, \
time_train, time_test = train_test_split(
    X,
    y_presence,
    y_line_encoded,     # Use encoded targets for training
    y_type_encoded,     # Use encoded targets for training
    y_location_encoded, # *** Use encoded location for training ***
    raw_data['Time_s'], # Keep time column separate for final output
    test_size=0.25,     # 25% for testing
    random_state=42,    # for reproducibility
    stratify=y_presence # Stratify based on the sparse fault presence label
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
print("\nFault Presence distribution in Training Set:")
print(pd.Series(y_presence_train).value_counts(normalize=True)*100)
print("\nFault Presence distribution in Test Set:")
print(pd.Series(y_presence_test).value_counts(normalize=True)*100)


# Scale input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use transform only on test data

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

# --- Step 3: Model Training (Using Classifiers for All Tasks) ---

print("\n--- Training Models ---")
# Define default classifier settings (can be tuned)
classifier_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}

# 1. Fault Presence Model (Binary Classification)
print("Training Fault Presence Classifier...")
# Use class_weight='balanced' due to expected high imbalance with the window definition
presence_clf = RandomForestClassifier(**classifier_params, class_weight='balanced')
presence_clf.fit(X_train_scaled, y_presence_train)
joblib.dump(presence_clf, 'presence_classifier.joblib')
print("Fault Presence Classifier Trained.")

# 2. Line Identification Model (Multi-class Classification)
print("Training Line Identifier...")
line_clf = RandomForestClassifier(**classifier_params)
line_clf.fit(X_train_scaled, y_line_train) # Train on encoded labels
joblib.dump(line_clf, 'line_classifier.joblib')
print("Line Identifier Trained.")

# 3. Fault Type Classification Model (Multi-class Classification)
print("Training Fault Type Classifier...")
type_clf = RandomForestClassifier(**classifier_params)
type_clf.fit(X_train_scaled, y_type_train) # Train on encoded labels
joblib.dump(type_clf, 'type_classifier.joblib')
print("Fault Type Classifier Trained.")

# 4. Fault Location Model (Multi-class Classification) 
print("Training Fault Location Classifier...")
location_clf = RandomForestClassifier(**classifier_params) # Use Classifier
location_clf.fit(X_train_scaled, y_location_train) # Train on encoded location labels
joblib.dump(location_clf, 'location_classifier.joblib') # Save classifier
print("Fault Location Classifier Trained.")

print("--- Model Training Complete ---")

# --- Step 4: Prediction on Test Set ---

print("\n--- Evaluating Models on Test Set ---")

# Predict using trained models
presence_pred = presence_clf.predict(X_test_scaled)
line_pred_encoded = line_clf.predict(X_test_scaled)
type_pred_encoded = type_clf.predict(X_test_scaled)
location_pred_encoded = location_clf.predict(X_test_scaled) # *** Predict encoded location ***

# Decode categorical predictions
line_pred = line_encoder.inverse_transform(line_pred_encoded)
type_pred = type_encoder.inverse_transform(type_pred_encoded)
location_pred = location_encoder.inverse_transform(location_pred_encoded) # *** Decode location ***

# --- Step 5: Evaluation (Using Classification Metrics for All) ---

print("\nFault Presence Classification Report:")
print(classification_report(y_presence_test, presence_pred, target_names=['No Fault (0)', 'Fault (1)']))
print("Confusion Matrix:")
print(confusion_matrix(y_presence_test, presence_pred))

print("\nLine Identification Classification Report:")
# Decode y_line_test for reporting
y_line_test_decoded = line_encoder.inverse_transform(y_line_test)
print(classification_report(y_line_test_decoded, line_pred, labels=line_encoder.classes_, target_names=line_encoder.classes_))

print("\nFault Type Classification Report:")
# Decode y_type_test for reporting
y_type_test_decoded = type_encoder.inverse_transform(y_type_test)
print(classification_report(y_type_test_decoded, type_pred, labels=type_encoder.classes_, target_names=type_encoder.classes_))

print("\nFault Location Classification Report:")
# Decode y_location_test for reporting
y_location_test_decoded = location_encoder.inverse_transform(y_location_test)
# Ensure labels and target_names are sorted correctly if needed, LabelEncoder classes_ should be sorted
location_classes_str = [str(c) for c in location_encoder.classes_] # Convert numeric labels to strings for report names
print(classification_report(y_location_test_decoded, location_pred, labels=location_encoder.classes_, target_names=location_classes_str))


# --- Step 6: Format Final Output Table ---

print("\n--- Generating Final Test Output Table ---")

# Create a DataFrame with the original test features
results_df = X_test.copy()
results_df['Time_s'] = time_test # Add the time column back

# Add predicted target columns
results_df['Predicted_Fault_Presence'] = presence_pred
results_df['Predicted_Line'] = line_pred
results_df['Predicted_Fault_Type'] = type_pred
# *** Location prediction is now the decoded category ***
results_df['Predicted_Fault_Location_Percent'] = location_pred

# Reorder columns to match the requested format:
# Time, 6x V/I Terminal i, 6x V/I Terminal j, 4x Target Predictions
output_columns = [
    'Time_s',
    'Va_i_kV', 'Vb_i_kV', 'Vc_i_kV', 'Ia_i_kA', 'Ib_i_kA', 'Ic_i_kA', # Terminal i
    'Va_j_kV', 'Vb_j_kV', 'Vc_j_kV', 'Ia_j_kA', 'Ib_j_kA', 'Ic_j_kA', # Terminal j
    'Predicted_Fault_Presence', 'Predicted_Line', 'Predicted_Fault_Type', 'Predicted_Fault_Location_Percent' # Predictions
]
# Ensure all expected columns exist before reordering
missing_cols = [col for col in output_columns if col not in results_df.columns]
if missing_cols:
    raise KeyError(f"Columns missing for final output: {missing_cols}. Check COLUMN_MAPPING and INPUT_FEATURES.")

final_output_df = results_df[output_columns]

# Display the first few rows of the final table
print("\nFinal Output Table (Test Set Predictions) - First 20 rows:")
print(final_output_df.head(20))

# Optionally, save the results to a CSV
output_csv_path = 'final_test_predictions_classified_location_windowed_fault.csv'
final_output_df.to_csv(output_csv_path, index=False)
print(f"\nFull final output table saved to: {output_csv_path}")
