"""
SIMPLIFIED BUG CLASSIFICATION MODEL
Direct code execution without functions or pipelines
Clear sections for each preprocessing step
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BUG CLASSIFICATION MODEL - SIMPLIFIED VERSION")
print("=" * 80)

# ============================================================================
# SECTION 1: LOAD THE DATA
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: LOADING DATA FROM EXCEL")
print("=" * 80)

# CHANGE THIS PATH TO YOUR EXCEL FILE
excel_file_path = 'bug_data.xlsx'  # <-- UPDATE THIS WITH YOUR FILE PATH

# Read the Excel file
df = pd.read_excel(excel_file_path)

# Display basic information
print(f"\n✅ Loaded {len(df)} rows from Excel")
print(f"📋 Columns found: {list(df.columns)}")

# Remove any extra spaces from column names
df.columns = df.columns.str.strip()

# Check the first few rows
print("\n📊 First 5 rows of data:")
print(df.head())

# ============================================================================
# SECTION 2: DATA CLEANING AND VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: DATA CLEANING AND VALIDATION")
print("=" * 80)

# Check for missing values
print("\n🔍 Checking for missing values:")
print(df[['description', 'Rootcause', 'Fixed by']].isnull().sum())

# Remove rows with missing values in critical columns
initial_count = len(df)
df = df.dropna(subset=['description', 'Rootcause', 'Fixed by'])
final_count = len(df)

print(f"\n✅ Removed {initial_count - final_count} rows with missing values")
print(f"📊 Final dataset size: {final_count} rows")

# Display unique values
print(f"\n📈 Unique Root Causes: {df['Rootcause'].nunique()}")
print(f"   Categories: {df['Rootcause'].unique()[:10]}")  # Show first 10

print(f"\n👥 Unique Fix Teams: {df['Fixed by'].nunique()}")
print(f"   Teams: {df['Fixed by'].unique()[:10]}")  # Show first 10

# ============================================================================
# SECTION 3: TEXT PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: TEXT PREPROCESSING")
print("=" * 80)

# Create a copy of descriptions for processing
descriptions = df['description'].copy()

print("\n📝 Original description example:")
print(f"   '{descriptions.iloc[0]}'")

# Step 3.1: Convert to lowercase
print("\n3.1 Converting to lowercase...")
descriptions = descriptions.str.lower()

# Step 3.2: Remove special characters (keep only letters, numbers, and spaces)
print("3.2 Removing special characters...")
descriptions = descriptions.str.replace(r'[^a-z0-9\s]', ' ', regex=True)

# Step 3.3: Remove extra whitespace
print("3.3 Removing extra whitespace...")
descriptions = descriptions.str.replace(r'\s+', ' ', regex=True)
descriptions = descriptions.str.strip()

# Save cleaned descriptions
df['description_clean'] = descriptions

print("\n📝 Cleaned description example:")
print(f"   '{df['description_clean'].iloc[0]}'")

# ============================================================================
# SECTION 4: FEATURE EXTRACTION (TEXT TO NUMBERS)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: FEATURE EXTRACTION - CONVERTING TEXT TO NUMBERS")
print("=" * 80)

# Initialize TF-IDF Vectorizer
print("\n🔤 Creating TF-IDF Vectorizer...")
print("   Parameters:")
print("   - Max features: 500 (top 500 most important words)")
print("   - N-gram range: (1,3) (single words, pairs, and triplets)")
print("   - Remove English stop words: Yes")

vectorizer = TfidfVectorizer(
    max_features=500,        # Use top 500 features
    ngram_range=(1, 3),      # Use 1, 2, and 3 word combinations
    stop_words='english',    # Remove common English words
    min_df=2,               # Word must appear in at least 2 documents
    max_df=0.95             # Word can't appear in more than 95% of documents
)

# Transform descriptions to numerical features
X = vectorizer.fit_transform(df['description_clean'])

print(f"\n✅ Text converted to matrix of shape: {X.shape}")
print(f"   ({X.shape[0]} bugs × {X.shape[1]} features)")

# Show some extracted features (words/phrases)
feature_names = vectorizer.get_feature_names_out()
print(f"\n📊 Sample features extracted: {feature_names[:10]}")

# ============================================================================
# SECTION 5: ENCODE TARGET VARIABLES
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: ENCODING TARGET VARIABLES")
print("=" * 80)

# Create label encoders for converting text labels to numbers
print("\n🏷️ Encoding Root Cause labels...")
root_cause_encoder = LabelEncoder()
y_rootcause = root_cause_encoder.fit_transform(df['Rootcause'])

print("   Root Cause mapping:")
for i, label in enumerate(root_cause_encoder.classes_[:5]):  # Show first 5
    print(f"   '{label}' → {i}")

print("\n🏷️ Encoding Fix Team labels...")
fix_team_encoder = LabelEncoder()
y_fixteam = fix_team_encoder.fit_transform(df['Fixed by'])

print("   Fix Team mapping:")
for i, label in enumerate(fix_team_encoder.classes_[:5]):  # Show first 5
    print(f"   '{label}' → {i}")

# ============================================================================
# SECTION 6: SPLIT DATA INTO TRAINING AND TESTING
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: SPLITTING DATA FOR TRAINING AND TESTING")
print("=" * 80)

print("\n📊 Splitting data: 80% for training, 20% for testing")

# Split for Root Cause prediction
X_train, X_test, y_train_root, y_test_root = train_test_split(
    X, y_rootcause, 
    test_size=0.2, 
    random_state=42,
    stratify=y_rootcause  # Ensure balanced split
)

# Split for Fix Team prediction (using same X splits)
_, _, y_train_team, y_test_team = train_test_split(
    X, y_fixteam, 
    test_size=0.2, 
    random_state=42,
    stratify=y_fixteam
)

print(f"\n✅ Training set: {X_train.shape[0]} bugs")
print(f"✅ Testing set: {X_test.shape[0]} bugs")

# ============================================================================
# SECTION 7: TRAIN RANDOM FOREST MODEL FOR ROOT CAUSE
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: TRAINING ROOT CAUSE PREDICTION MODEL")
print("=" * 80)

print("\n🌲 Creating Random Forest Classifier for Root Cause...")
print("   Parameters:")
print("   - Number of trees: 100")
print("   - Max depth: 20")
print("   - Random state: 42 (for reproducibility)")

# Create and train the model
root_cause_model = RandomForestClassifier(
    n_estimators=100,    # Number of trees in the forest
    max_depth=20,        # Maximum depth of trees
    min_samples_split=5, # Minimum samples to split a node
    random_state=42,     # For reproducibility
    n_jobs=-1           # Use all CPU cores
)

print("\n⏳ Training Root Cause model...")
root_cause_model.fit(X_train, y_train_root)
print("✅ Root Cause model trained!")

# Make predictions on test set
root_predictions = root_cause_model.predict(X_test)

# Calculate accuracy
root_accuracy = accuracy_score(y_test_root, root_predictions)
print(f"\n🎯 Root Cause Prediction Accuracy: {root_accuracy:.2%}")

# ============================================================================
# SECTION 8: TRAIN RANDOM FOREST MODEL FOR FIX TEAM
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 8: TRAINING FIX TEAM PREDICTION MODEL")
print("=" * 80)

print("\n🌲 Creating Random Forest Classifier for Fix Team...")

# Create and train the model
fix_team_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

print("\n⏳ Training Fix Team model...")
fix_team_model.fit(X_train, y_train_team)
print("✅ Fix Team model trained!")

# Make predictions on test set
team_predictions = fix_team_model.predict(X_test)

# Calculate accuracy
team_accuracy = accuracy_score(y_test_team, team_predictions)
print(f"\n🎯 Fix Team Prediction Accuracy: {team_accuracy:.2%}")

# ============================================================================
# SECTION 9: DETAILED PERFORMANCE EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 9: DETAILED PERFORMANCE EVALUATION")
print("=" * 80)

# Decode predictions back to original labels
root_pred_labels = root_cause_encoder.inverse_transform(root_predictions)
root_true_labels = root_cause_encoder.inverse_transform(y_test_root)

team_pred_labels = fix_team_encoder.inverse_transform(team_predictions)
team_true_labels = fix_team_encoder.inverse_transform(y_test_team)

print("\n📊 ROOT CAUSE CLASSIFICATION REPORT")
print("-" * 50)
print(classification_report(root_true_labels, root_pred_labels))

print("\n📊 FIX TEAM CLASSIFICATION REPORT")
print("-" * 50)
print(classification_report(team_true_labels, team_pred_labels))

# ============================================================================
# SECTION 10: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 10: FEATURE IMPORTANCE - WHICH WORDS MATTER MOST?")
print("=" * 80)

# Get feature importance from Root Cause model
root_importance = root_cause_model.feature_importances_

# Create dataframe of features and their importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': root_importance
})

# Sort by importance
importance_df = importance_df.sort_values('importance', ascending=False)

print("\n🔝 Top 15 Most Important Features for Root Cause Prediction:")
print("-" * 50)
for idx, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']:30s} → {row['importance']:.4f}")

# ============================================================================
# SECTION 11: SAVE THE TRAINED MODELS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 11: SAVING TRAINED MODELS")
print("=" * 80)

# Create a dictionary with all necessary components
model_package = {
    'vectorizer': vectorizer,
    'root_cause_model': root_cause_model,
    'fix_team_model': fix_team_model,
    'root_cause_encoder': root_cause_encoder,
    'fix_team_encoder': fix_team_encoder,
    'feature_names': feature_names,
    'accuracy_scores': {
        'root_cause': root_accuracy,
        'fix_team': team_accuracy
    }
}

# Save to file
model_filename = 'trained_bug_classifier.pkl'
joblib.dump(model_package, model_filename)
print(f"\n💾 Models saved to '{model_filename}'")

# ============================================================================
# SECTION 12: TEST PREDICTIONS ON NEW BUGS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 12: TESTING PREDICTIONS ON NEW BUG DESCRIPTIONS")
print("=" * 80)

# Example new bug descriptions to test
new_bugs = [
    "Database connection timeout during peak hours",
    "Login button not responding on mobile app",
    "API returns 500 error when processing payment",
    "Customer data not syncing between systems",
    "Search functionality showing duplicate results"
]

print("\n🔮 Making predictions for new bugs:")
print("-" * 80)

for bug_description in new_bugs:
    # Clean the text (same preprocessing as training)
    clean_bug = bug_description.lower()
    clean_bug = pd.Series([clean_bug]).str.replace(r'[^a-z0-9\s]', ' ', regex=True)[0]
    clean_bug = pd.Series([clean_bug]).str.replace(r'\s+', ' ', regex=True)[0].strip()
    
    # Convert to features using the trained vectorizer
    bug_features = vectorizer.transform([clean_bug])
    
    # Predict root cause
    root_pred_encoded = root_cause_model.predict(bug_features)[0]
    root_pred_proba = root_cause_model.predict_proba(bug_features)[0]
    root_pred_label = root_cause_encoder.inverse_transform([root_pred_encoded])[0]
    root_confidence = max(root_pred_proba) * 100
    
    # Predict fix team
    team_pred_encoded = fix_team_model.predict(bug_features)[0]
    team_pred_proba = fix_team_model.predict_proba(bug_features)[0]
    team_pred_label = fix_team_encoder.inverse_transform([team_pred_encoded])[0]
    team_confidence = max(team_pred_proba) * 100
    
    # Display results
    print(f"\n🐛 Bug: '{bug_description}'")
    print(f"   → Root Cause: {root_pred_label} (Confidence: {root_confidence:.1f}%)")
    print(f"   → Fix Team: {team_pred_label} (Confidence: {team_confidence:.1f}%)")

# ============================================================================
# SECTION 13: CREATE PREDICTION FUNCTION FOR EASY REUSE
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 13: HOW TO USE THE SAVED MODEL")
print("=" * 80)

print("""
To use the saved model later, run this code:

import joblib
import pandas as pd

# Load the saved model
model = joblib.load('trained_bug_classifier.pkl')

# Extract components
vectorizer = model['vectorizer']
root_model = model['root_cause_model']
team_model = model['fix_team_model']
root_encoder = model['root_cause_encoder']
team_encoder = model['fix_team_encoder']

# New bug to classify
bug = "Your bug description here"

# Clean the text
bug_clean = bug.lower()
bug_clean = pd.Series([bug_clean]).str.replace(r'[^a-z0-9\s]', ' ', regex=True)[0]
bug_clean = pd.Series([bug_clean]).str.replace(r'\s+', ' ', regex=True)[0].strip()

# Transform and predict
features = vectorizer.transform([bug_clean])
root_cause = root_encoder.inverse_transform(root_model.predict(features))[0]
fix_team = team_encoder.inverse_transform(team_model.predict(features))[0]

print(f"Root Cause: {root_cause}")
print(f"Fix Team: {fix_team}")
""")

# ============================================================================
# SECTION 14: SUMMARY AND NEXT STEPS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 14: SUMMARY AND RECOMMENDATIONS")
print("=" * 80)

print(f"""
📊 MODEL PERFORMANCE SUMMARY:
   - Root Cause Accuracy: {root_accuracy:.2%}
   - Fix Team Accuracy: {team_accuracy:.2%}
   - Total bugs processed: {len(df)}
   - Features extracted: {len(feature_names)}

✅ WHAT THE MODEL LEARNED:
   - {len(root_cause_encoder.classes_)} different root cause categories
   - {len(fix_team_encoder.classes_)} different fix teams
   - Key patterns in bug descriptions

📈 RECOMMENDATIONS:
   1. If accuracy < 70%: Add more training data
   2. If specific categories perform poorly: Balance the dataset
   3. For better results: Ensure consistent bug descriptions
   4. Regular retraining: Update monthly with new bugs

🚀 NEXT STEPS:
   1. Deploy the model for real-time predictions
   2. Create a web interface or API
   3. Integrate with your bug tracking system
   4. Monitor performance and retrain periodically
""")

print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 80)
