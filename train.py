import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import joblib
from time import time

# Load dataset in chunks (memory-efficient for large files)
chunks = pd.read_csv('C:/Users/saile/OneDrive/Desktop/000/revised_kddcup_dataset.csv', chunksize=10000)
df = pd.concat(chunks)

# Verify dataset size
print(f"\n📊 Dataset loaded with {len(df):,} entries")
print("Class distribution:")
print(df['result'].value_counts())

# Feature selection (optimized set)
features = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'count', 'srv_count',
    'same_srv_rate', 'dst_host_srv_count'
]

# Preprocessing
print("\n⚙️ Preprocessing data...")
start = time()
df['is_attack'] = df['result'].apply(lambda x: 0 if x == 'normal' else 1)

# Encode categoricals
for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    joblib.dump(le, f'{col}_encoder.pkl')

# Prepare data
X = df[features]
y = df['is_attack']

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')
print(f"Preprocessing completed in {time()-start:.2f} seconds")

# Train Random Forest (optimized for large data)
print("\n🌲 Training Random Forest...")
start = time()
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=25,
    min_samples_split=5,
    n_jobs=-1,  # Use all cores
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, y_train)
print(f"RF trained in {time()-start:.2f} seconds")

# Train Neural Network
print("\n🧠 Training Neural Network...")
start = time()
ann = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    batch_size=512,
    early_stopping=True,
    random_state=42
)
ann.fit(X_train, y_train)
print(f"ANN trained in {time()-start:.2f} seconds")

# Save models
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(ann, 'ann_model.pkl')

# Evaluate
print("\n🔍 Evaluation Results:")
print("Random Forest:")
print(classification_report(y_test, rf.predict(X_test)))
print("\nNeural Network:")
print(classification_report(y_test, ann.predict(X_test)))

print("\n✅ Training complete! Models saved.")