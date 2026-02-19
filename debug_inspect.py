import joblib
import pandas as pd
import numpy as np

best_model = joblib.load('best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
training_features = joblib.load('training_features.pkl')

ex = {
    'Hours_Studied': 5.0,
    'Attendance': 80.0,
    'Parental_Involvement': 'Medium',
    'Access_to_Resources': 'Medium',
    'Extracurricular_Activities': 'Yes',
    'Sleep_Hours': 7.5,
    'Previous_Scores': 70.0,
    'Motivation_Level': 'Low',
    'Internet_Access': 'Yes',
    'Tutoring_Sessions': 2.0,
    'Family_Income': 'Medium',
    'Teacher_Quality': 'Medium',
    'School_Type': 'Public',
    'Peer_Influence': 'Neutral',
    'Physical_Activity': 3.0,
    'Learning_Disabilities': 'No',
    'Parental_Education_Level': 'Bachelor',
    'Distance_from_Home': 'Moderate',
    'Gender': 'Female'
}

df = pd.DataFrame([ex])
# numeric cols from preprocessor
num_cols = []
for name, transformer, cols in preprocessor.transformers_:
    if name == 'num':
        num_cols = list(cols)
print('num_cols:', num_cols)
num_df = df.reindex(columns=num_cols).copy()
for c in num_df.columns:
    num_df[c] = pd.to_numeric(num_df[c], errors='coerce').fillna(0).astype('float64')
print('num_df values:', num_df.values)
scaler = preprocessor.named_transformers_.get('num', None)
print('scaler type:', type(scaler))
try:
    X_num = scaler.transform(num_df)
    print('X_num shape:', X_num.shape)
    print('X_num values:', X_num)
except Exception as e:
    print('scaler.transform error:', e)

# Check training_features first 40
print('\ntraining_features[:40]:')
print(training_features[:40])

# Check categorical mapping for Motivation_Level
mot_colname = 'cat__Motivation_Level_High'
if mot_colname in training_features:
    print('\nFound', mot_colname)
else:
    print('\nMissing', mot_colname)

# Build aligned df and print head
feat_cols = list(training_features)
X_aligned_df = pd.DataFrame(0.0, index=np.arange(len(df)), columns=feat_cols, dtype='float64')
# fill numeric
for col in feat_cols:
    if col.startswith('num__'):
        orig = col.replace('num__','')
        if orig in num_cols and scaler is not None:
            j = num_cols.index(orig)
            X_aligned_df[col] = X_num[:, j]
        else:
            X_aligned_df[col] = pd.to_numeric(df.get(orig,0), errors='coerce').fillna(0).astype('float64')
# fill categorical
for col in feat_cols:
    if col.startswith('cat__'):
        try:
            _, rest = col.split('__',1)
            kname, kcat = rest.rsplit('_',1)
        except Exception:
            continue
        if kname in df.columns:
            val = str(df[kname].astype(str).values[0])
            if val == kcat:
                X_aligned_df[col] = 1.0

print('\nAligned head:')
print(X_aligned_df.iloc[0,:40].to_string())
