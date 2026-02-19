import joblib
import pandas as pd
import numpy as np

best_model = joblib.load('best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
training_features = joblib.load('training_features.pkl')


def map_numeric_to_category(value, kind):
    try:
        v = float(value)
    except Exception:
        return str(value)
    if kind == '3level':
        if v < 4:
            return 'Low'
        elif v < 8:
            return 'Medium'
        else:
            return 'High'
    if kind == 'yesno':
        return 'Yes' if v > 0 else 'No'
    if kind == 'peer':
        if v <= 3:
            return 'Negative'
        elif v <= 7:
            return 'Neutral'
        else:
            return 'Positive'
    if kind == 'distance':
        if v < 10:
            return 'Near'
        elif v < 25:
            return 'Moderate'
        else:
            return 'Far'
    return str(value)


def manual_align(df_input):
    # fallback manual alignment similar to app.manual_transform
    num_cols = []
    if hasattr(preprocessor, 'transformers_'):
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                num_cols = list(cols)
    # numeric
    num_df = df_input.reindex(columns=num_cols).copy()
    for c in num_df.columns:
        num_df[c] = pd.to_numeric(num_df[c], errors='coerce').fillna(0).astype('float64')
    scaler = preprocessor.named_transformers_.get('num', None)
    if scaler is not None:
        X_num = scaler.transform(num_df)
    else:
        X_num = num_df.values
    feat_cols = list(training_features)
    X_aligned_df = pd.DataFrame(0.0, index=np.arange(len(df_input)), columns=feat_cols, dtype='float64')
    for col in feat_cols:
        if col.startswith('num__'):
            orig = col.replace('num__', '')
            if orig in num_cols and scaler is not None:
                j = num_cols.index(orig)
                X_aligned_df[col] = X_num[:, j]
            else:
                X_aligned_df[col] = pd.to_numeric(df_input.get(orig, 0), errors='coerce').fillna(0).astype('float64')
    for col in feat_cols:
        if col.startswith('cat__'):
            try:
                _, rest = col.split('__', 1)
                kname, kcat = rest.rsplit('_', 1)
            except Exception:
                continue
            if kname in df_input.columns:
                val = str(df_input[kname].values[0])
                if val == kcat:
                    X_aligned_df[col] = 1.0
    return X_aligned_df


# Define two examples: A (low motivation), B (high motivation)
base = {
    'Hours_Studied': 5.0,
    'Attendance': 80.0,
    'Parental_Involvement': map_numeric_to_category(5, '3level'),
    'Access_to_Resources': map_numeric_to_category(6, '3level'),
    'Extracurricular_Activities': map_numeric_to_category(5, 'yesno'),
    'Sleep_Hours': 7.5,
    'Previous_Scores': 70.0,
    'Motivation_Level': None,  # to fill
    'Internet_Access': 'Yes',
    'Tutoring_Sessions': 2.0,
    'Family_Income': 'Medium',
    'Teacher_Quality': map_numeric_to_category(6, '3level'),
    'School_Type': 'Public',
    'Peer_Influence': map_numeric_to_category(5, 'peer'),
    'Physical_Activity': 3.0,
    'Learning_Disabilities': 'No',
    'Parental_Education_Level': 'Bachelor',
    'Distance_from_Home': map_numeric_to_category(12.0, 'distance'),
    'Gender': 'Female'
}

examples = []
# Low motivation example
ex_a = base.copy()
ex_a['Motivation_Level'] = map_numeric_to_category(2, '3level')
# High motivation example
ex_b = base.copy()
ex_b['Motivation_Level'] = map_numeric_to_category(9, '3level')
examples.append(('LowMotivation', ex_a))
examples.append(('HighMotivation', ex_b))

for name, ex in examples:
    print('\n===', name, '===')
    # convert scalar-valued dict to single-row DataFrame
    df = pd.DataFrame([ex])
    # Ensure correct dtypes: numeric fields
    for c in ['Hours_Studied','Attendance','Sleep_Hours','Previous_Scores','Tutoring_Sessions','Physical_Activity']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    # categorical ensure str
    for c in ['Parental_Involvement','Access_to_Resources','Extracurricular_Activities','Motivation_Level',
              'Internet_Access','Family_Income','Teacher_Quality','School_Type','Peer_Influence','Learning_Disabilities',
              'Parental_Education_Level','Distance_from_Home','Gender']:
        if c in df.columns:
            df[c] = df[c].astype(str)
    print('\nCleaned input:')
    print(df.T)
    # Try transforming
    try:
        Xt = preprocessor.transform(df)
        # If transform returns a numpy array, attempt to use training_features as column names
        try:
            if hasattr(Xt, 'shape') and len(Xt.shape) == 2 and Xt.shape[1] == len(training_features):
                Xdf = pd.DataFrame(Xt, columns=list(training_features))
            else:
                Xdf = pd.DataFrame(Xt)
        except Exception:
            Xdf = pd.DataFrame(Xt)
        # align
        X_aligned = pd.DataFrame(0, index=np.arange(Xdf.shape[0]), columns=training_features, dtype='float64')
        for col in Xdf.columns:
            if col in X_aligned.columns:
                X_aligned[col] = Xdf[col].values
        print('\nAligned features (head):')
        print(X_aligned.iloc[0,:40].to_string())
    except Exception as e:
        print('transform failed:', e)
        X_aligned = manual_align(df)
        print('\nManual aligned features (head):')
        print(X_aligned.iloc[0,:40].to_string())
    pred = int(best_model.predict(X_aligned)[0])
    probs = None
    if hasattr(best_model, 'predict_proba'):
        probs = best_model.predict_proba(X_aligned)[0]
    print('\nPrediction:', pred)
    print('Probabilities:', probs.tolist() if probs is not None else None)

print('\nDone')
