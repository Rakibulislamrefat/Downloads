import joblib
import pandas as pd
import numpy as np

try:
    best_model = joblib.load('best_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    training_features = joblib.load('training_features.pkl')
except Exception as e:
    print('LOAD_ERROR', e)
    raise SystemExit(1)

# Medium preset values (match Quick preset in app)
qi_input = pd.DataFrame({
    'Hours_Studied': [5.0],
    'Attendance': [80.0],
    'Parental_Involvement': [5.0],
    'Access_to_Resources': [6.0],
    'Extracurricular_Activities': [1.0],
    'Sleep_Hours': [7.5],
    'Previous_Scores': [70.0],
    'Motivation_Level': [6.0],
    'Internet_Access': [1.0],
    'Tutoring_Sessions': [2.0],
    'Family_Income': ['Medium'],
    'Teacher_Quality': [6.0],
    'School_Type': ['Public'],
    'Peer_Influence': [5.0],
    'Physical_Activity': [3.0],
    'Learning_Disabilities': [0.0],
    'Parental_Education_Level': ['Bachelor'],
    'Distance_from_Home': [12.0],
    'Gender': ['Female']
})

# Ensure dtypes
numeric_cols = ['Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources',
                'Extracurricular_Activities', 'Sleep_Hours', 'Previous_Scores', 'Motivation_Level',
                'Internet_Access', 'Tutoring_Sessions', 'Teacher_Quality', 'Peer_Influence',
                'Physical_Activity', 'Learning_Disabilities', 'Distance_from_Home']
for col in numeric_cols:
    qi_input[col] = pd.to_numeric(qi_input[col], errors='coerce').fillna(0).astype('float64')

for col in ['Family_Income', 'School_Type', 'Parental_Education_Level', 'Gender']:
    qi_input[col] = qi_input[col].astype('str')

try:
    # Attempt direct transform
    try:
        X_qi = preprocessor.transform(qi_input)
        X_qi = pd.DataFrame(X_qi).astype('float64')
        X_qi_aligned = pd.DataFrame(0, index=np.arange(X_qi.shape[0]), columns=training_features, dtype='float64')
        for col in X_qi.columns:
            if col in X_qi_aligned.columns:
                X_qi_aligned[col] = X_qi[col].values
    except Exception as e:
        print('DIRECT_TRANSFORM_FAILED', e)
        # Manual transform fallback: scale numeric and one-hot align to training_features
        num_cols = []
        if hasattr(preprocessor, 'transformers_'):
            for name, transformer, cols in preprocessor.transformers_:
                if name == 'num':
                    num_cols = list(cols)

        num_df = qi_input.reindex(columns=num_cols).copy()
        for c in num_df.columns:
            num_df[c] = pd.to_numeric(num_df[c], errors='coerce').fillna(0).astype('float64')

        scaler = preprocessor.named_transformers_.get('num', None)
        if scaler is not None:
            X_num = scaler.transform(num_df)
        else:
            X_num = num_df.values

        feat_cols = list(training_features)
        X_aligned_df = pd.DataFrame(0.0, index=np.arange(len(qi_input)), columns=feat_cols, dtype='float64')

        # fill numeric
        for col in feat_cols:
            if col.startswith('num__'):
                orig = col.replace('num__', '')
                if orig in num_cols and scaler is not None:
                    j = num_cols.index(orig)
                    X_aligned_df[col] = X_num[:, j]
                else:
                    X_aligned_df[col] = pd.to_numeric(qi_input.get(orig, 0), errors='coerce').fillna(0).astype('float64')

        # fill categorical one-hot
        for col in feat_cols:
            if col.startswith('cat__'):
                parts = col.split('__', 2)
                if len(parts) == 3:
                    rest = parts[2]
                    try:
                        kname, kcat = rest.rsplit('_', 1)
                    except Exception:
                        continue
                    if kname in qi_input.columns:
                        val = str(qi_input[kname].values[0])
                        if val == kcat:
                            X_aligned_df[col] = 1.0

        X_qi_aligned = X_aligned_df

    pred = int(best_model.predict(X_qi_aligned)[0])
    perf_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    label = perf_map.get(pred, 'Unknown')

    print('PREDICTION', pred, label)

    if hasattr(best_model, 'predict_proba'):
        proba = best_model.predict_proba(X_qi_aligned)[0]
        print('PROBABILITIES', proba.tolist())

    print('SUCCESS: Quick-predict test completed')
except Exception as e:
    print('PREDICT_ERROR', e)
    raise
