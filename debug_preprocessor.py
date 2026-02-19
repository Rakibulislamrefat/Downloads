import joblib
import numpy as np

pre = joblib.load('preprocessor.pkl')
print('PREPROCESSOR TYPE:', type(pre))
if hasattr(pre, 'transformers_'):
    for i, (name, trans, cols) in enumerate(pre.transformers_):
        print(f'>>> Transformer {i}:', name, type(trans), 'cols=', cols)
        try:
            if hasattr(trans, 'categories_'):
                print('   CATEGORIES types:')
                for j, cats in enumerate(trans.categories_):
                    print('    - idx', j, 'len', len(cats), 'dtype', getattr(cats, 'dtype', type(cats)), 'sample', cats[:5])
        except Exception as e:
            print('   Could not read categories_', e)
        try:
            if hasattr(trans, 'get_params'):
                print('   params keys:', list(trans.get_params().keys())[:10])
        except Exception as e:
            pass
else:
    print('No transformers_ attribute')

# Print training features
try:
    feats = joblib.load('training_features.pkl')
    print('TRAINING FEATURES COUNT:', len(feats))
    print('SAMPLE FEATURES:', feats[:40])
except Exception as e:
    print('Could not load training_features.pkl', e)

# Print first 10 categories if any
print('done')
