import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import catboost

# Load datasets
from catboost import CatBoostClassifier

df = pd.read_csv("./data/train.csv")
df_tst = pd.read_csv("./data/test.csv")
# Fix missings
for d in df, df_tst:
    d['Cabin'].fillna('Unknown', inplace=True)
    d['Embarked'].fillna('Unknown', inplace=True)
# Select columns for X
cols = list(df.columns)
cols.remove('Survived')
cols.remove('PassengerId')

# Build model
cbc = CatBoostClassifier(iterations=1000, random_seed=0, depth=3).fit(df[cols].values, df['Survived'],
                                                                               cat_features=[0, 1, 2, 6, 8, 9])
# Submit prediction
pd.DataFrame({'PassengerId': df_tst['PassengerId'], 'Survived': cbc.predict(df_tst[cols].values).astype(int)}).to_csv(
    'demo_catboost.csv', index=False)
