import kagglehub
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from xgboost import plot_importance 
import matplotlib.pyplot as plt
import seaborn as sns 

# load
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# input and target (drop target from input)
train_labels = train_df['SalePrice']
train_data = train_df.drop(['SalePrice', 'Id'], axis=1) 
test_ids = test_df['Id']
test_data = test_df.drop(['Id'], axis=1) 
combined = pd.concat([train_data, test_data], keys=['train', 'test'])

#process
combined = combined.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1) 

#potential missing values 
for col in combined.select_dtypes(include=['object']).columns:
    combined[col] = combined[col].fillna('Missing')
for col in combined.select_dtypes(exclude=['object']).columns:
    combined[col] = combined[col].fillna(combined[col].median())

# label encoding
for col in combined.select_dtypes(include='object').columns:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# split into train/test
X_train = combined.loc['train']
X_test = combined.loc['test']
y_train = train_labels
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#model xgbr regresion
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.2,
    early_stopping_rounds=5,
    n_jobs=-1)


model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False)

#predict
predictions = model.predict(X_test) 
output = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions})  

val_predictions = model.predict(X_val)

# calc r^2
r2 = r2_score(y_val, val_predictions)
print(f" r^2 Score: {r2}") #0.911!!

#graph
val_predictions = model.predict(X_val) 

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_val, y=val_predictions, alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--') 
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs. Predicted Sale Price')
plt.show()


#graphing freature importance 
plt.figure(figsize=(12, 8))
plot_importance(model, max_num_features=20) 
plt.show()
