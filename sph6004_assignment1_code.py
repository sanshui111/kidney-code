# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:27:08 2024

@author: zzzzzm
"""

from sklearn.model_selection import train_test_split
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report

pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)
data_path = 'D:/2024-1学年/sph6004/sph6004_assignment1_data.csv'
df = pd.read_csv(data_path)
rows, cols = df.shape
print(f"数据集有 {rows} 行和 {cols} 列。")
print("\n描述性统计信息：")
print(df.describe())
print("\n列的缺失值数量：")
print(df.isnull().sum())

for col in ['gender', 'race']:
    mode_value = df[col].mode()[0]  
    df[col].fillna(mode_value, inplace=True)   
missing_percent = df.drop(columns=['gender', 'race']).isnull().mean() * 100
high_missing_columns = missing_percent[missing_percent > 50].index
df.drop(columns=high_missing_columns, inplace=True)
low_missing_columns = missing_percent[missing_percent <= 50].index
for col in low_missing_columns:
    if col != 'gender' and col != 'race':
        median_value = df[col].mean()
        df[col].fillna(median_value, inplace=True)
numeric_columns = df.select_dtypes(include=['number']).columns
z_scores = stats.zscore(df[numeric_columns])
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)  
df = df[filtered_entries]
encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object': 
        df[col] = encoder.fit_transform(df[col])
X = df.drop(columns=['id','aki'])
y = df['aki']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
scaled_df = pd.concat([X_train_scaled_df, X_test_scaled_df])
scaled_df.to_csv('D:/2024-1学年/sph6004/scaled_data.csv', index=False)
print(scaled_df)
data = pd.read_csv('D:/2024-1学年/sph6004/scaled_data.csv')
forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train_scaled, y_train)
plt.figure(figsize=(10, 6))
sns.barplot(x=forest.feature_importances_, y=X.columns)
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
selector = SelectFromModel(forest,threshold=0.01,prefit=True)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42)
}
results = {}
for name, model in models.items():
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, zero_division=1)
    results[name] = {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'Support': support
    }
for name, model in models.items():
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))
plt.figure(figsize=(16, 12))
for i, metric in enumerate(['Precision', 'Recall', 'F1 Score', 'Support'], 1):
    plt.subplot(2, 2, i)
    sns.barplot(x=list(results.keys()), y=[results[model][metric][1] for model in results.keys()])
    plt.title(f'{metric} for Class 1')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.show()
