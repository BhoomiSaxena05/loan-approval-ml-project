                                          #  Objective:
# Based on applicant data (income, education, employment status, credit history, etc),
#  predict whether load will be approved (Yes) or not (No).

# step 1: import library $ dataset
import pandas as pd
df = pd.read_csv('LoanPrediction(Train).csv')
print(df.head())

# step2: data cleaning
df.isnull().sum() # check missing values

# fill missing values
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents']= df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mode()[0])
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])

# Step-3:Encode Categorical Features
from sklearn.preprocessing import LabelEncoder

cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status', 'Dependents']
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

# Step-4: Train test split
from sklearn.model_selection import train_test_split

x = df.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)  # sab kuch except ID & target
y = df[['Loan_Status']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# step-5: train model
# let's use random forest (you can aslo try naive bayes or svm)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Model = RandomForestClassifier()
Model.fit(x_train, y_train)

# predictmodel
y_pred = Model.predict(x_test)

# step 6 : evaluation model
print("accuracy:", accuracy_score(y_pred, y_test))
print("classification_report:", classification_report(y_pred, y_test))
print("confusion_matrix:", confusion_matrix(y_pred, y_test))



