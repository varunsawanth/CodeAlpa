import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
# Load dataset
df = pd.read_csv('/content/train_and_test2.csv')
df.head()
df.isnull().sum()
df.describe()
df.rename(columns={'2urvived': 'Survived'}, inplace=True)
df.head()
sns.countplot(x='Survived',data=df)
sns.barplot(x='Pclass',data=df)
sns.barplot(x='Sex',data=df)
df['Age'].fillna(df['Age'].median(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df.isnull().sum()
df.drop(['Passengerid'], axis=1, inplace=True)
df.head()
x= df.drop('Survived',axis=1)
y=df['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5, random_state=42)
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
accuracy = accuracy_score(y_test,y_pred)
print(f'Accurcay :{accuracy*100:.2f}%')

cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True,fmt="d")
print(classification_report(y_test,y_pred))
feature_importances = pd.DataFrame(model.feature_importances_, index=x_train.columns,columns=['Importance']).sort_values('Importance', ascending = False)
print(feature_importances)
feature_importances.plot(kind='bar')
