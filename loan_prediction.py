import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

df = pd.read_csv('loan_data.csv')
df.info()
df.describe()

df.head()


####  EDA
plt.figure(figsize = (10,5))
df[df['credit.policy'] == 1]['fico'].plot.hist(bins = 30)
df[df['credit.policy'] == 0]['fico'].plot.hist(bins = 30,alpha = 0.6)
plt.legend(['credit.policy=1','credit.ploicy=0'])
plt.xlabel('FICO')

plt.figure(figsize = (10,5))
df[df['not.fully.paid'] == 0]['fico'].plot.hist(bins = 30)
df[df['not.fully.paid'] == 1]['fico'].plot.hist(bins = 30,alpha = 0.6)
plt.legend(['not.fully.paid=0','not.fully.paid=1'])
plt.xlabel('FICO')
plt.tight_layout()

plt.figure(figsize = (10,5))
sns.countplot(x = 'purpose',data = df,hue = 'not.fully.paid')
plt.tight_layout()

sns.jointplot(x = 'fico',y = 'int.rate',data = df)
plt.tight_layout()

sns.lmplot(x = 'fico',y = 'int.rate',data = df,col = 'not.fully.paid',hue = 'credit.policy')

#Notice that the Purpose column as categorical 
#That means we need to transform them using dummy variables so sklearn will be able to understand them.

cat_feats = ['purpose']

final_data = pd.get_dummies(df,columns = cat_feats,drop_first=True)
final_data.head()

from sklearn.model_selection import train_test_split
y = final_data['not.fully.paid']
X = final_data.drop('not.fully.paid',axis = 1)

#Now its time to split our data into a training set and a testing set!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the model with Single instance of Decision tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

pred = dtree.predict(X_test)

#Creating an instance of the RandomForestClassifier class and fitting it to our training data from the previous step.
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train,y_train)

predictions = rfc.predict(X_test)

from sklearn.metrics import accuracy_score
print("DecisionTree Accuracy:",accuracy_score(y_test,pred))
print("Random Forest Accuracy",accuracy_score(y_test,pre))