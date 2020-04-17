# KaggleProj
---
[![alt text](https://avatars1.githubusercontent.com/u/59831504?s=400&v=4 "MtheEPIC User Icon")](https://github.com/MtheEPIC/KaggleProj)
by MtheEPIC
(use git lfs or the link in the card fraud section for the data base)
## Apple Store
here we will show some trends in the Apple Store Data using Matplotlib and Seaborn
### Matplotlib Graphs
![alt text](https://github.com/MtheEPIC/KaggleProj/blob/master/graphs/plt1.png)
![alt text](https://github.com/MtheEPIC/KaggleProj/blob/master/graphs/plt2.png)
![alt text](https://github.com/MtheEPIC/KaggleProj/blob/master/graphs/plt3.png)
![alt text](https://github.com/MtheEPIC/KaggleProj/blob/master/graphs/plt4.png)
![alt text](https://github.com/MtheEPIC/KaggleProj/blob/master/graphs/plt5.png)
![alt text](https://github.com/MtheEPIC/KaggleProj/blob/master/graphs/plt6.png)
### Seaborn Graphs
![alt text](https://github.com/MtheEPIC/KaggleProj/blob/master/graphs/sns1.png)
![alt text](https://github.com/MtheEPIC/KaggleProj/blob/master/graphs/sns2.png)
![alt text](https://github.com/MtheEPIC/KaggleProj/blob/master/graphs/sns3.png)
![alt text](https://github.com/MtheEPIC/KaggleProj/blob/master/graphs/sns4.png)
![alt text](https://github.com/MtheEPIC/KaggleProj/blob/master/graphs/sns5.png)
![alt text](https://github.com/MtheEPIC/KaggleProj/blob/master/graphs/sns6.png)

---
## Card Fraud
in the [Card Fraud Dataset](https://www.kaggle.com/ntnu-testimon/paysim1 "Synthetic Financial Datasets For Fraud Detection") we will try to understand the data trends and create a model to predict fraud

### We have 11 initial features:
* **step:** Maps a unit of time in the real world. In this case 1 step is 1 hour of time, 743 is the end of the month
* **type:** CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER
* **amount:** amount of the transaction in local currency _(may be bigger then the account balance)_
* **nameOrig:** customer who started the transaction _(start with C for customer)_
* **oldbalanceOrg:** initial balance before the transaction _(at least 0)_
* **newbalanceOrig:** customer's balance after the transaction. _(at least 0)_
* **nameDest:** recipient ID of the transaction. _(start with C for customer)_
* **oldbalanceDest:** initial recipient balance before the transaction. _(at least 0)_
* **newbalanceDest:** recipient's balance after the transaction. _(at least 0)_
* **isFraud:** identifies a fraudulent transaction (1) and non fraudulent (0)
* **isFlaggedFraud:** flags illegal attempts to transfer more than 200.000 in a single transaction.

### Some key trends
#### Fraud is only in: TRANSFER, CASH OUT
```
fraud = df[df['isFraud'] == 1]
fraud['type'].value_counts(normalize=True)*100
```
#### If The Transaction Amount Was The Same As The Balance: Its Fraud
```
df[df['amount'] == df['oldbalanceOrg']]['isFraud'].unique()
```
#### If the Existing Fraud Detection Flags the Transaction as Fraud, It's Fraud
```
df[df['isFlaggedFraud'] == 1]['isFraud'].unique()
```

### Clear and Prepare data
#### For example:
based on the trends we can remove the transaction without fraud and create an improved fraud system
```
d1 = df[df['type']=='TRANSFER']
d2 = df[df['type']=='CASH_OUT']
d1['type'] = 1
d2['type'] = 0
df = pd.concat([d1, d2])
df.head()
```
remove the data that we don't need
```
df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud', 'step'], axis=1)
X=df.drop('isFraud', axis=1)
y=df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### Models to use for predictions
note that the new fraud detection system makes the models run very well (even to well)
after reading in the [Card Fraud Dataset](https://www.kaggle.com/ntnu-testimon/paysim1/discussion/99799 "Synthetic Financial Datasets For Fraud Detection") we can see that it's that because this data is synthetic this one attribute is very crucial

The models we will use are:
* Naive Bayes
* Decision Tree
* Adaboost
* Logistic Regression
* Random Forest
* Linear SVC

(knn takes __too__ much time)

the best model is Random Forest:
```
Model AUC Score: 0.94935
threshold: 0.02946730172263339
The model is better then a 'dumb' model
confusion_matrix:
 [[551096   1308]
 [   166   1512]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    552404
           1       0.54      0.90      0.67      1678

    accuracy                           1.00    554082
   macro avg       0.77      0.95      0.84    554082
weighted avg       1.00      1.00      1.00    554082
```