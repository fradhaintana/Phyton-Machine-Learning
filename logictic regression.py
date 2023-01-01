import pandas as pd
dataku = pd.read_csv('diabetes.csv')
x = dataku.iloc[:,:].values

# menghilangkan baris atau kolom
## menghilangkan kolom terakhir
y = dataku.iloc[:,:-1].values
## menghilangkan baris terakhir
z = dataku.iloc[:-1,:].values

# eksplorasi data
## akan dilakukan cleaning data
data_diabet = dataku.dropna()
data_diabet

## Kategori variabel Y 
dummy = pd.get_dummies(dataku['Outcome'])
dummy.head()
dataedit = pd.concat((dataku,dummy), axis=1)
datanew = dataedit.drop(['Outcome'], axis=1)
datanew = datanew.drop(['Control'], axis=1)
### rename kolom
datanew = datanew.rename(columns={'Case':'Hasil'})

## Membuat tabel frekensi untuk variabel Y hasil
count = datanew['Hasil'].value_counts()
print(count)

tabelfrek = {'Hasil_pasien': ['Kontrol', 'Case'],
        'jumlah': [490, 262]}
df1 = pd.DataFrame(tabelfrek)
df1

## Plotting Data
import matplotlib.pyplot as plt
plt.figure()
df1['jumlah'].plot.hist()
df1.plot(kind='pie', y = 'jumlah', labels=df1['Hasil_pasien'])

# Machine Learning
## Regresi Logistik
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

### Pembagian Data Latih dan Data Uji
X = datanew[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
             'BMI','DiabetesPedigreeFunction','Age']]
y = datanew[['Hasil']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

### Pembuatan Model
log_regression = LogisticRegression()
log_regression.fit(X_train,y_train)

### Menguji Prediksi
y_pred = log_regression.predict(X_test)
y_pred # langsung bentuk klasifikasi tanpa perlu cutoff

### Confussion Matrix Data Uji
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

### Akurasi
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Hasil akurasi 72.84%

### ROC Plotting
y_pred_proba = log_regression.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.show()
