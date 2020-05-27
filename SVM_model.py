import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
    
#DATA INPUT- DO cannot be lower than 0% and higher than 100%
data = pd.read_excel (r'C:\Users\Mateusz\Desktop\SSI\Sem_4\Data\Thesis_codes\Classification_S1_S2\Correlation_new\alldata.xlsx')
df2= data[['BatchID','f_timeh','DO_live','rpm_live','Phase']]
df3=df2.loc[(df2.Phase<3)]#only Phase 1 and 2
df=df3.loc[(df3.rpm_live<=10000) & (df3.DO_live<=100) & (df3.DO_live>=0)] #taking out outliers
#df = df4[df4.index % 50 == 0] #taking every 50 sample (for better diagrams visibility)

x = df.drop('Phase', axis=1)
y=df[["BatchID","Phase"]]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

#Preparation for training and veryfication- erasing BatchID from the model
BatchIDX_test=X_test[['BatchID']]
X_train2=X_train.drop('BatchID',axis=1)
X_test2=X_test.drop('BatchID',axis=1)

BatchIDy_test=y_test[['BatchID']]
y_train2=y_train.drop('BatchID',axis=1)
y_test2=y_test.drop('BatchID',axis=1)

#SVM
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train2,y_train2.iloc[:,0])
#svclassifier.coef_ #-variables importances (f_timeh, DO_live, rpm_live)
#PLOT- ALL TOGETHER ON ONE PLOT 

y_pred = svclassifier.predict(X_test2)
plt.scatter(X_test2.f_timeh,X_test2.DO_live, c=y_pred, cmap='viridis',alpha=0.5)
plt.title('All batches')
plt.xlabel('time[h]')
plt.ylabel('DO level [%]')
plt.show()

print(confusion_matrix(y_test2,y_pred))
print(classification_report(y_test2,y_pred))

#PLOT- SEPERATELY ALL BATCHES ON SEPERATE PLOTS

X_test2['BatchID']=BatchIDX_test
y_test2['BatchID']=BatchIDy_test
batches = list(set(X_test2.BatchID))
    
for i in batches:
    df3_test = X_test2.loc[X_test2.BatchID == i]
    df10_test=df3_test.drop('BatchID',axis=1)
    y_pred = svclassifier.predict(df10_test)
    print(i)
    plt.scatter(df10_test.f_timeh,df10_test.DO_live, c=y_pred, cmap='viridis')
    plt.title('Batch ' +str(i))
    plt.xlabel('time[h]')
    plt.ylabel('DO level [%]')
    plt.show()
    df_y2=y_test2.loc[y_test2.BatchID ==i]#.reset_index(drop=True)
    df_y22=df_y2.drop('BatchID',axis=1)
    print(confusion_matrix(df_y22,y_pred))
    print(classification_report(df_y22,y_pred))