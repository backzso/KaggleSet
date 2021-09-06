import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#LEtsgoooo 
from sklearn import preprocessing

##CSV dosyasının okunması
autos=pd.read_csv('autos.csv',encoding = "ISO-8859-1")


##CSV dosyalarında ki boş olan yerlerin okunması
print(autos.isnull().sum())



# resim sayısı tüm ilanlarda aynı oldugu icin, seller sütünunda 2 satıcı var ama ikincisinden sadece 3 tane değer olduğu için
#tarihler ise ilanın ne zaman oluşturulduğu gibi analizde ihtiyaç duyulmayan bilgiler olduğu için ve name sütunu sadece ilan adını verdiği için
#ihtayaç duymadım ve de sildim name sütunu da gereksiz ilan adına veri analizinde ihtiyacımız yok 
autos.drop(['name','seller', 'offerType', 'nrOfPictures', 'postalCode',"dateCrawled","dateCreated","lastSeen"], axis='columns', inplace=True)
print(autos.head())


#boş verilerin silinmesi
autos = autos.dropna(axis=0)
print(autos.isnull().sum())
print(autos.shape)
print(autos.info())




#dosyadaki nan değerleri ve gereksiz sütunları sildikten sonra bunu yeni bir 
#csv dosyasına yazdım
autosdataFrame = autos.copy()
autosdataFrame.to_csv('autosdf.csv',index = False)
print(autosdataFrame.head())
autosData=pd.read_csv('autosdf.csv')



#Kategorik Verilerin Sayısallaştırılması
le = preprocessing.LabelEncoder()                           
dtype_object=autosData.select_dtypes(include=['object'])
for x in dtype_object.columns:
    autosData[x]=le.fit_transform(autosData[x])
# =============================================================================
# le = preprocessing.LabelEncoder()  
# autosData["vehicleType"] =le.fit_transform(autosData["vehicleType"])
# autosData["fuelType"] =le.fit_transform(autosData["fuelType"])
# autosData["gearbox"] =le.fit_transform(autosData["gearbox"])
# autosData["notRepairedDamage"] =le.fit_transform(autosData["notRepairedDamage"])
# autosData["brand"] =le.fit_transform(autosData["brand"])
# autosData["model"] =le.fit_transform(autosData["model"])
# autosData["abtest"] =le.fit_transform(autosData["abtest"])
# =============================================================================

#Bağımlı ve Bağımsız Değişkenleri tanımlanması
X=autosData.drop(['gearbox'],axis=1)
y=autosData['gearbox'].values

#Eğitim ve test için verileri %80 - %20 böldük.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20 )    


#Sayısal verilerin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()                               
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

algorithms=[]                                       
score=[]                                                   




##LOJİSTİK REGRESYON
from sklearn.linear_model import LogisticRegression
logisticReg = LogisticRegression()
logisticReg.fit(X_train,y_train)
score.append(logisticReg.score(X_test,y_test)*100)
algorithms.append("Logistic Regression")
print("Logistic Regression accuracy {}".format(logisticReg.score(X_test,y_test)))


#Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
y_pred=logisticReg.predict(X_test)
y_true=y_test
confusionMat=confusion_matrix(y_true,y_pred)
print(confusionMat)

#Confusion Matrix on Heatmap
fig,ax=plt.subplots(figsize=(5,5))
sns.heatmap(confusionMat,annot=True,linewidths=0.5,linecolor="red",cmap="YlGnBu",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Logistic Regression Confusion Matrix")
plt.show()
print(classification_report(y_true,y_pred))



## KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
knn.predict(X_test)
score.append(knn.score(X_test,y_test)*100)
algorithms.append("KNN")
print("KNN accuracy =",knn.score(X_test,y_test)*100)
y_pred=knn.predict(X_test)
y_true=y_test
confusionMatKNN=confusion_matrix(y_true,y_pred)
print(confusionMatKNN)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(confusionMatKNN,annot=True,linewidths=0.5,linecolor="red",cmap="BuPu",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title(" KNN Confusion Matrix")
plt.show()
print(classification_report(y_true,y_pred))



# DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

decisionTree=DecisionTreeClassifier()
decisionTree.fit(X_train,y_train)
print("Decision Tree accuracy:",decisionTree.score(X_test,y_test)*100)
score.append(decisionTree.score(X_test,y_test)*100)
algorithms.append("Decision Tree")

#Confusion Matrix
y_pred=decisionTree.predict(X_test)
y_true=y_test
confusionMatDecisionTree=confusion_matrix(y_true,y_pred)
print(confusionMatDecisionTree)

#Confusion Matrix on Heatmap
figd,ax=plt.subplots(figsize=(5,5))
sns.heatmap(confusionMatDecisionTree,annot=True,linewidths=0.5,linecolor="red",cmap="YlGnBu",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Decision Tree Confusion Matrix")
plt.show()
print(classification_report(y_true, y_pred))




x_pos = [i for i, _ in enumerate(algorithms)]

sns.barplot(x_pos,score)
plt.xlabel("Algoritmalar")
plt.ylabel("Basari Yüzdeleri")
plt.title("Basari Sıralamalar")

plt.xticks(x_pos, algorithms,rotation=45)

plt.show()
