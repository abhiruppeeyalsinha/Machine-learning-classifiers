import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix



dataset = r"E:\Projects & Tutorial\CNN Project\Hyperparameter Tuning a Neural Network\diabetes.csv"
dataset = pd.read_csv(dataset)
x = dataset.iloc[:,:-1].values
y = dataset["Outcome"].values
sc = MinMaxScaler()
x = sc.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.25,shuffle=True)


err_rate = []
for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn_model = knn.fit(x_train,y_train)
    # model = pickle.dump(knn_model, open("KNN_3.pkl","wb"))
    result = knn_model.predict(x_test)
    err_rate.append(np.mean(result != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),err_rate,color='blue',linestyle="--",markersize=10,markerfacecolor='red',marker='o')
plt.title('k versus Error rate')
plt.xlabel('k')
plt.ylabel('Error_rate')
plt.show()

# print(result)


    # print(confusion_matrix(y_test,result))
    # print(classification_report(y_test,result))





