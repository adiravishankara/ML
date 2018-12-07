from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston


bos = load_boston()
hd = pd.DataFrame(bos.data)


hd['PRICE'] = bos.target

X = hd.drop('PRICE',axis=1)
Y = hd['PRICE']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

plt.plot(X_train,Y_train)
mlp = MLPRegressor(solver='sgd',learning_rate='adaptive',max_iter=100000)
mlp.fit(X_train,Y_train)

predictions = mlp.predict(X_test)

# plt.figure()
# plt.plot(Y_test)
# plt.plot(predictions)
# plt.show()

print(explained_variance_score(Y_test,predictions)*100)

#print(classification_report(Y_test,predictions))
