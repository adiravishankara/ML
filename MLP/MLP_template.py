from sklearn.neural_network import #
from sklearn.model_selection import train_test_split
import pandas as pd

#import Data
data = pd.read_csv(Filename,var = [variable_names])

# Set X and Y
X = Features
Y = Trained Answers

#Separate Training Data and Testing data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

mlp = MLPClassifier(Parameters of the MLP)
mlp.fit(X_train,Y_train)

predictions = mlp.predict(X_test)

#Display the results
print(classification_report(y_test,predictions)
