'''import proper libraries like
    data from preprocess, keras,
    keras.Sequential, keras,layers
    keras.utils, keras.metrics'''

'''Since all the Functions are in preprocess, we just have to call them here'''

X_train, X_test, y_train, y_test = get_train_test()
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

model = Sequential()

''' Remember that this network also has several other layers such as
a Conv2D layer, maxpool, flatten, dropout, and so on. Remember to build it
the way you need it '''

model.fit(X_train, y_train_hot, other parameters, validation_data=(X_test,y_test_hot))

''' finally you can print the accuracy of your network '''
print(categorical_accuracy(y_test_hot,predictions))
