from sklearn.svm import SVC
def train_model(X_train, y_train):
    model = SVC(kernel='rbf', C=0.5, gamma='scale', random_state=42, probability=True)
    model.fit(X_train, y_train)
    return model
def predict(model, X_test):
    return model.predict(X_test)