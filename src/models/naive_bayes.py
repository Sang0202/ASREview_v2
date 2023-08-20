from sklearn.naive_bayes import MultinomialNB
# import joblib
# create a function that create a Naive Bayes model and fit with the training data
# input: training data
# output: fitted model
#
def fit_naive_bayes(X_train, y_train, alpha = 1.0):
    nb = MultinomialNB(alpha=alpha)
    nb.fit(X_train, y_train)
    return nb

# create a function that predicts the labels of the test data
# input: fitted model, test data
# output: predicted labels
#
def predict_naive_bayes(nb, X_test):
    y_pred = nb.predict(X_test)
    return y_pred

# create a function that predicts the probabilities of the test data
# input: fitted model, test data
# output: predicted probabilities
#
def predict_proba_naive_bayes(nb, X_test):
    y_pred_proba = nb.predict_proba(X_test)
    return y_pred_proba

# # create a function that saves the fitted model with joblib
# # input: fitted model, model name
# # output: saved model
# #
# def save_model(model, model_name):
#     joblib.dump(model, model_name)

# # create a function that loads the fitted model with joblib
# # input: model name
# # output: loaded model
# #
# def load_model(model_name):
#     model = joblib.load(model_name)
#     return model

