import joblib
# create a function that saves the fitted model with joblib
# input: fitted model, model name
# output: saved model
#
def save_model(model, model_name):
    joblib.dump(model, model_name)

# create a function that loads the fitted model with joblib
# input: model name
# output: loaded model
#
def load_model(model_name):
    model = joblib.load(model_name)
    return model