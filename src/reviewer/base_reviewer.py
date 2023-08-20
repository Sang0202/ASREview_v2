
from utils.prepare_query import diff_top_two_pred, top_two_pred
from utils.sample_to_label import sample_to_label
from utils.save_load_model import load_model, save_model
from strategies.query_dominate import idx_dominate_to_label
from preprocessing.feature_extraction import fit_tfidf, transform_tfidf
from preprocessing.balance import simple_sample
from preprocessing.feature_extraction.one_hot_encode import one_hot_encode
from models import fit_naive_bayes, predict_proba_naive_bayes
from sklearn.pipeline import Pipeline # chua dung toi



def review_process(data, label_priors, n_label_priors=10, feature_extraction='Tfidf', balance_mode='simple', one_hot=False, classifier='Naive_Bayes', query_strategie='Dominate', n_to_label=5, evaluatoin=False):
    columns_name = ['text']
    if columns_name in data.columns:
        data = data[columns_name]
    else:
        raise Exception('The columns name is not correct: text')
    
    # # sample data to label prior
    # data_to_label = sample_to_label(data, n_label_priors) 

    columns_name_label = ['text','label']
    if columns_name_label in label_priors.columns:
        label_priors = label_priors[columns_name_label]
    else:
        raise Exception('The columns name is not correct: text, label')
    
    ## feature extraction
    # fit the feature extraction model to the whole data
    if feature_extraction == 'Tfidf':
        fe_model = fit_tfidf(data['text'])
        save_model(fe_model, 'fe_model.joblib')

    else:
        raise Exception('The feature extraction is not correct: Tfidf, Count')
    
    # transform the training data to the matrix of TF-IDF features
    if feature_extraction == 'Tfidf':
        fe_model = load_model('fe_model.joblib')
        X = transform_tfidf(fe_model, label_priors['text'])
        X_total = transform_tfidf(fe_model, data['text'])
    
    if one_hot:
        y = one_hot_encode(label_priors[['label']], 'label')
    else:
        y = label_priors['label']

    ## balance training data
    if balance_mode == 'simple':
        X_train, y_train = simple_sample(X, y)
    else:
        raise Exception('The balance mode is not correct: simple')
    
    ## train the model
    if classifier == 'Naive_Bayes':
        classifier_model = fit_naive_bayes(X_train, y_train)
        pred_proba = predict_proba_naive_bayes(classifier_model, X_total)
    else:
        raise Exception('The classifier is not correct: Naive_Bayes')
    
    ## query strategy
    top_2 = top_two_pred(pred_proba)
    diff_top_2 = diff_top_two_pred(top_2)

    if query_strategie == 'Dominate':
        idx_to_label = idx_dominate_to_label(diff_top_2, label_priors.index, n_to_label)
    
    # return the index of the instances to label
    # label idx_to_label instances
