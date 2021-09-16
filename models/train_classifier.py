import sys

import pandas as pd
import nltk
import pickle

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    This function loads from database and separates out target column
    :param database_filepath: string
    :return: X, Y and column names
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('clean_messages', engine)
    X = df.message
    Y = df.iloc[:, 4:]
    col_names = list(Y.columns)

    return X, Y, col_names


def tokenize(text):
    """
    Creates tokens from sentences using WordNetLemmatizer
    :param text: string
    :return: list of tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    creates a model using Pipeline and trains on best hyperparameters
    :return: model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # This parameter list took too long to run and created a pickle file of
    # more than 950 MBs
    # parameters = {
    #         'vect__max_df': (0.5, 0.75),
    #     'tfidf__use_idf': (True, False),
    #         'clf__estimator__n_estimators': [10, 50],
    # }

    parameters = {
        'vect__max_df': ([0.75]),
        'tfidf__use_idf': ([False]),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predicts using test set and prints classification report
    :param model:
    :param X_test: dataframe
    :param Y_test: dataframe
    :param category_names: list
    :return:
    """
    y_pred = model.predict(X_test)

    for i in range(0, len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test.values[:, i], y_pred[:, i]))
        print('Accuracy {}'.format(
            accuracy_score(Y_test.values[:, i], y_pred[:, i])))
        print('*******************************************************')


def save_model(model, model_filepath):
    """
    Dumps the ML model to a pickle file
    :param model:
    :param model_filepath: string
    :return: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Calls the function that perform following tasks:
    Loads the data
    Splits into train and test sets
    Builds, Evaluate, fit and finally saves the model
    :return: None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                            random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
