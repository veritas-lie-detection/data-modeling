from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import en_core_web_sm
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import nltk
import pandas as pd
from tracemalloc import stop
import spacy
import tracemalloc
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import time
t0 = time.time()


def tokenize_and_stopwords(text, stopwords):
    """Tokenize and remove all stop words from the text snippets
        Args:
            text: A segment of text that needs to analysed.
            stopwords: A list of "stopwords" that are going to be removed from the text.
        Returns:
            Section text that has been tokenized and stop words removed.
        """
    tokens = nltk.word_tokenize(text)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha() or w.isnumeric()]
    meaningful_words = [w for w in token_words if not w in stopwords]
    joined_words = (" ".join(meaningful_words))
    return joined_words


def benfords(text):
    """Applys benfords law to the text segment
        Args:
            text: A segment of text that needs to analysed.
        Returns:
            a vector of values corresponding with the leading digits according to benfords law
        """
    tokens = nltk.word_tokenize(text)
    leading_digits = [0] * 10

    for token in tokens:
        if token.isdigit():

            leading_digits[int(token[0])] += 1

    total = sum(leading_digits)
    if total != 0:
        leading_digits = {
            "leading " + str(i): leading_digits[i] / total for i in range(len(leading_digits))
        }
    else:
        leading_digits = {
            "leading " + str(i): 0 for i in range(len(leading_digits))
        }
    return leading_digits


def test_model(pipeline, train_x, train_y, test_x, test_y):
    """Test a model given a training and test set to return metrics about the accuracy
        Args:
            pipeline: The pipeline that defines the model
            train_x: the training set of explanatory variables
            train_y: the training set of response variables
            test_x: the test set of explanatory variables
            test_y: the test set of response variables
        Returns:
            prints out the accuracy, confusion matrix and classification report
        """
    t1 = time.time()
    model = pipeline.fit(train_x, train_y)
    t2 = time.time()
    print("run time for fitting: ", t2-t1)
    prediction = model.predict(test_x)
    print("accuracy: {}%".format(
        round(accuracy_score(test_y, prediction) * 100, 2)))
    print(confusion_matrix(test_y, prediction))
    print(classification_report(test_y, prediction, digits=4))
    print("run time to predict: ", time.time()-t2)
    print("run time total: ", time.time()-t1)
    print("")
    print("")
    print("")


def split_by_doc(df, test_size, target_name, doc_index="url", seed=None):
    """Split up the 10k data set by document into given train/test sets
         Args:
            df: The given data frame (specifically made for the 10k set)
            test_size: specify what percentage of the data should be in the test set
            target_name: the response variables name in the data frame
            doc_index: specify the column in which you can identify differing documents
            seed: can set the seed for consistent results
        Returns:
        a training set and test set that are split by documents in the form of x_train, y_train, x_test, y_test
        """
    if seed is not None:
        np.random.seed(seed)
    ids = df[doc_index].unique()
    train = np.random.choice(ids, size=int(
        ids.size * (1 - test_size)), replace=False)

    train_df = df[df[doc_index].isin(train)]
    test_df = df.loc[~df.index.isin(train_df.index)]

    # remove identifier column
    train_df = train_df.drop([doc_index], axis=1)
    test_df = test_df.drop([doc_index], axis=1)

    return (train_df.drop([target_name], axis=1), train_df[target_name],
            test_df.drop([target_name], axis=1), test_df[target_name])


def replace_ents(string, nlp_engine):
    """replaces any instance of an entity with a tag
        Args:
            string: a string value that needs entitys removed
            nlp_engine: the preset nlp engine 
        Returns:
            the string with all entitys replaced with a tag
        """
    str_list = list(string)
    doc = nlp_engine(string)
    for ent in doc.ents:
        str_list[ent.start_char: ent.end_char] = ent.label_
    return "".join(str_list)


def create_df(csv_path):
    """Creates a data set with all of the preprocessing steps
        Args:
            csv_path: the path to the raw data set
        Returns:
            a data frame with all of the preprocessing done to it and ready to model
        """

    nlp = en_core_web_sm.load()
    # nlp = spacy.load("en_core_web_md")
    stop_words = stopwords.words("english")
    stop_words = set(stop_words)

    df = pd.read_csv(csv_path)

    df = df.drop(["Unnamed: 0"], axis=1).dropna()
    df = pd.melt(df, id_vars=["target"],
                 value_vars=["text"])
    df = df.drop(["variable"], axis=1)
    df = df[df["value"] != "undefined"]

    df["value"] = df["value"].apply(
        replace_ents, nlp_engine=nlp)  # takes 20 min
    df["value"] = df["value"].apply(
        tokenize_and_stopwords, stopwords=stop_words)

    df["benfords"] = df["value"].apply(benfords)

    sia = SentimentIntensityAnalyzer()
    df["polarity"] = df["value"].apply(lambda x: sia.polarity_scores(x))
    df.dropna(inplace=True)
    df_temp = pd.json_normalize(df["benfords"])
    df = pd.concat([df.drop(["benfords"], axis=1), df_temp], axis=1)

    df_temp = pd.json_normalize(df["polarity"])
    df = pd.concat([df.drop(["polarity"], axis=1), df_temp], axis=1)
    df.dropna(inplace=True)

    df.to_csv("data-collection/df_book_data_final.csv")


if __name__ == "__main__":
    # create_df("data-collection/df_raw.csv")
    t1 = time.time()
    #read in data
    df_opspan = pd.read_csv("data-collection/df_op-span_data_final.csv")
    df_original = pd.read_csv("data-collection/df_final_2.csv")
    df_book = pd.read_csv("data-collection/df_book_data_final.csv")

    #drop unwanted columns
    df_original = df_original.drop(["Unnamed: 0"], axis=1).dropna()
    df_opspan = df_opspan.drop(["Unnamed: 0"], axis=1).dropna()
    df_book = df_book.drop(["Unnamed: 0"], axis=1).dropna()

    #change response to the standard for opspan
    df_opspan = df_opspan.assign(
        classification=lambda dataframe: dataframe['classification'].map(
            lambda i: "fraudulent" if i == "deceptive" else "nonfraudulent")
    )

    #split all of the training and testing
    x_train_0, y_train_0, x_test_0, y_test_0 = split_by_doc(
        df_original, test_size=0.3, target_name="classification")

    X = df_opspan.drop(["file_type", "classification"], axis=1)
    y = df_opspan["classification"]

    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(
        X, y, test_size=0.3, random_state=42)

    X = df_book.drop(["target"], axis=1)
    y = df_book["target"]

    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # x_train = x_train_0
    # x_test = x_test_0
    # y_train = y_train_0
    # y_test = y_test_0

    x_train = pd.concat([x_train_0, x_train_1, x_train_2],
                        ignore_index=True, axis=0)
    x_test = pd.concat([x_test_0, x_test_1, x_test_2],
                       ignore_index=True, axis=0)
    y_train = pd.concat([y_train_0, y_train_1, y_train_2],
                        ignore_index=True, axis=0)
    y_test = pd.concat([y_test_0, y_test_1, y_test_2],
                       ignore_index=True, axis=0)

    #split by using hotel and 10k as training with book as test
    # x_train = pd.concat([x_train_0, x_test_0],
    #                        ignore_index=True, axis=0)
    # x_test = pd.concat([x_train_1, x_test_1],
    #                    ignore_index=True, axis=0)
    # y_train = pd.concat([y_train_0, y_test_0],
    #                     ignore_index=True, axis=0)
    # y_test = pd.concat([y_train_1, y_test_1],
    #                    ignore_index=True, axis=0)

    transformer = ColumnTransformer(
        [("vec", TfidfVectorizer(ngram_range=(1, 3)), "value")], remainder="passthrough")

    # starting the monitoring
    tracemalloc.start()

    # # LogReg
    # print("Logistic Regression")
    # pipe = Pipeline([("vect", transformer), ("model", LogisticRegression())])
    # test_model(pipe, x_train, y_train,  x_test, y_test)

    # print("memory usage: ", tracemalloc.get_traced_memory())
    # tracemalloc.stop()

    # tracemalloc.start()

    # # SVC
    # print("SVC")
    # pipe = Pipeline([("vect", transformer), ("model", SVC(kernel="poly"))])
    # test_model(pipe, x_train, y_train,  x_test, y_test)
    # print("memory usage: ", tracemalloc.get_traced_memory())
    # tracemalloc.stop()

    # tracemalloc.start()

    # # Random forest classifier
    # print("Random Forest Classifier")
    # pipe = Pipeline([("vect", transformer), ("model", RandomForestClassifier())])
    # test_model(pipe, x_train, y_train,  x_test, y_test)
    # print("memory usage: ", tracemalloc.get_traced_memory())
    # tracemalloc.stop()

    # tracemalloc.start()

    # # GradientBoostingClassifier
    # print("Gradient Boosting Classifier")
    # pipe = Pipeline(
    #     [("vect", transformer), ("model", GradientBoostingClassifier())])
    # print("time preprocessing")
    # print(time.time()-t1)
    # test_model(pipe, x_train, y_train,  x_test, y_test)
    # print("memory usage: ", tracemalloc.get_traced_memory())
    # tracemalloc.stop()

    # tracemalloc.start()

    # # model = pipe.fit(X, y)
    # # scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
    # # print("Cross Validation Scores")
    # # print(scores)

    # Naive bayes
    print("GaussianNB")
    pipe = Pipeline(
        [
            ("vect", transformer),
            ("dense", FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
            ("model", GaussianNB())
        ]
    )
    test_model(pipe, x_train, y_train,  x_test, y_test)
    print("memory usage: ", tracemalloc.get_traced_memory())
    tracemalloc.stop()

    # # Multinomial naive bayes
    # print("Multinomial NB")
    # pipe = Pipeline(
    #     [
    #         ("vect", transformer),
    #         ("dense", FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
    #         ("scale", MinMaxScaler()),
    #         ("model", MultinomialNB())
    #     ]
    # )
    # test_model(pipe, x_train, y_train,  x_test, y_test)
