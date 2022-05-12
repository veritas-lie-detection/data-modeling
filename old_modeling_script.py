from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import spacy


def tokenize_and_stopwords(text, stopwords):
    tokens = nltk.word_tokenize(text)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha() or w.isnumeric()]
    meaningful_words = [w for w in token_words if not w in stopwords]
    joined_words = (" ".join(meaningful_words))
    return joined_words


def benfords(text):
    tokens = nltk.word_tokenize(text)
    leading_digits = [0] * 10

    for token in tokens:
        if token.isnumeric():
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
    model = pipeline.fit(train_x, train_y)
    prediction = model.predict(test_x)
    print("accuracy: {}%".format(round(accuracy_score(test_y, prediction) * 100, 2)))
    print(confusion_matrix(test_y, prediction))
    print(classification_report(test_y, prediction, digits=4))


def split_by_doc(df, test_size, target_name, doc_index="url", seed=None):
    if seed is not None:
        np.random.seed(seed)
    ids = df[doc_index].unique()
    train = np.random.choice(ids, size=int(ids.size * (1 - test_size)), replace=False)

    train_df = df[df[doc_index].isin(train)]
    test_df = df.loc[~df.index.isin(train_df.index)]

    # remove identifier column
    train_df = train_df.drop([doc_index], axis=1)
    test_df = test_df.drop([doc_index], axis=1)

    return (train_df.drop([target_name], axis=1), train_df[target_name],
            test_df.drop([target_name], axis=1), test_df[target_name])


def replace_ents(string, nlp_engine):
    str_list = list(string)
    doc = nlp_engine(string)
    for ent in doc.ents:
        str_list[ent.start_char: ent.end_char] = ent.label_
    return "".join(str_list)


if __name__ == "__main__":
    # nlp = spacy.load("en_core_web_md")
    # stop_words = stopwords.words("english")
    # stop_words = set(stop_words)
    #
    # df = pd.read_csv("data/df_raw.csv")
    # df = pd.melt(df, id_vars=["url", "classification"], value_vars=["1A", "7", "7A"])
    # df = df.drop(["variable"], axis=1)
    # df = df[df["value"] != "undefined"]
    #
    # df["value"] = df["value"].apply(replace_ents, nlp_engine=nlp)  # takes 20 min
    # df["value"] = df["value"].apply(tokenize_and_stopwords, stopwords=stop_words)
    #
    # df["benfords"] = df["value"].apply(benfords)
    #
    # sia = SentimentIntensityAnalyzer()
    # df["polarity"] = df["value"].apply(lambda x: sia.polarity_scores(x))
    # df.dropna(inplace=True)
    # df_temp = pd.json_normalize(df["benfords"])
    # df = pd.concat([df.drop(["benfords"], axis=1), df_temp], axis=1)
    #
    # df_temp = pd.json_normalize(df["polarity"])
    # df = pd.concat([df.drop(["polarity"], axis=1), df_temp], axis=1)
    # df.dropna(inplace=True)

    # df.to_csv("data/df_final_2.csv")
    df = pd.read_csv("data/df_final_2.csv")
    df = df.drop(["Unnamed: 0"], axis=1)

    x_train, x_test, y_train, y_test = split_by_doc(df, test_size=0.3, target_name="classification")

    transformer = ColumnTransformer([("vec", TfidfVectorizer(ngram_range=(1, 3)), "value")], remainder="passthrough")

    # # LogReg
    # print("Logistic Regression")
    # pipe = Pipeline([("vect", transformer), ("model", LogisticRegression())])
    # test_model(pipe, x_train, x_test, y_train, y_test)

    # # SVC
    # print("SVC")
    # pipe = Pipeline([("vect", transformer), ("model", SVC(kernel="poly"))])
    # test_model(pipe, x_train, x_test, y_train, y_test)

    # # Random forest classifier
    # print("Random Forest Classifier")
    # pipe = Pipeline([("vect", transformer), ("model", RandomForestClassifier())])
    # test_model(pipe, x_train, x_test, y_train, y_test)

    # GradientBoostingClassifier
    print("Gradient Boosting Classifier")
    pipe = Pipeline([("vect", transformer), ("model", GradientBoostingClassifier())])
    test_model(pipe, x_train, x_test, y_train, y_test)

    # # Naive bayes
    # print("GaussianNB")
    # pipe = Pipeline(
    #     [
    #         ("vect", transformer),
    #         ("dense", FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
    #         ("model", GaussianNB())
    #     ]
    # )
    # test_model(pipe, x_train, x_test, y_train, y_test)

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
    # test_model(pipe, x_train, x_test, y_train, y_test)
