import time
import spacy
from tracemalloc import stop
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
t0 = time.time()

from collections import Counter
import pandas as pd
import nltk
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
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
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix


stop_words = stopwords.words('english')

df = pd.read_csv("data-collection/df.csv")


df = pd.melt(df, id_vars=['url', 'classification'], value_vars=[
        '1A', "7", "7A"])

df = df[df['value'] != "undefined"]


nlp = spacy.load('en_core_web_sm')
df['value'] = df['value'].apply(lambda x: " ".join(
    [ent.text for ent in nlp(x) if not ent.ent_type_]))
# print(df[df['value'] != "undefined"])
# print(len(df))
# print(len(df[df['value'] != "undefined"]))

def tokenizeandstopwords(text):
    tokens = nltk.word_tokenize(text)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha() or w.isnumeric()]
    meaningful_words = [w for w in token_words if not w in stop_words]
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
            "leading " + str(i): leading_digits[i] / total for i in range(len(leading_digits))}
    else:
        leading_digits = {
            "leading " + str(i): leading_digits[i] for i in range(len(leading_digits))}
    return leading_digits


df['value'] = df['value'].apply(tokenizeandstopwords)



# df = pd.read_csv("df_tokenized.csv")

df['test'] = df['value'].apply(benfords)
df2 = pd.json_normalize(df['test'])
df = pd.concat([df.drop(['test'], axis=1), df2], axis=1)


df.dropna(inplace=True)
sia = SentimentIntensityAnalyzer()


df['test'] = df['value'].apply(lambda x: sia.polarity_scores(x))
df2 = pd.json_normalize(df['test'])
df = pd.concat([df.drop(['test'], axis=1), df2], axis=1)
df.to_csv('df_final_without_ent.csv')



# df = pd.read_csv("df_final.csv")
df.dropna(inplace=True)


# LogReg
print("Logistic Regression")
X = df.drop(['url', 'variable', 'classification'], axis=1)
y = df['classification']
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2020)

# # transformer = ColumnTransformer(
# #     [('vec', TfidfVectorizer(), 'value')],
# #     remainder='passthrough'
# # )

# # pipe = Pipeline([('vect', transformer),
# #                  ('model', LogisticRegression())])

# # model = pipe.fit(x_train, y_train)
# # prediction = model.predict(x_test)
# # print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100, 2)))
# # print(confusion_matrix(y_test, prediction))
# # print(classification_report(y_test, prediction))

# # #SVC
# # print("SVC")
# # transformer = ColumnTransformer(
# #     [('vec', TfidfVectorizer(ngram_range=(1,3)), 'value')],
# #     remainder='passthrough'
# # )

# # pipe = Pipeline([('vect', transformer),
# #                  ('model', SVC(kernel='poly'))])

# # model = pipe.fit(x_train, y_train)
# # prediction = model.predict(x_test)
# # print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
# # print(confusion_matrix(y_test, prediction))
# # print(classification_report(y_test, prediction, digits=4))


# # #random forest classifier
# # print("Random Forest Classifier")
# # transformer = ColumnTransformer(
# #     [('vec', TfidfVectorizer(), 'value')],
# #     remainder='passthrough'
# # )

# # pipe = Pipeline([('vect', transformer),
# #                  ('model', RandomForestClassifier())])

# # model = pipe.fit(x_train, y_train)
# # prediction = model.predict(x_test)
# # print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100, 2)))
# # print(confusion_matrix(y_test, prediction))
# # print(classification_report(y_test, prediction, digits=4))


#GradientBoostingClassifier
print("Gradient Boosting Classifier")

transformer = ColumnTransformer(
    [('vec', TfidfVectorizer(), 'value')],
    remainder='passthrough'
)

pipe = Pipeline([('vect', transformer),
                 ('model', GradientBoostingClassifier())])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100, 2)))
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction, digits=4))


t1 = time.time()

total = t1-t0
print(total)

# #Niave Baise
# print("GaussianNB")
# transformer = ColumnTransformer(
#     [('vec', TfidfVectorizer(), 'value')],
#     remainder='passthrough'
# )

# pipe = Pipeline([('vect', transformer),
#                  ('dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
#                  ('model', GaussianNB())])

# model = pipe.fit(x_train, y_train)
# prediction = model.predict(x_test)
# print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
# print(confusion_matrix(y_test, prediction))
# print(classification_report(y_test, prediction, digits=4))


# #Multinomial niave baise

# transformer = ColumnTransformer(
#     [('vec', TfidfVectorizer(), 'value')],
#     remainder='passthrough'
# )

# pipe = Pipeline([('vect', transformer),
#                  ('dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
#                  ('scale', MinMaxScaler()),
#                  ('model', MultinomialNB())])

# model = pipe.fit(x_train, y_train)
# prediction = model.predict(x_test)
# print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
# print(confusion_matrix(y_test, prediction))
# print(classification_report(y_test, prediction, digits=4))
