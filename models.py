#Library for Data Preprocessing Analysing
import pandas as pd
#Library for Text Data Preprocessing 
import nltk
import re
import string
import nltk
# Library for Splitting Data into Training and Testing
from sklearn.model_selection import train_test_split
# Library for converting text into vectors
from sklearn.feature_extraction.text import TfidfVectorizer
# Library for Machine Learning Models/ Estimators
# Logisitic Regression
from sklearn.linear_model import LogisticRegression
# Support Vector Machine
from sklearn import svm
# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
# Library for Machine Learning Models/ Estimators Evaluation Pattern
from sklearn.metrics import classification_report, confusion_matrix
# Model Save and Load 
import pickle
data = pd.read_csv("clickbait_data.csv")

data.head()

data.info()

data['clickbait'].value_counts()

# Data Preprocessing

# Writing Function to remove the mentions  URL's  and String with @
def removeURL(text):
    tweet_out = re.sub(r'@[A-Za-z0-9]+','',str(text))
    re.sub('https?://[A-zA-z0-9]+','',str(text))
    return tweet_out

# Writing function to remove the non-numeric characters
def removeNonAlphanumeric(text):
    text_out = "".join([char for char in text if char not in string.punctuation])
    return text_out

data["NoURL"] = data["headline"].apply(lambda x:removeURL(x))
data["NoPun"] = data["headline"].apply(lambda x:removeNonAlphanumeric(x))

data.head()

# Tokenization

def tokenization(text):
    token = re.split("\W+",text)
    return token

data['Tokens'] = data['NoPun'].apply(lambda x:tokenization(x))

data.head(10)

# Stemming

ps = nltk.PorterStemmer()

def stemming(text):
    out_text = [ps.stem(word) for word in text]
    return out_text

data['Stem'] = data['Tokens'].apply(lambda x:stemming(x))

data.head()

# Lemmatization

nltk.download('wordnet')

wn = nltk.WordNetLemmatizer()

def lemmatize(text):
    out_text = [wn.lemmatize(word) for word in text]
    return out_text

data['Lem'] = data['Tokens'].apply(lambda x:lemmatize(x))

data.head()

# Stop Words

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

def remove_stopwords(token_list):
    text_out = [word for word in token_list if word not in stopwords]
    return text_out

data['StopWordsRemoval'] = data['Lem'].apply(lambda x:remove_stopwords(x))

def final_join(token):
    document = " ".join([word for word in token if not word.isdigit()])
    return document

data['FinalJoin'] = data['StopWordsRemoval'].apply(lambda x:final_join(x))

data.head()

# Vectorization 

X = data["FinalJoin"]
y = data["clickbait"]
cv = TfidfVectorizer(min_df=1,stop_words='english')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

len(X_train), len(X_test), len(y_train), len(y_test)

X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

# Logistic Regression

modelLR = LogisticRegression()

modelLR.fit(X_train,y_train)

y_pred = modelLR.predict(X_test)

# Accuracy Score
modelLR.score(X_test,y_test)


# Classification Report
print(classification_report(y_test,y_pred))


# Confusion Matrix
print(confusion_matrix(y_test,y_pred))


# Random Forest

modelRF = RandomForestClassifier()

modelRF.fit(X_train,y_train)

y_pred = modelRF.predict(X_test)

# Accurracy Score
modelRF.score(X_test,y_test)


# Classification Report
print(classification_report(y_test,y_pred))

# Confusion Matrix
print(confusion_matrix(y_test,y_pred))

#### Support Vector Machine

modelSVM = svm.SVC(kernel="linear")

modelSVM.fit(X_train,y_train)

y_pred = modelSVM.predict(X_test)

# Accurracy Score
modelSVM.score(X_test,y_test)


# Classification Report
print(classification_report(y_test,y_pred))

# Confusion Matrix
print(confusion_matrix(y_test,y_pred))

# Naive Bayes

modelNB = MultinomialNB()

modelNB.fit(X_train,y_train)

y_pred = modelNB.predict(X_test)

# Accurracy Score
modelNB.score(X_test,y_test)

# Classification Report
print(classification_report(y_test,y_pred))

# Confusion Matrix
print(confusion_matrix(y_test,y_pred))

# Saving All 4 Models to System and Vectorizer as well

pickle.dump(modelLR,open("LogisticRegressionModel.pkl","wb"));

pickle.dump(modelRF,open("RandomForestModel.pkl","wb"));

pickle.dump(modelSVM,open("SupportVectorMachineModel.pkl","wb"));

pickle.dump(modelNB,open("NavieBayesModel.pkl","wb"));

pickle.dump(cv,open('vectorizer.pkl','wb'))
