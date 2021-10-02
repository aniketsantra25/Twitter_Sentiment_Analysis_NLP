# Twitter_Sentiment_Analysis_NLP

Dataset link:-  https://www.kaggle.com/kazanova/sentiment140/download

In this, we have to analyse the sentiment of tweets. Firstly we have to do DataPreProcessing, for that I have firstly removed all the punctuations from the review part of the datasegt, then used PotterStemmer under nltk for Stemming. Then used TfidfVectorizer provided by sklearn which convert a collection of text documents to a matrix of token counts. Then separated the X and y from the dataset and used testtrainsplit provided by sklearn for dividing the dataset into test and train part. Then used different models like LinearSVC, BernoulliNB, LogisticRegression and got the best accuracy by Logisitic Regression Model.

**BernoulliNB**

BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel)
y_pred1 = BNBmodel.predict(X_test)


**SVC model**

SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)
model_Evaluate(SVCmodel)
y_pred2 = SVCmodel.predict(X_test)


**Logistic Regression**


LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel)
y_pred3 = LRmodel.predict(X_test)
