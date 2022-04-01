from flask import Flask, render_template, request
import re
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/", methods = ['POST','GET'])
def predict():
    if request.method == "POST":
        sms = [x for x in request.form.values()][0]
        print(type(sms))
        # Data Cleaning
        sms = re.sub('[^a-zA-Z]', ' ', sms) 
        sms = sms.lower()
        sms = sms.split()

        # Stemming and Generating Corpus
        user_corpus = []
        ps = PorterStemmer()
        sms = [ps.stem(word) for word in sms if not word in stopwords.words('english')]
        sms = ' '.join(sms)
        user_corpus.append(sms)

        # Applying Count Vectorizer to Corpus
        tfidf = pickle.load(open('Phase 4/vectorizer.pkl', 'rb'))
        x = tfidf.transform(user_corpus)

        # Loading ML Model
        log_model = pickle.load(open('Phase 4/log_model.pkl','rb'))
        result = log_model.predict(x)[0]
        if result == 1:
            res = "Positive"
        elif result == -1:
            res = "Negative"
        else:
            res = "Neutral"
        return render_template("index.html", csrf_token = res)
        #return redirect("/", result = res)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug = True)