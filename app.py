import json
import pickle

from model import Tokenizer
from flask import Flask
from flask import render_template, request, jsonify
from flask_bootstrap import Bootstrap

model = pickle.load(open('model.dump', 'rb'))
vectorizer = pickle.load(open('vectorizer.dump', 'rb'))

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def search_page():
    return render_template('index.html')


@app.route('/rate', methods=['POST'])
def get_counts():
    data = json.loads(request.data.decode())
    feedback = data['q']

    x = vectorizer.transform([feedback]).toarray()
    y = model.predict(x)[0]

    if y < 3:
        color, rating = 'red', 'Negative ({:0.3f})'.format(y)
    elif y >= 4:
        color, rating = 'green', 'Positive ({:0.3f})'.format(y)
    else:
        color, rating = 'black', 'Neutral ({:0.3f})'.format(y)

    result = {
        'color': color,
        'rating': rating
    }

    return jsonify([result])


app.run(host='127.0.0.1')
