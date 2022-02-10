from flask import Flask, render_template, request
from search import score, retrieve
from time import time
import pickle as p

app = Flask(__name__, template_folder='.')

class Document:
    def __init__(self, title, text):
        self.title = title
        self.text = text
    
    def format(self):
        return [self.title, self.text + ' ...']


with open(r'C:\Users\Anton\Desktop\проектыPython\ml\lesson9\домашка\index.pkl', 'rb') as file:
    straight_index = p.load(file)

@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')
    if not (query is None):
        documents = retrieve(query)
        #documents = sorted(documents, key=lambda doc: -score(query, doc))
        results = [straight_index[doc].format()+['%.2f' % score(doc)] for doc in documents]
    else:
        results = ''
    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Yandex',
        results=results
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
