"""Web Attack (Threat) Detection System using ML""" 
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload_data():
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)