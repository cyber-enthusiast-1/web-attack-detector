"""Web Attack (Threat) Detection System using ML""" 
from flask import Flask, render_template, request, url_for, redirect, flash
import os
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import numpy as np


app = Flask(__name__)
app.secret_key = os.environ.get('APP_KEY')
UPLOAD_FOLDER = 'static/upload_data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# load the deep learning model
model = load_model('models/web_attack_detection_neural_network.keras')

# load the scaler
scaler = joblib.load('models/scaler.pkl')

# load the Principal Component Analysis object
pca = joblib.load('models/pca.pkl')

# define a function to remove starting space and the columns consistent
def rename_column(column):
  return column.strip().lower().replace(' ', '_')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_data():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            expected_columns = pd.read_csv('static/template.csv')
        
            # load the file uploaded
            df = pd.read_csv(filepath)

            # rename the data columns for consistency with the trained model
            df = df.rename(columns=rename_column)

            # fill missing value with 0 value
            df.fillna(0, inplace=True)

            if 'label' in df.columns:
                df.drop(labels=['label'], axis=1, inplace=True)

            missing = [col for col in expected_columns if col not in df.columns]

            if missing:
                flash(f'The uploaded file is missing these columns: {missing}', 'danger')
                return redirect(request.url)

            # standardize the data
            X_scaled = scaler.transform(df)

            # apply the PCA to the scaled data
            X_pca = pca.fit_transform(X_scaled)

            flash(f"Data uploaded and transformed successfully! Shape: {X_pca.shape}", "success")

            # reshape the data before predicting to match the required shape in the model
            X_lstm = np.reshape(X_pca, X_pca.shape[0], X_pca.shape[1], 1)

            # make predictions
            label_predictions = model.predict(X_lstm)
            predicted_classes = np.argmax(label_predictions)
            
            # define a list of all possible attacks the model pedicts
            labels = ['Benign', 'Web Attack - Brute Force', 'Web Attack - XSS', 'Web Attack - SQL Injection']

            predicted_labels = [labels[i] for i in predicted_classes]

            # add column prediction to the data
            df['prediction'] = predicted_labels

            result_html = df[['prediction']].to_html(classes='table table-striped', index=False)

            return render_template('upload.html', table=result_html)
    
    return render_template('upload.html')



if __name__ == '__main__':
    app.run(debug=True)