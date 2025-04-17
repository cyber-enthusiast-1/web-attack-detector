"""Web Attack (Threat) Detection System using ML""" 
from flask import Flask, render_template, request, url_for, redirect, flash, session
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' = Filter INFO and WARNING logs

from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio


# load the environment variables from .env
load_dotenv()


app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

MODEL_PATH = os.getenv('MODEL_PATH')
PCA_PATH = os.getenv('PCA_PATH')
SCALER_PATH = os.getenv('SCALER_PATH')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not MODEL_PATH:
    raise ValueError("MODEL_PATH environment variable not set!")

# load the deep learning model
model = load_model(MODEL_PATH)

# load the scaler
scaler = joblib.load(SCALER_PATH)

# load the Principal Component Analysis object
pca = joblib.load(PCA_PATH)

# define a function to remove starting space and the columns consistent
def rename_column(column):
  return column.strip().lower().replace(' ', '_')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_data():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        if file.filename.endswith('.csv'):
            tmp_dir = '/tmp'
            filepath = os.path.join(tmp_dir, file.filename)
            file.save(filepath)

            # read the file
            df = pd.read_csv(filepath)

            session['filename'] = file.filename
            session['rows'] = df.shape[0]
            session['cols'] = df.shape[1]

            # save the file path to session
            session['file_uploaded'] = filepath
            # flash('File Uploaded Successfully!', 'success')
            return redirect(url_for('upload_success'))

        flash('Please upload a valid CSV file', 'danger')
        return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/upload-success')
def upload_success():
    filename = session.get('filename')
    rows = session.get('rows', 0)
    cols = session.get('cols',0)

    return render_template('upload_success.html', filename=filename, rows=rows, cols=cols)

@app.route('/predict')
def predict():
    filepath = session.get('file_uploaded')

    if not filepath or not os.path.exists(filepath):
        flash("No file uploaded. Please upload a file first.", "warning")
        return redirect(url_for('upload_data'))

    # read the data
    df = pd.read_csv(filepath)
    expected_columns = list(pd.read_csv('static/template.csv').columns)
    # print(f'expected_columns: {expected_columns}')
    # df.columns = expected_columns
    # print(f'data columns: {df.columns}')
    

    # rename the columns to match columns in training and for consistency
    df.rename(columns=rename_column)

    # replace infinite value with NaN value
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True) # replace missing values with 0

    if 'label' in df.columns:
        df.drop('label', axis=1, inplace=True)
    
    # check for missing columns
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        flash(f'Missing columns: {missing[:3]} and others. Please follow the template.', 'danger')
        return redirect(url_for('upload_data'))

    df = df[expected_columns] # index out the required columns
    
    # transform and reshape data
    X_scaled = scaler.transform(df)
    X_pca = pca.transform(X_scaled)
    X_lstm = X_pca.reshape((X_pca.shape[0], X_pca.shape[1], 1))
    
    # predict
    label_predictions = model.predict(X_lstm)
    predicted_classes = np.argmax(label_predictions, axis=1)
    labels = ['Benign', 'Web Attack - Brute Force', 'Web Attack - XSS', 'Web Attack - SQL Injection']
    predicted_labels = [labels[i] for i in predicted_classes]

    # add column to the original data
    df['prediction'] = predicted_labels
    result_html = df[['prediction']].to_html(classes='table table-striped', index=False)

    # save the file with predicted values
    labeled_path = filepath.replace('.csv', '_labeled.csv')
    df.to_csv(labeled_path, index=False)

    # add the labeled data to session
    session['labeled_data'] = labeled_path

    flash('Prediction complete. View Threat Statistics Summary for Insights.', 'success')

    return render_template('predict.html', table=result_html)

# route for model information
@app.route('/model')
def model_info():
    # model summary
    model_details = {
        "architecture": [
            "Input → 78 features → StandardScaler",
            "Principal Component Analysis → 20 features",
            "Long-Short Term Memory layers",
            "Dense output (4 classes)"
        ],
        "accuracy": "81.81%",
        "loss_function": "sparse_categorical_crossentropy",
        "optimizer": "adam",
        "output_classes": ["Benign", "Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - SQL Injection"],
        "pca_components": 20,
        "scaling": "StandardScaler"
    }
    return render_template('model_info.html', model=model_details)

@app.route('/threat-summary')
def threat_summary():
    summary = dict()  # create an empty dictionary
    filepath = session.get('labeled_data')  # fetch the last file upload from session
    bar_plot_div = ''

    if not filepath or not os.path.exists(filepath):
        flash('No uploaded data found! Please upload a dataset first', 'warning')
        return redirect(url_for('upload_data'))

    df = pd.read_csv(filepath)

    if 'prediction' not in df.columns:
        flash('Predictions not found in the uploaded data.Please ensure the prediction step is completed.', 'danger')
        return redirect(url_for('predict'))
    
    summary = df['prediction'].value_counts().to_dict()

    fig = go.Figure(data=[go.Bar(
        x=list(summary.keys()),
        y=list(summary.values()),
        marker_color='indianred'
    )])

    fig.update_layout(
        title='Threat Class Distribution',
        xaxis_title='Threat Class',
        yaxis_title='Count'
    )

    # generate a HTML div for embedding
    bar_plot_div = pio.to_html(fig, full_html=False)

    return render_template('threat.html', summary=summary, bar_plot_div=bar_plot_div)

if __name__ == '__main__':
    app.run(debug=False)