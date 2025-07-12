from flask import Flask, render_template, request, send_file
from transformers import pipeline
import pickle
import re
import pandas as pd
from io import BytesIO


app = Flask(__name__)

# Global variable to store batch prediction results
session_df = None

def preprocess(text):
    """
    Preprocess input text by removing HTML tags, punctuation, non-alphabetic characters,
    trimming whitespace, and converting to lowercase.
    
    Args:
        text (str): Raw input text.
    
    Returns:
        str: Cleaned and normalized text.
    """

    text = re.sub('(<.*?>)', ' ', text)                  # Remove markup
    text = re.sub('[,\.!?:()"]', '', text)               # Remove punctuation
    text = text.strip()                                  # Remove leading/trailing whitespace
    text = re.sub('[^a-zA-Z"]',' ', text)                # Keep letters only
    return text.lower()                                  # Convert to lowercase

MODEL_PATHS = {
    "Logistic Regression": (r"models\logistic_regression.pkl", r"models\logistic_vectorizer.pkl"),
    "SVM": (r"models\svm.pkl", r"models\svm_vectorizer.pkl"),
    "Naive Bayes": (r"models\naive_bayes.pkl", r"models\nb_vectorizer.pkl"),
    "Transformer": (),      # # Transformer model handled separately
}

@app.route('/', methods=['GET', 'POST'])

def index():
    """
    Handle text classification requests.
    Supports single text input or batch CSV upload.
    
    - Loads the selected model.
    - Preprocesses input text.
    - Runs prediction and computes confidence scores.
    - Displays results in the webpage.
    - Stores batch results in memory for optional CSV download.
    
    Returns:
        Rendered HTML template with classification results, errors, and UI data.
    """

    global session_df

    result = None
    confidence = None
    color = None
    error = None
    selected_model = request.form.get('model')
    table = None
    download_ready = False

    if request.method == 'POST':
        user_input = request.form['text_input'].strip()
        uploaded_file = request.files.get('file')

        try:
            # Batch prediction from uploaded CSV
            if uploaded_file and uploaded_file.filename.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    error = "CSV must contain a 'text' column."
                else:
                    texts = df['text'].fillna('').astype(str).tolist()

                    if selected_model == "Transformer":
                        # Hugging Face pipeline for batch sentiment analysis
                        model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
                        results = model(texts)
                        df['label'] = [r['label'].capitalize() for r in results]
                        df['confidence'] = [f"{r['score']:.2%}" for r in results]

                    else:
                        # Load pre-trained sklearn model and vectorizer
                        model_path, vectorizer_path = MODEL_PATHS[selected_model]
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        with open(vectorizer_path, 'rb') as f:
                            vectorizer = pickle.load(f)

                        processed = [preprocess(text) for text in texts]
                        print('got 2')
                        X = vectorizer.transform(processed)
                        preds = model.predict(X)
                        probs = model.predict_proba(X).max(axis=1)

                        df['label'] = ['Positive' if p == 1 else 'Negative' for p in preds]
                        df['confidence'] = [f"{p:.2%}" for p in probs]

                    table = df.to_html(classes='data', header="true", index=False)
                    session_df  = df  # save to global memory temporarily for dowload
                    download_ready = True

            # Single text input prediction
            elif user_input:
                if len(user_input.split()) < 3:
                    error = "⚠️ Warning: Text is too short for reliable prediction."
                else:
                    if selected_model == "Transformer":
                        model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
                        prediction = model([user_input])[0]
                        result, confidence = prediction['label'].capitalize(), f"{prediction['score']:.2%}"
                        color = "green" if result == "Positive" else "red"
                    else:
                        model_path, vectorizer_path = MODEL_PATHS[selected_model]
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        with open(vectorizer_path, 'rb') as f:
                            vectorizer = pickle.load(f)

                        X = vectorizer.transform([preprocess(user_input)])
                        pred = model.predict(X)[0]
                        conf = model.predict_proba(X).max()
                        result = "Positive" if pred == 1 else "Negative"
                        confidence = f"{conf:.2%}"
                        color = "green" if pred == 1 else "red"
            else:
                error = "❌ Error: Please input text or upload a CSV file."

        except Exception as e:
            error = f"Unexpected error: {e}"

    return render_template('index.html', result=result, 
                                         confidence=confidence, 
                                         color=color, 
                                         error=error,
                                         model_options=MODEL_PATHS.keys(), 
                                         selected_model=selected_model,
                                         table=table,
                                         download_ready=download_ready)

@app.route('/download')
def download():
    """
    Endpoint to download the batch prediction results as a CSV file.

    Returns:
        Response to download CSV file if available, else error message.
    """

    global session_df

    if session_df is None:
        return "No data to download.", 400

    output = BytesIO()
    session_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='predictions.csv')

if __name__ == '__main__':
    # Run the Flask development server with debug enabled
    app.run(debug=True)
