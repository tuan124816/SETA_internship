# 🤖 Week 4: Transformers and LLMs

## 🧠 Overview
This is a web application for text classification using pre-trained models. Users can input text or upload a CSV file, and the app will classify each entry as **Positive** or **Negative** sentiment with a confidence score.

## 🛠️ Engineering Practice
- Hugging Face Transformers
- Sklearn for simple NLP models

## 📌 Project
Fine-tune a pre-trained language model on a custom dataset

### Supported Models
- Logistic Regression
- SVM (Support Vector Machine)
- Naive Bayes
- Transformer (DistilBERT from Hugging Face)

## Requirements
- **Python**: 3.11.2  
- Install dependencies using the command:
  ```bash
  pip install -r requirements.txt

## 📂 Folder Structure
 
    week04_transformers_llms
    ├── ... 
    ├── notebooks               # Jupyter notebooks for model exploration and training
    │   ├── huggingface_pretrained.ipynb
    │   ├── logistic_regression.ipynb 
    │   ├── naive_bayes.ipynb        
    │   ├── svm.ipynb                   
    ├── models                  # Pre-trained models
    │   ├── logistic_regression.pkl              
    │   ├── logistic_vectorizer.pkl 
    │   ├── naive_bayes.pkl             
    │   ├── nb_vectorizer.pkl
    │   ├── svm_vectorizer.pkl             
    │   ├── svm.pkl                   
    ├── templates               # HTML templates for the web app
    │   ├── index.html                   
    ├── dataset                 # Dataset files (ignored by git)
    │   ├── imdb_dataset.csv   
    ├── app.py                 # Main application file  
    └── README.md
    └── requirements.txt         # Python dependencies
    └── sample_input.csv         # Sample input file for testing

## 📥 Installation
1. **Activate your virtual environment (if needed):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

2. **Install the required packages:**

   ```bash
    pip install -r requirements.txt
    ```
3. **Run the application:**
    ```bash
    python app.py
    ```
4. **Access the app:**
    Open your web browser and navigate to the following URL to access the application:
    
    ```
    http://127.0.0.1:5000/
    ```

## 📄 Usage
1. **Input Text**: Type or paste your text into the input box.
2. **Upload CSV**: Alternatively, you can upload a CSV file with a column named `text` containing the text entries to classify.
3. **Submit**: Click the "Classify" button to get the sentiment classification.
4. **Results**: The app will display the classification results, including the sentiment label (Positive/Negative) and confidence score for each entry.
5. **Download Results**: You can download the results as a CSV file.

## 📄 Example Input (in .csv case)
```csv
text
I love this movie!
I hate this movie!
This is the worst experience ever.
This is the best experience ever.
```

## 📄 Example Output
| text                                | label    | confidence |
|-------------------------------------|----------|------------|
| I love this movie!                  | Positive | 95%        |
| I hate this movie!                  | Negative | 90%        |
| This is the worst experience ever.  | Negative | 85%        |
| This is the best experience ever.   | Positive | 92%        |

## Data for Training
The models were trained using the IMDB Movie Reviews dataset, which contains 50,000 labeled movie reviews for sentiment analysis (positive or negative).  
You can download the dataset from Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)  
Once downloaded, place the `IMDB Dataset.csv` file into the `dataset` folder.

## 📄 License
Will write a license file later.

## 📄 Contributing
Will write a contributing guide later.

## 📄 Acknowledgements  
- Thanks to the Hugging Face community for providing pre-trained models.
- Thanks to the Scikit-learn community for providing simple NLP models examples.
- Thanks to Kaggle for the dataset used in this project.
- Thanks OpenAI for the help with HTML user interface.

## 📄 Contact
For questions or support, contact [tuan124817@example.com].