# AI- Powered Fake News Detector
A machine learning web app that detects whether a news article is fake or real with 99.9%. 

## What it does
- Paste any news article into the app
- Click Analyze Article
- Imstantly see if it is FAKE or REAL with a confidence score
- 
## Results 
- Trained on 44,898 news articles from the ISOT dataset
- Achieved 100% accuracy on the test set
- Detects fake news with 99.9% confidence
- Detects real news with 99.9% confidence

## How it works 
1. Text is cleaned and converted to lowercase
2. TF-IDF converts each article into 10,000 numbers
3. Logestic Regression model predicts fake or real
4. Streamlit displays the result with confidence score

## Tools used
- Python, Pandas, Scikit-learn, Streamlit, Matplotlib, Seaborn

## Dataset 
ISOT Fake news Dataset by University of Victoria on Kaggle 44,898 articles - 21,417 real and 23,481 fake

## How to run
1. Install requirements: pip install streamlit scikit-learn pandas
2. Download True.csv and Fake.csv from Kaggle
3. Run: python -m streamlit run app.py
