# AI Spam Detector

A web app that uses machine learning to detect whether a text message is spam or not.

## Tech used

- Python
- scikit-learn — model training and prediction
- Flask — web server
- pandas — data handling
- HTML/CSS — web interface

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download `spam.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place it in the project folder.

3. Train the model:
   ```
   python3 train.py
   ```

4. Start the web app:
   ```
   python3 app.py
   ```

5. Open `http://127.0.0.1:5000` in your browser, paste any message, and hit Check.

## Model accuracy

97.94% on test data — precision 0.97 and recall 0.89 on spam specifically.
