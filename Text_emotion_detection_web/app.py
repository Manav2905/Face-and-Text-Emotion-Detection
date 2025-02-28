from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the text emotion detection model and vectorizer
text_model = joblib.load('saved_model/text_emotion_model.pkl')
vectorizer = joblib.load('saved_model/tfidf_vectorizer.pkl')

# Define the emotion labels for text detection
text_emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

@app.route('/', methods=['GET', 'POST'])
def index():
    emotion = None
    if request.method == 'POST':
        user_input = request.form['text']
        
        # Vectorize the user input
        user_input_vec = vectorizer.transform([user_input])

        # Predict the emotion
        predicted_label = text_model.predict(user_input_vec)[0]
        emotion = text_emotion_map[predicted_label]

    return render_template('index.html', emotion=emotion)

if __name__ == '__main__':
    app.run(debug=True)