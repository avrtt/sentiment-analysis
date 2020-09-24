import joblib

class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load('/home/lenferdetroud/Desktop/demo/clf.joblib')
        self.vectorizer = joblib.load('/home/lenferdetroud/Desktop/demo/vectorizer.joblib')
        self.classes_dict = {0: 'Negative', 1: 'Positive', -1: 'Error'}

    @staticmethod
    def get_probability_words(probability):
        if round(probability, 2) < 0.70:
            return 'confidence: < 70% (probably neutral)'

        if round(probability, 2) == 0.71:
            return 'confidence: 71%'
        if round(probability, 2) == 0.72:
            return 'confidence: 72%'
        if round(probability, 2) == 0.73:
            return 'confidence: 73%'
        if round(probability, 2) == 0.74:
            return 'confidence: 74%'
        if round(probability, 2) == 0.75:
            return 'confidence: 75%'
        if round(probability, 2) == 0.76:
            return 'confidence: 76%'
        if round(probability, 2) == 0.77:
            return 'confidence: 77%'
        if round(probability, 2) == 0.78:
            return 'confidence: 78%'
        if round(probability, 2) == 0.79:
            return 'confidence: 79%'
        if round(probability, 2) == 0.80:
            return 'confidence: 80%'
        if round(probability, 2) == 0.81:
            return 'confidence: 81%'
        if round(probability, 2) == 0.82:
            return 'confidence: 82%'
        if round(probability, 2) == 0.83:
            return 'confidence: 83%'
        if round(probability, 2) == 0.84:
            return 'confidence: 84%'
        if round(probability, 2) == 0.85:
            return 'confidence: 85%'
        if round(probability, 2) == 0.86:
            return 'confidence: 86%'
        if round(probability, 2) == 0.87:
            return 'confidence: 87%'
        if round(probability, 2) == 0.88:
            return 'confidence: 88%'
        if round(probability, 2) == 0.89:
            return 'confidence: 89%'
        if round(probability, 2) == 0.90:
            return 'confidence: 90%'
        if round(probability, 2) == 0.91:
            return 'confidence: 91%'
        if round(probability, 2) == 0.92:
            return 'confidence: 92%'
        if round(probability, 2) == 0.93:
            return 'confidence: 93%'
        if round(probability, 2) == 0.94:
            return 'confidence: 94%'
        if round(probability, 2) == 0.95:
            return 'confidence: 95%'
        if round(probability, 2) == 0.96:
            return 'confidence: 96%'
        if round(probability, 2) == 0.97:
            return 'confidence: 97%'
        if round(probability, 2) == 0.98:
            return 'confidence: 98%'
        if round(probability, 2) == 0.99:
            return 'confidence: 99%'
        if round(probability, 2) == 1.00:
            return 'confidence: 100%'
        else:
            return 'undefined confidence'

    def predict_text(self, text):
        try:
            vectorized = self.vectorizer.transform([text])
            return self.model.predict(vectorized)[0],\
                   self.model.predict_proba(vectorized)[0].max()
        except:
            print('Error')
            return -1, 0.8

    def predict_list(self, list_of_texts):
        try:
            vectorized = self.vectorizer.transform(list_of_texts)
            return self.model.predict(vectorized),\
                   self.model.predict_proba(vectorized)
        except:
            print ('Error')
            return None

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        prediction_probability = prediction[1]
        return self.classes_dict[class_prediction] + ', ' + self.get_probability_words(prediction_probability)
