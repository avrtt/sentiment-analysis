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
        
        '''
        if round(probability, 2) == 0.71:
            return 'confidence: 71%'
            
            ... // that was the shittiest code i've ever written, do for instead 
        
        if round(probability, 2) == 1.00:
            return 'confidence: 100%'
        else:
            return 'undefined confidence'
            '''

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
