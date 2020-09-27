

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model_pickle = pickle.load(open('data.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model_pickle.predict(final_features)

  #  output = round(prediction[0], 2)
    if (prediction== 0):
        output = 'wont buy insurance'
    else:
        output= 'will buy insurance'

    return render_template('index.html', prediction_text='Person  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=False)

'''
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080 )
    '''