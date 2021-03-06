#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask,render_template,request
import pickle


app = Flask(__name__)
model = pickle.load(open('RForest.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_stages',methods=['POST'])
def predict_stages():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction

    return render_template('index.html', prediction_text='Predicted stage: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

