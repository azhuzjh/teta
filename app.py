#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
#import pickle
import joblib
import json


app = Flask(__name__)


@app.route('/')
def helloworld():
    return 'welcome to risk scoring engine'


# score testing example: http://localhost:5000/score?pl=5.0&pw=3.0&sl=1.2&sw=2.5
# http://100.0.0.3:5000/score?pl=5.0&pw=3.0&sl=1.2&sw=2.5

#model = pickle.load(open('final_mod_v1.pkl','rb'))

#rf = joblib.load(os.path.join("./uploads/rf_model.joblib"))  ## trained model
rf = joblib.load("rf_model.joblib")

@app.route('/score', methods=['POST', 'GET'])
# @cross_origin()
def predict():
    pl = request.values['pl']
    pw = request.values['pw']  # passing prod data
    sl = request.values['sl']  # passing prod data
    sw = request.values['sw']  # passing prod data

    flower = [pl, pw, sl, sw]
    #now = datetime.now()
    #current_time = now.strftime("%H:%M:%S")

    ## below are the code to convert to json payload
    result = {}
    result['vendor'] = "vendor anem" 
    result['input_pl'] = pl
    result['input_pw'] = pw
    result['input_sl'] = sl
    result['input_sw'] = sw
    result['output'] = str(rf.predict([flower]))
    json_data = json.dumps(result)
    return json_data


if __name__ == "__main__":
    app.run(debug=True)


