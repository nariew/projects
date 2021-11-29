from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle
import model

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

X_test = np.array([[
                        float(6.8), 
                        float(0.26), 
                        float(0.32),
                        float(5.2), 
                        float(0.043), 
                        float(34),
                        float(134), 
                        float(0.9937), 
                        float(3.18),
                        float(0.47), 
                        float(10.4)
                    ]])
y_test = model.predict(X_test)

print('test if the model has been loaded successfully!')
print(y_test[0])

@app.route('/', methods=["GET", "POST"])
def index():
    outputStr=''

    if request.method == "POST":
        fixed_acidity = request.form["fixed_acidity"]
        volatile_acidity = request.form["volatile_acidity"]
        citric_acid = request.form["citric_acid"]
        residual_sugar = request.form["residual_sugar"]
        chlorides = request.form["chlorides"]
        free_sulfur_dioxide = request.form["free_sulfur_dioxide"]
        total_sulfur_dioxide = request.form["total_sulfur_dioxide"]
        density = request.form["density"]
        pH = request.form["pH"]
        sulphates = request.form["sulphates"]
        alcohol = request.form["alcohol"]
        X = np.array([[
                        float(fixed_acidity), 
                        float(volatile_acidity), 
                        float(citric_acid),
                        float(residual_sugar), 
                        float(chlorides), 
                        float(free_sulfur_dioxide),
                        float(total_sulfur_dioxide), 
                        float(density), 
                        float(pH),
                        float(sulphates), 
                        float(alcohol)
                    ]])
        y_pred = model.predict(X)[0]
        
        if y_pred==0:
            outputStr='medium or low'
        else:
            outputStr='high'
        
        
    return render_template("index.html", pred=outputStr)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)