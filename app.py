from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("dia_pred.pkl", "rb"))



@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # Pregnancies
        Pregnancies = float(request.form["Pregnancies"] )   
        # print("Pregnancies: ",Pregnancies)

        # Glucose
        Glucose = float(request.form["Glucose"] )   
        # print("Glucose: ",Glucose)

        # BloodPressure
        BloodPressure = float(request.form["BloodPressure"] )   
        # print("BloodPressure: ",BloodPressure)

        # SkinThickness
        SkinThickness = float(request.form["SkinThickness"] )   
        # print("SkinThickness: ",SkinThickness)

        # Insulin
        Insulin = float(request.form["Insulin"])    
        # print("Insulin: ",Insulin)

        # BMI
        BMI = float(request.form["BMI"])    
        # print("BMI: ",BMI)

        # DiabeteesPedigreeFun
        DiabeteesPedigreeFun = float(request.form["Diabetespedigree"] )   
        # print("Diabetees Pedigree Function: ",DiabeteesPedigreeFun)

        # Age
        Age = float(request.form["Age"])    
        # print("Age: ",Age)

        
    #     ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabeteesPedigreeFun', 'Age']
        
        prediction=model.predict([[
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabeteesPedigreeFun,
            Age
        ]])

        if prediction == 1:
            return render_template('home.html',prediction_text="Most likely to have Diabetes")
        else:
            return render_template('home.html',prediction_text="Not likely to have Diabetes")


        


    return render_template("home.html")




if __name__ == "__main__":
    app.run(debug=True)
