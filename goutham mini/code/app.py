from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd




app = Flask(__name__)
with open('GEI_file.pkl','rb')as file:
                         model= pickle.load(file)





@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict')
def innerpage():
    return render_template("inner-page.html")

@app.route('/process',methods=["POST","GET"]) # Ensure only POST requests are accepted
def submit():
   
        # Reading the inputs given by the user from the form
        AMA_exchange_rate = float(request.form["AMA_exchange_rate"])
        Year = float(request.form["year"])
        Change_in_inventories = float(request.form["Change_in_inventories"])
        Country = int(request.form["Country"])
        ISIC_AB = float(request.form["(ISIC A-B)"])
        Population = float(request.form["Population"])
    
        # Transform the input features as required by your model
        X = [[AMA_exchange_rate, Year,Change_in_inventories,Country, ISIC_AB, Population]]

        # Make predictions using the loaded model
        prediction = model.predict(X)
        result="The IMF-Based Exchange Rate is "+str(round(prediction[0],4))
        # Provide the prediction to the template
        return render_template("portfolio-details.html", predict=result)


    



    
    
if __name__ == '__main__':
    app.run(debug=True, port=4000)
