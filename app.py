from flask import Flask,render_template,request
import pandas as pd
import pickle

app=Flask(__name__)

def prediction_input_data(input_df):
    trsn1=pickle.load(open("columntrans.pkl","rb"))
    vc=pickle.load(open("model.pkl","rb"))

    x=trsn1.fit_transform(input_df)

    ans=vc.predict(x)[0]

    if ans == 1:
        return "You have a strong chance of securing loan.\n🤓"
    else:
        return "it's possible that you may not get loan.😟"
    
@app.route("/")
def display_form():
    return render_template("home.html")

# ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
#        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#        'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']
@app.route("/predict",methods=["POST"])
def get_input_data():
    Gender=request.form.get("Gender")
    Married=request.form.get("Married")
    Dependents=float(request.form.get("Dependents"))
    Education=request.form.get("Education")
    Self_Employed=request.form.get("Self_Employed")
    ApplicantIncome=float(request.form.get("ApplicantIncome"))
    CoapplicantIncome=float(request.form.get("CoapplicantIncome"))
    LoanAmount=float(request.form.get("LoanAmount"))
    Loan_Amount_Term=float(request.form.get("Loan_Amount_Term"))
    Credit_History=float(request.form.get("Credit_History"))
    Property_Area=request.form.get("Property_Area")

    inpu_df=pd.DataFrame(data=[[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome, CoapplicantIncome, LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]],
                         columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History', 'Property_Area'])

    ans=prediction_input_data(inpu_df)
    return render_template("display.html",data=ans)


if __name__=="__main__":
    app.run(debug=True)


    


