#First, we will import necessary library for our project
from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

#We will add the dataset of patient which is in csv form
actual_patient_data = pd.read_csv('diabetes_data_upload.csv')

#We will convert the dataset in the numberical form
converted_data=pd.get_dummies(actual_patient_data, prefix=['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
       'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity', 'class'], drop_first=True)


#We will use the alogritm which can help us to get the output 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(converted_data.drop('class_Positive', axis=1),converted_data['class_Positive'], test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF_classifier.fit(X_train, y_train)

def predict_note_authentication(age,gender,polyuria,polydipsia,weight,weakness,polyphagia,genital_thrush,visual_blurring,itching,irritability, delayed_healing,partial_paresis,muscle_stiffness,alopecia,obesity):

    prediction=RF_classifier.predict(sc.transform(np.array([[int(age),int(gender),int(polyuria),int(polydipsia),int(weight),int(weakness),int(polyphagia),int(genital_thrush),int(visual_blurring),int(itching),int(irritability), int(delayed_healing),int(partial_paresis),int(muscle_stiffness),int(alopecia),int(obesity)]])))
    print(prediction)
    return prediction


@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/obesity')
def obesity():
    return render_template("obesity.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/demo')
def demo():
    return render_template("demo.html")

@app.route('/predict')
def abc():
    return render_template("predict.html")

@app.route('/predicts',methods=['POST','GET'])
def predict():
    text1 = int(request.form['1'])
    text01 = int(request.form['01'])
    text2 = int(request.form['2'])
    text3 = int(request.form['3'])
    text4 = int(request.form['4'])
    text5 = int(request.form['5'])
    text6 = int(request.form['6'])
    text7 = int(request.form['7'])
    text8 = int(request.form['8'])
    text9 = int(request.form['9'])
    text10 = int(request.form['10'])
    text11 = int(request.form['11'])
    text12 = int(request.form['12'])
    text13 = int(request.form['13'])
    text14= int(request.form['14'])
    text15= int(request.form['15'])
    # row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7,text8])])
    # print(row_df)
    # prediction=model.predict_proba(row_df)
    # output='{0:.{1}f}'.format(prediction[0][1], 2)
    # output = str(float(output)*100)+'%'
    # if output>str(0.5):
    #     return render_template('result.html',pred=f'You have chance of having diabetes.\nProbability of having Diabetes is {output}')
    # else:
    #     return render_template('result.html',pred=f'You are safe.\n Probability of having diabetes is {output}')

    result=predict_note_authentication(text1,text01,text2,text3,text4,text5,text6,text7,text8,text9,text10,text11,text12,text13,text14,text15)

    if result==1:
        return render_template('result.html',pred=f'You have chance of having diabetes.\nPLease consult your doctor')
    else:
        return render_template('result.html',pred=f'You are safe.\n There is less probablity of having diabetes ')



if __name__ == '__main__':
    app.run(debug=True)
