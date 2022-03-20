import numpy as np
import pandas as pd
import streamlit as st


actual_patient_data = pd.read_csv('diabetes_data_upload.csv')

converted_data=pd.get_dummies(actual_patient_data, prefix=['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
       'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity', 'class'], drop_first=True)

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

def main():
   
    st.title("Diabetes Predictor")
   

    hide_st_style= """
      <div style="background-color:#green;padding:10px;width:100%;">
    <h2 style="color:white;text-align:center;">Diabetes Predictor</h2>
    </div>
    <style>
   
    </style>
    """
    st.markdown(hide_st_style,unsafe_allow_html=True)
    st.subheader("Please choose below options as per your knowledge and click on 'Predict' to know your status")
    
    age = st.number_input("What is your Age?")

    gender = st.radio("What is your Gender?",("Male","Female"))
    if gender == 'Male':
        gender = 1
    else:
        gender = 0
    
    polyuria = st.radio("Do you have Polyuria?",("Yes","No"))
    if polyuria == 'Yes':
        polyuria = 1
    else:
        polyuria = 0
    link1 = '[What is Polyuria?](https://en.wikipedia.org/wiki/Polyuria)'
    st.markdown(link1, unsafe_allow_html=True)
       
    polydipsia = st.radio("Do you have Polydipsia?",("Yes","No"))
    if polydipsia == 'Yes':
        polydipsia = 1
    else:
        polydipsia = 0
    link2 = '[What is Polydipsia?](https://en.wikipedia.org/wiki/Polydipsia)'
    st.markdown(link2, unsafe_allow_html=True)

    weight = st.radio("Recently do you observe sudden weight loss?",("Yes","No"))
    if weight == 'Yes':
        weight = 1
    else:
        weight = 0

    weakness = st.radio("Do you feel any Weekness?",("Yes","No"))
    if weakness == 'Yes':
        weakness = 1
    else:
        weakness = 0

    polyphagia = st.radio("Do you have Polyphagia?",("Yes","No"))
    if polyphagia == 'Yes':
        polyphagia = 1
    else:
        polyphagia = 0
    link3 = '[What is Polyphagia?](https://en.wikipedia.org/wiki/Polyphagia)'
    st.markdown(link3, unsafe_allow_html=True)
        
    genital_thrush = st.radio("Do you have Genital thrush?",("Yes","No"))
    if genital_thrush == 'Yes':
        genital_thrush = 1
    else:
        genital_thrush = 0
    link4 = '[What is Genital thrush?](https://www.ticahealth.org/interactive-guide/your-body/genital-problems/genital-thrush/)'
    st.markdown(link4, unsafe_allow_html=True)

    visual_blurring = st.radio("Do you have Visual blurring?",("Yes","No"))
    if visual_blurring == 'Yes':
        visual_blurring = 1
    else:
        visual_blurring = 0
    link5 = '[What is Visua blurring?](https://en.wikipedia.org/wiki/Blurred_vision)'
    st.markdown(link5, unsafe_allow_html=True)

    itching = st.radio("Do you have Itching?",("Yes","No"))
    if itching == 'Yes':
        itching = 1
    else:
        itching = 0
    link6 = '[What is Itching?](https://en.wikipedia.org/wiki/Itch)'
    st.markdown(link6, unsafe_allow_html=True) 

    irritability = st.radio("Do you have Irritability?",("Yes","No"))
    if irritability == 'Yes':
        irritability = 1
    else:
        irritability = 0
    link7 = '[What is Irritability?](https://en.wikipedia.org/wiki/Irritability)'
    st.markdown(link7, unsafe_allow_html=True)

    delayed_healing = st.radio("Do you have Delayed healing?",("Yes","No"))
    if delayed_healing == 'Yes':
        delayed_healing = 1
    else:
        delayed_healing = 0

    partial_paresis = st.radio("Do you have Partial paresis?",("Yes","No"))
    if partial_paresis == 'Yes':
        partial_paresis = 1
    else:
        partial_paresis = 0
    link8 = '[What is Paresis?](https://en.wikipedia.org/wiki/Paresis)'
    st.markdown(link8, unsafe_allow_html=True)

    muscle_stiffness = st.radio("Do you have Muscle stiffness?",("Yes","No"))
    if muscle_stiffness == 'Yes':
        muscle_stiffness = 1
    else:
        muscle_stiffness = 0

    alopecia = st.radio("Do you have Alopecia?",("Yes","No"))
    if alopecia == 'Yes':
        alopecia = 1
    else:
        alopecia = 0
    link9 = '[What is Alopecia?](https://en.wikipedia.org/wiki/Hair_loss)'
    st.markdown(link9, unsafe_allow_html=True)

    obesity = st.radio("Do you have Obesity?",("Yes","No"))
    if obesity == 'Yes':
        obesity = 1
    else:
        obesity = 0

    result=""
    if st.button("Predict"):
        result=predict_note_authentication(age,gender,polyuria,polydipsia,weight,weakness,polyphagia,genital_thrush,visual_blurring,itching,irritability, delayed_healing,partial_paresis,muscle_stiffness,alopecia,obesity)
        if result ==1:
            st.warning('You might have Diabeties. Please consult with a Doctor.')
        else:
            st.success("Hurray! You don't have Diabeties. Please consult with Doctor for verification.")
    # github = '[GitHub](https://github.com/soumyabrataroy)'
    # st.markdown(github, unsafe_allow_html=True)

if __name__=='__main__':
    main()


html_temp1 = """
    <div style="background-color:white;padding:10px">
    
    </div>
    """
st.markdown(html_temp1,unsafe_allow_html=True)

#sidebars
# st.sidebar.header("Diabeties Predictor V2")
# st.sidebar.text("Developed by Soumyabrata Roy")
# st.sidebar.text("This is just a predictor based on ML ")
# st.sidebar.text("model. Before taking any decisions, ")
# st.sidebar.text("please consult with your Doctor.")
