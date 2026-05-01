import streamlit as st
import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load Dataset
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

# Reduced Data
reduced_data = training.groupby(training['prognosis']).max()

# Preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

# Train Model
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

# Dictionaries
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)

def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)

def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []

def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)
    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1
    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def calc_condition(exp, days):
    sum_severity = 0
    for item in exp:
        sum_severity += severityDictionary.get(item, 0)
    if (sum_severity * days) / (len(exp) + 1) > 13:
        return "You should take the consultation from doctor."
    else:
        return "It might not be that bad but you should take precautions."

def get_disease_and_symptoms(tree, feature_names, disease_input):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            val = 1 if name == disease_input else 0
            if val <= threshold:
                return recurse(tree_.children_left[node])
            else:
                return recurse(tree_.children_right[node])
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            return present_disease, list(symptoms_given)
    return recurse(0)

# Load data
getSeverityDict()
getDescription()
getprecautionDict()

# Streamlit UI
st.title("Healthcare Chatbot")

if 'step' not in st.session_state:
    st.session_state.step = 0

if st.session_state.step == 0:
    name = st.text_input("Your Name?")
    if st.button("Submit Name"):
        st.session_state.name = name
        st.session_state.step = 1
        st.rerun()

elif st.session_state.step == 1:
    st.write(f"Hello, {st.session_state.name}")
    symptom = st.text_input("Enter the symptom you are experiencing")
    if st.button("Submit Symptom"):
        conf, cnf_dis = check_pattern(list(cols), symptom)
        if conf == 1:
            if len(cnf_dis) > 10:
                st.error("Too many matches, please enter a more specific symptom.")
            else:
                st.session_state.cnf_dis = cnf_dis
                st.session_state.step = 2
                st.rerun()
        else:
            st.error("Enter valid symptom.")

elif st.session_state.step == 2:
    selected = st.selectbox("Select the one you meant", st.session_state.cnf_dis)
    if st.button("Confirm Symptom"):
        st.session_state.disease_input = selected
        st.session_state.step = 3
        st.rerun()

elif st.session_state.step == 3:
    days = st.number_input("From how many days?", min_value=1, step=1)
    if st.button("Submit Days"):
        st.session_state.num_days = days
        present_disease, symptoms_given = get_disease_and_symptoms(clf, cols, st.session_state.disease_input)
        st.session_state.present_disease = present_disease
        st.session_state.symptoms_given = symptoms_given
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    st.write("Are you experiencing any of these?")
    symptoms_exp = []
    for sym in st.session_state.symptoms_given:
        if st.checkbox(sym):
            symptoms_exp.append(sym)
    if st.button("Submit Symptoms"):
        st.session_state.symptoms_exp = symptoms_exp
        st.session_state.step = 5
        st.rerun()

elif st.session_state.step == 5:
    second_prediction = sec_predict(st.session_state.symptoms_exp)
    condition = calc_condition(st.session_state.symptoms_exp, st.session_state.num_days)
    st.write(condition)
    if st.session_state.present_disease[0] == second_prediction[0]:
        st.write(f"You may have {st.session_state.present_disease[0]}")
        st.write(description_list[st.session_state.present_disease[0]])
    else:
        st.write(f"You may have {st.session_state.present_disease[0]} or {second_prediction[0]}")
        st.write(description_list[st.session_state.present_disease[0]])
        st.write(description_list[second_prediction[0]])
    precaution_list = precautionDictionary[st.session_state.present_disease[0]]
    st.write("Take following measures:")
    for i, j in enumerate(precaution_list):
        st.write(f"{i+1}) {j}")
    if st.button("Start Over"):
        st.session_state.step = 0
        st.rerun()