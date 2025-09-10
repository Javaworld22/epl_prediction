import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
from datetime import date


#with open ("C:\\Users\\USER\\Documents\\pre_model.pkl", "rb") as model_file1:
#    post_model = pickle.load(model_file1)

clubs_mapping = {
    "Arsenal": 1, "Aston Villa":2, "Brenford": 4, "Brighton": 5, "Burnley": 6 ,
    "Chelsea": 7, "Crystal Palace": 8, "Liverpool": 13, "Luton Town": 14, }

leagues = ("Arsenal", "Aston Villa", "Brenford", "Brighton", "Burnley", "Chelsea",
           "Crystal Palace", "Liverpool", "Luton Town")

seiected_option1 = st.selectbox("Select Teams for Home", leagues)


seiected_option2 = st.selectbox("Select Teams for Away", leagues)

day = st.slider("Day ", min_value = 1, max_value=31, step=1)

month = st.slider("Month ", min_value = 1, max_value=12, step=1)
year = st.slider("Year ", min_value = 2020, max_value=2026, step=1)




if st.button("Predict"):
    my_date = date(year, month, day)
    with open ("C:\\Users\\USER\\Documents\\pre_model.pkl", "rb") as model_file1:
        post_model = pickle.load(model_file1)
    st.info(seiected_option1)
    st.info(seiected_option2)
    #st.write(type(post_model))
    teams = []
    teams.append(clubs_mapping[seiected_option1])
    teams.append(clubs_mapping[seiected_option2])
    teams.append(1)
    teams.append(2)
    teams.append(year)
    teams.append(month)
    teams.append(my_date.weekday())
    teams.append(my_date.timetuple().tm_yday)
    data_1 = np.array(teams)
    mTemP = pd.DataFrame([data_1.T], columns=['Home','Away','League','Matches','Year','Month','DayOfWeek','DayOfYear'])
    st.write(mTemP)
    y_pred = post_model.predict(mTemP)
    st.write(y_pred)
    mY_pred = pd.DataFrame(y_pred)
    st.write(mTemP.iat[0,0])
    st.write(mTemP.iat[0,3])
    st.write(mY_pred.at[0,2])

    data = {
        'Home': mTemP.iat[0,0],
        'Away' : mTemP.iat[0,1],
        'League': mTemP.iat[0,2],
        'Matches':mTemP.iat[0,3],
        #'goalA': [7],
        #'goalsB': [16],
        'possA': mY_pred.iat[0,0],
        'possB': mY_pred.iat[0,1],
        'cornerA' : mY_pred.iat[0,2],
        'cornerB' : mY_pred.iat[0,3],
        'shot_goalA': mY_pred.iat[0,4],
        'shot_goalB': mY_pred.iat[0,5],
        'shot_wideA': mY_pred.iat[0,6],
        'shot_wideB': mY_pred.iat[0,7],
        's_opportA':mY_pred.iat[0,8],
        's_opportB':mY_pred.iat[0,9],
        'shotsA':mY_pred.iat[0,10],
        'shotsB':mY_pred.iat[0,11],
        'h_woodworkA':mY_pred.iat[0,12],
        'h_woodworkB':mY_pred.iat[0,13],
        'penaltyA':mY_pred.iat[0,14],
        'penaltyB':mY_pred.iat[0,15],
        'dribbleA':mY_pred.iat[0,16],
        'dribbleB':mY_pred.iat[0,17],
        's_accuracyA':mY_pred.iat[0,18],
        's_accuracyB':mY_pred.iat[0,19],
        'conversion_RA':mY_pred.iat[0,20],
        'conversion_RB':mY_pred.iat[0,21],
        'savesA': mY_pred.iat[0,22],
        'savesB':mY_pred.iat[0,23],
        'interceptA':mY_pred.iat[0,24],
        'interceptB':mY_pred.iat[0,25],
        'tackleA':mY_pred.iat[0,26],
        'tackleB':mY_pred.iat[0,27],
        't_passA':mY_pred.iat[0,28],
        't_passB':mY_pred.iat[0,29],
        'pass_accA':mY_pred.iat[0,30],
        'pass_accB':mY_pred.iat[0,31],
        'crossA':mY_pred.iat[0,32],
        'crossB':mY_pred.iat[0,33],
        'yell_cardA':mY_pred.iat[0,34],
        'yell_cardB': mY_pred.iat[0,35],
        'red_cardA':mY_pred.iat[0,36],
        'red_cardB':mY_pred.iat[0,37],
        'foulA':mY_pred.iat[0,38],
        'foulB':mY_pred.iat[0,39],
        'foul_againstA':mY_pred.iat[0,40],
        'foul_againstB':mY_pred.iat[0,9],
        'offideA':mY_pred.iat[0,9],
        'offsideB':mY_pred.iat[0,9],
        'year':[year],
        'month': [month],
        'dayofweek':mTemP.iat[0,6],
        'dayofyear':mTemP.iat[0,7]
        }
    with open ("C:\\Users\\USER\\Documents\\new_model.pkl", "rb") as model_file2:
        xgboost_model = pickle.load(model_file2)
    df =pd.DataFrame(data)
    st.dataframe(df)
    st.write("Type of object is: ")
    st.write(type(xgboost_model))
    y_pred_1000 = xgboost_model.predict(df)
    
    st.write(df)
    st.write(y_pred_1000)
    df1 =pd.DataFrame(y_pred_1000)
    st.write(df1.iat[0,0])
    if df1.iat[0,0] == 2:
        st.info("Home Win")
    elif df1.iat[0,0] == 1:
        st.info("Away Win")
    elif df1.iat[0,0] == 0:
        st.info("Draw!")
    
    
