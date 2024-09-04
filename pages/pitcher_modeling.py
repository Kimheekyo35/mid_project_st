import pandas as pd
import streamlit as st
import joblib
import os
import numpy as np
from pyparsing import empty
import xgboost as xgb

st.set_page_config(layout="wide")
empty1,col1,col3=st.columns([0.3,1.0,1.0])

def run_ml_app():
    st.title("🧤투수 예측 페이지")

    col1,col3=st.columns(2)

    with col1:
        st.subheader("ERA, QS_G, SO, WAR_x")
        st.subheader("W, IP, K_BB, exp_QS, SO_G")
        st.subheader("QS, NP, RA_9, 연차, TBF, 현재연봉")
        st.subheader(" WHIP, K-BB, NP/IP 을 입력하세요")


        ERA_value=st.number_input('ERA 값',1.30,11.00)
        R_value=st.number_input('R값',5,200)
        QS_value=st.number_input('QS 값',0,100)
        G_value=st.number_input('G 값',0,100)
        SO_value=st.number_input('SO 값',14, 230)
        WAR_x_value=st.number_input('WAR_X 값',-1.10,8.50 )
        W_value=st.number_input('W 값',0,20)
        IP_value=st.number_input('IP 값',30.0,210.0)
        BB_value=st.number_input('BB 값',0,100)
        QS_value=st.number_input('QS 값',0,30)
        NP_value=st.number_input('NP 값',500,4000)
        career_value=st.number_input('연차',1,100)
        TBF_value=st.number_input('TBF 값',140,1000)
        salary_value=st.number_input('현재 연봉',2000,1000000)
        WHIP_vlaue=st.number_input('WHIP 값',0.50,2.50)
        KBB_value=st.number_input('K-BB 값',-8.00,25.00)
        NP_IP_value=st.number_input('NP/IP 값',10.00,25.00)
        RA_9_value=st.write(R_value/IP_value*9)
        SO_G_value=st.write(SO_value/G_value)
        QS_G_value=st.write(QS_value/G_value)
        K_BB_value=st.write(SO_value/BB_value)
        exp_QS_value=st.write(QS_G_value*QS_value)
        sample=[ERA_value,QS_G_value,SO_value,WAR_x_value,W_value,IP_value,K_BB_value,exp_QS_value,SO_G_value,
                QS_value,NP_value,RA_9_value,career_value,TBF_value,salary_value,
                WHIP_vlaue,KBB_value,NP_IP_value]
        sample=np.array(sample).reshape(1,18)
        sample=pd.DataFrame(data=sample,
                            columns=['ERA','QS_G','SO','WAR_x','W','IP','K_BB','exp_QS','SO_G', 'QS', 'NP', 'RA_9', '연차', 'TBF', '현재연봉', 'WHIP', 'K-BB', 'NP/IP'])
    
    with empty1:
        empty()
    


    with col3:
        st.subheader('⚾예측값 확인하기!')

        #모델 불러오기
        MODEL_PATH='./modeling/pitcher_model.pkl'
        model=joblib.load(open(os.path.join(MODEL_PATH),'rb'))

        
        xgb_matrix=xgb.DMatrix(sample)
    

        prediction=model.predict(xgb_matrix)
        st.write(prediction)

        if prediction==0:
            st.success('연봉이 8150만원 미만입니다.')
        elif prediction==1:
            st.success('연봉이 8150만원 이상 14250만원 미만입니다.')
        elif prediction==2:
            st.success('연봉이 14250만원 이상 26125만원 미만입니다.')
        else:
            st.success('연봉이 26125만원 이상입니다.')

run_ml_app()

        

