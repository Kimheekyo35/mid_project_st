import pandas as pd
import streamlit as st
import joblib
import os
import numpy as np
from pyparsing import empty

st.set_page_config(layout="wide")
empty1,col1,col3=st.columns([0.3,1.0,1.0])

def run_ml_app():
    st.title("🏏타자 예측 페이지")
    
    col1,col3=st.columns(2)
    
    with col1:
        st.subheader("H, OBP, HBP, OPS")
        st.subheader("TB, BB, 연차, 현재연봉구간을 입력하세요.")

        H_value=st.number_input("H 값", 0.0,200.0)
        OBP_value=st.number_input("OBP 값",0.000,1.000)
        HBP_value=st.number_input("HBP값",1,100)
        ops_value=st.number_input("OPS+ 값",-1.00,2.00)
        tb_value=st.number_input("2B 값",0,50)
        bb_value=st.number_input("BB 값",0,110)
        year_value=st.number_input("연차",1,100)
        salary_distance=st.number_input("현재연봉구간",0,5)
        sample=[H_value,OBP_value,HBP_value,year_value,ops_value,tb_value,bb_value,salary_distance]
        
    with empty1:
        empty()
    
    with col3:

        st.subheader("⚾ 예측값 확인하기!")
        

        #모델 불러오기
        MODEL_PATH=r'./modeling/hitter_model.pkl'
        from pitcher_modeling_python import xgb_c_model


        model=joblib.load(open(os.path.join(MODEL_PATH),'rb'))
        new_df=np.array(sample).reshape(1,-1)
        #예측값 출력 탭
        prediction=model.predict(new_df)
        st.write(prediction)

        if prediction==0:
            st.success("연봉이 4500만원 미만입니다.")
        
        elif prediction==1:
            st.success("연봉이 4500만원 이상 9000만원 미만입니다.")

        elif prediction==2:
            st.success("연봉이 9000만원 이상 3억 미만입니다.")
        
        else:
            st.success("연봉이 3억 이상입니다.")
    
run_ml_app()