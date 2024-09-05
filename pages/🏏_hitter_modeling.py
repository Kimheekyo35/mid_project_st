import pandas as pd
import streamlit as st
import joblib
import os
import numpy as np
from pyparsing import empty
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

st.set_page_config(layout="wide")
empty1,col1,col3=st.columns([0.3,1.0,1.0])
data=r'./data/hitter_final2.csv'
data=pd.read_csv(data)
palette=sns.color_palette("BuGn")



def run_ml_app():
    st.title("🏏타자 예측 페이지")
    

    col1,col3=st.columns(2)
    '''
    'TB', 'H', '연차', 'AB', 'RBI', 'MH', 'R', 'BB', '2B', 'IBB', 'WAR','현재연봉구간'
    '''
    with col1:
        st.subheader("루타수(TB), 안타(H), 연차, 타수(AB), 타점(RBI), 멀티히트(MH)")
        st.subheader("득점(R), 볼넷(BB), 2루타(2B), 고의사구 (IBB), 선수 대비 승리기여도(WAR), 현재연봉구간을 입력하세요.")

        H_value=st.number_input("안타 값", 0.0,200.0)
        ab_value=st.number_input("타수(AB) 값",1,1000)
        rbi_value=st.number_input("타점(RBI) 값",0,200)
        mh_value=st.number_input("멀티히트(MH) 값",-1.00,2.00)
        r_value=st.number_input("득점(R) 값",0,200)
        bb_value=st.number_input("볼넷(BB) 값",0,110)
        year_value=st.number_input("연차",1,100)
        twob_value=st.number_input("2루타(2B) 값:",0,100)
        ibb_value=st.number_input("고의사구 (IBB) 값:",0,20)
        war_value=st.number_input("선수대비 승리기여도 (WAR) 값:")
        salary_distance=st.number_input("현재연봉구간",-3.00,10.00)
        sample=[H_value,ab_value,rbi_value,year_value,mh_value,r_value,bb_value,salary_distance,twob_value,ibb_value,war_value]
        
    with empty1:
        empty()
    
    with col3:

        st.subheader("⚾ 예측값 확인하기!")
        

        #모델 불러오기
        MODEL_PATH=r'./modeling/hitter_model.pkl'

        model=joblib.load(open(os.path.join(MODEL_PATH),'rb'))
        new_df=np.array(sample).reshape(1,-1)
        #예측값 출력 탭
        prediction=model.predict(new_df)
        st.write(prediction)
        
       
        fig=plt.figure()
        plt.xlabel('Salary')

        if prediction==0:
            st.success('연봉구간이 하위 25%에 속합니다.')
            sns.countplot(x='연봉구간',data=data,palette={'0':palette[5],'1':palette[1],'2':palette[1],'3':palette[1]})
            st.pyplot(fig)
        elif prediction==1:
            st.success('연봉구간이 하위 25%와 하위 50%에 속합니다.')
            sns.countplot(x='연봉구간',data=data,palette={'0':palette[1],'1':palette[5],'2':palette[1],'3':palette[1]})
            st.pyplot(fig)
        
        elif prediction==2:
            st.success('연봉구간이 상위 50%와 상위 75%에 속합니다.')
            sns.countplot(x='연봉구간',data=data,palette={'0':palette[1],'1':palette[1],'2':palette[5],'3':palette[1]})
            st.pyplot(fig)
          
        else:
            st.success('연봉구간이 상위 75% 이상에 속합니다.')
            sns.countplot(x='연봉구간',data=data,palette={'0':palette[1],'1':palette[1],'2':palette[1],'3':palette[5]})
            st.pyplot(fig)

    

         
    
run_ml_app()