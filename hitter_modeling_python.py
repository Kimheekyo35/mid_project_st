import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import streamlit as st
import os
import joblib
import matplotlib.font_manager as fm

#font 오류 수정
# font_list = fm.findSystemFonts()
# font_name = None
# for font in font_list:
#     if 'AppleGothic' in font:
#         font_name = fm.FontProperties(fname=font).get_name()
# plt.rc('font', family=font_name)


warnings.filterwarnings('ignore')
hitter_data = './data/hitter_salary_debut.csv'
parkfactor = './data\pitcher_meanERA_parkfactor.csv'

df_hitter=pd.read_csv(hitter_data)

df_pf=pd.read_csv(parkfactor)

df_pf = df_pf[['연도','팀명','park_factor']].drop_duplicates().reset_index()
league = {
    '리그평균출루율' : [0.349,0.346,0.333,0.338],
    '리그평균장타율' : [0.409,0.383,0.379,0.374],
    '연도' : [2020, 2021, 2022, 2023]
}

df_league = pd.DataFrame(league)

def calculate_ops_plus(df_hitter):
    slg = df_hitter['SLG'] / (df_league.loc[df_league['연도'] == df_hitter['연도'],'리그평균장타율'].values[0])

    obp = df_hitter['OBP'] / (df_league.loc[df_league['연도'] == df_hitter['연도'],'리그평균출루율'].values[0])
    pf = df_pf.loc[(df_pf['연도']==df_hitter['연도']) & (df_pf['팀명'] == df_hitter['팀명']),'park_factor'].values[0]

    ops_plus = (slg + obp - 1) * 100 / pf
    return ops_plus

df_hitter['OPS+'] = df_hitter.apply(calculate_ops_plus, axis=1)
df = df_hitter.drop( ['연도','순위','선수명','팀명'] ,axis=1)

def assign_salary_range(row):
  if row < 4500:
    return 0
  elif row < 9000:
    return 1
  elif row < 30000:
    return 2
  else:
    return 3

df_hitter['연봉구간'] = df_hitter['후년연봉'].apply(assign_salary_range)
df_hitter['현재연봉구간'] = df_hitter['연봉(만원)'].apply(assign_salary_range)

from sklearn.model_selection import train_test_split

data = df_hitter[['H','OBP','HBP','OPS+','2B','BB','연차','현재연봉구간']]
target = df_hitter['연봉구간']

X_train, X_test, y_train, y_test = train_test_split(
    data,
    target,
    test_size=0.3,
    random_state=0
)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=6,min_samples_leaf=8, min_samples_split=8, n_estimators=300)

model.fit(X_train,y_train)

from sklearn.metrics import accuracy_score

# test 데이터 예측
y_pred = model.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}'.format(accuracy))

model_file=open("modeling/hitter_model.pkl","wb")
joblib.dump(model, model_file)
model_file.close()