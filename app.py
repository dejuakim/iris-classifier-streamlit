import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 1. 모델 불러오기
model = joblib.load('iris_model.pkl')

# 2. 제목 및 소개
st.set_page_config(page_title="Iris Classifier", layout="centered")
st.title("🌸 Iris Flower Classifier")
st.markdown("---")

# 3. 사이드바 입력창
st.sidebar.header("User Input Parameters")
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()

# 4. 모델 예측 및 확률 계산
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

target_names = ['Setosa', 'Versicolor', 'Virginica']
predicted_name = target_names[prediction[0]]

# 5. 결과 출력
st.subheader('🔮 Prediction Result')

# 큰 글씨로 결과 표시
st.success(f"이 꽃은 **{predicted_name}**일 확률이 가장 높습니다!")

# 예측 신뢰도를 시각화
st.write("### Species Probability")

# 그래프 데이터 준비
proba_df = pd.DataFrame({
    'Species': target_names,
    'Probability': prediction_proba[0] * 100
})

# 플라워 느낌의 커스텀 컬러맵 (Setosa: 핑크, Versicolor: 연보라, Virginica: 진보라)
flower_colors = {
    'Setosa': '#FFB7C5',      # 벚꽃 핑크
    'Versicolor': '#DCD0FF',  # 라벤더
    'Virginica': '#9370DB'    # 미디엄 퍼플
}

# Plotly 막대 그래프 생성
fig = px.bar(
    proba_df, 
    x='Species', 
    y='Probability',
    color='Species',
    color_discrete_map=flower_colors, # 여기서 색상을 지정합니다!
    text_auto='.1f'
)

# 그래프 디자인 다듬기 (미니멀하게)
fig.update_layout(
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    yaxis_title="확률 (%)",
    xaxis_title="",
    font=dict(size=14)
)

st.plotly_chart(fig, use_container_width=True)