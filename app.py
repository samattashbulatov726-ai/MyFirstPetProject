import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Movie Revenue Predictor", layout="centered")

st.title("🔮 Прогнозирование кассовых сборов")
st.write("Заполните данные о фильме ниже, чтобы нейросеть (XGBoost/RandomForest) предсказала его доход.")
col1, col2 = st.columns(2)

with col1:
    title = st.text_input("Название фильма", "Супер Хит")
    budget = st.number_input("Бюджет ($)", min_value=0, value=20000000)
    popularity = st.number_input("Популярность", min_value=0.0, value=25.0)
    vote_average = st.slider("Оценка", 1.0, 10.0, 6.5, 0.1)

with col2:
    vote_count = st.number_input("Количество оценок", min_value=0, value=300)
    genres = st.selectbox("Жанр", ["Action", "Comedy", "Drama", "Science Fiction", "Thriller", "None"])
    original_language = st.selectbox("Язык", ["en", "ru", "fr", "ja", "es"])
    release_date = st.date_input("Дата выхода")
if st.button("🚀 Предсказать кассовые сборы", use_container_width=True):
    model_path = 'best_movies_pipeline.joblib'
    if not os.path.exists(model_path):
        st.error(f"Файл '{model_path}' не найден! Сначала запусти свой скрипт обучения и сохрани модель.")
    else:
        model = joblib.load(model_path)
        input_data = pd.DataFrame([{
            'title': title,
            'budget': budget,
            'release_date': str(release_date),
            'vote_average': vote_average,
            'vote_count': vote_count,
            'popularity': popularity,
            'original_language': original_language,
            'genres': genres
        }])
        prediction = model.predict(input_data)[0]
        st.markdown("---")
        st.subheader("📊 Результат анализа:")
        if prediction > budget:
            st.success(f"Прогнозируемый доход: **${prediction:,.2f}** 📈 Фильм окупается!")
        else:
            st.error(f"Прогнозируемый доход: **${prediction:,.2f}** 📉 Риск убытков.")