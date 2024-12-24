import requests
import time
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

# Функция для сбора соискателей с платформы
def get_candidates_from_platform(query, num_pages=1):
    """
    Функция для сбора соискателей с платформы.
    :param query: Строка запроса (например, 'Python Developer')
    :param num_pages: Количество страниц для сбора данных
    :return: Список соискателей в формате JSON
    """
    base_url = 'https://hh.ru/search/resume'
    candidates = []
    
    # Заголовки для имитации запроса от браузера Safari
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Version/13.1.2 Safari/537.36"
    }
    
    for page in range(num_pages):
        params = {'text': query, 'page': page}
        try:
            response = requests.get(base_url, params=params, headers=headers)
            if response.status_code != 200:
                st.error(f"Ошибка при получении страницы: {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Проверяем, есть ли резюме на странице
            candidate_items = soup.find_all('div', {'class': 'resume-search-item'})
            
            if not candidate_items:
                st.warning(f"Соискатели не найдены на странице {page + 1}")
            
            for candidate in candidate_items:
                name = candidate.find('a', {'class': 'bloko-link'}).text.strip()
                position = candidate.find('div', {'class': 'resume-search-item__header'}).text.strip()
                link = candidate.find('a', {'class': 'bloko-link'})['href']
                skills = candidate.find('div', {'class': 'resume-search-item__skills'}).text.strip() if candidate.find('div', {'class': 'resume-search-item__skills'}) else 'Не указано'
                
                candidates.append({
                    'name': name,
                    'position': position,
                    'link': link,
                    'skills': skills
                })
            
            # Задержка между запросами
            time.sleep(1)
        
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка запроса: {e}")
    
    return candidates

# Пример данных для обучения модели
vacancy_descriptions = [
    "Looking for a Python developer with expertise in Machine Learning", 
    "Hiring for a cloud specialist with AWS knowledge", 
    "Seeking a data analyst skilled in SQL and data visualization"
]
candidate_profiles = [
    "Experienced Python developer with ML background", 
    "Expert in cloud technologies and AWS", 
    "Data analyst with strong SQL and visualization skills"
]
labels = [1, 1, 1]  # Метки, где 1 означает соответствие

# Преобразование текста в числовые векторы
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(vacancy_descriptions + candidate_profiles).toarray()

# Размерность входного слоя
input_dim = X.shape[1]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, labels * 2, test_size=0.2, random_state=42)

# Модель нейронной сети
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=2, validation_data=(np.array(X_test), np.array(y_test)))

# Прогнозирование соответствия кандидатов вакансии
def predict_candidate_match(vacancy, candidates):
    """
    Прогнозирует соответствие вакансии и кандидатов.
    :param vacancy: Текст вакансии
    :param candidates: Список соискателей
    :return: Список соискателей с их оценками соответствия
    """
    vacancy_vector = vectorizer.transform([vacancy]).toarray()
    results = []
    
    for candidate in candidates:
        candidate_vector = vectorizer.transform([candidate['position'] + ' ' + candidate['skills']]).toarray()
        match_score = model.predict(np.array([np.concatenate((vacancy_vector, candidate_vector))]))
        results.append({
            'name': candidate['name'],
            'position': candidate['position'],
            'link': candidate['link'],
            'skills': candidate['skills'],
            'match_score': match_score[0][0]
        })
    
    return results

# Интерфейс Streamlit
st.title("Интеллектуальный подбор соискателей")

# Ввод текста вакансии
vacancy_input = st.text_area("Введите текст вакансии")

# Запрос соискателей по кнопке
if st.button("Поиск соискателей"):
    if vacancy_input:
        query = 'Python Developer'  # Пример поиска
        num_pages = 1
        result_placeholder = st.empty()
        
        # Получаем соискателей синхронно
        candidates = get_candidates_from_platform(query, num_pages)
        
        if len(candidates) > 0:
            # Получаем список подходящих соискателей
            candidate_matches = predict_candidate_match(vacancy_input, candidates)
            
            st.write("Подходящие соискатели:")
            for match in candidate_matches:
                st.write(f"**Имя**: {match['name']}")
                st.write(f"**Позиция**: {match['position']}")
                st.write(f"**Навыки**: {match['skills']}")
                st.write(f"**Ссылка**: {match['link']}")
                st.write(f"**Оценка соответствия**: {match['match_score']:.2f}")
                st.write("---")
        else:
            st.error("Соискатели не найдены!")
    else:
        st.error("Введите текст вакансии!")
