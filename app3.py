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

# Функция для сбора вакансий с сайта HeadHunter
def get_vacancies_from_headhunter(query, num_pages=1):
    """
    Функция для сбора вакансий с сайта HeadHunter по запросу.
    :param query: Строка запроса (например, 'Python Developer')
    :param num_pages: Количество страниц для сбора вакансий
    :return: Список вакансий в формате JSON
    """
    base_url = 'https://hh.ru/search/vacancy'
    vacancies = []
    
    # Заголовки для имитации запроса от браузера Safari
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Version/13.1.2 Safari/537.36"
    }
    
    for page in range(num_pages):
        params = {'text': query, 'page': page}
        try:
            response = requests.get(base_url, params=params, headers=headers)
            # Если запрос не удался, пропускаем эту страницу
            if response.status_code != 200:
                st.error(f"Ошибка при получении страницы: {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Проверяем, есть ли вакансии на странице
            vacancy_items = soup.find_all('div', {'class': 'vacancy-serp-item'})
            
            if not vacancy_items:
                st.warning(f"Вакансии не найдены на странице {page + 1}")
            
            for vacancy in vacancy_items:
                title = vacancy.find('a', {'class': 'bloko-link'}).text.strip()
                company = vacancy.find('div', {'class': 'vacancy-serp-item__meta-info'}).text.strip()
                link = vacancy.find('a', {'class': 'bloko-link'})['href']
                
                vacancies.append({
                    'title': title,
                    'company': company,
                    'link': link
                })
            
            # Добавляем задержку, чтобы не заблокировали запросы
            time.sleep(1)
        
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка запроса: {e}")
    
    return vacancies

# Пример данных для обучения модели
resume_texts = [
    "Experienced in Python and Machine Learning", 
    "Expert in AWS and cloud technologies", 
    "Skilled in data analysis and SQL"
]
job_descriptions = [
    "Looking for a Python developer with ML expertise", 
    "Hiring for an AWS specialist", 
    "Seeking a SQL data analyst"
]
labels = [1, 1, 1]  # Метки, где 1 означает соответствие, а 0 - несоответствие

# Преобразование текста в числовые векторы
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(resume_texts + job_descriptions).toarray()

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

# Прогнозирование соответствия резюме вакансии
def predict_vacancy_match(resume, vacancies):
    """
    Прогнозирует соответствие резюме и вакансий.
    :param resume: Текст резюме
    :param vacancies: Список вакансий
    :return: Список вакансий с их оценками соответствия
    """
    resume_vector = vectorizer.transform([resume]).toarray()
    results = []
    
    for vacancy in vacancies:
        # Преобразуем вакансию в вектор
        vacancy_vector = vectorizer.transform([vacancy['title'] + ' ' + vacancy['company']]).toarray()
        
        # Модель ожидает вектор для резюме и вакансии
        match_score = model.predict(np.array([np.concatenate((resume_vector.flatten(), vacancy_vector.flatten()))]))
        
        results.append({
            'title': vacancy['title'],
            'company': vacancy['company'],
            'link': vacancy['link'],
            'match_score': match_score[0][0]
        })
    
    return results

# Интерфейс Streamlit
st.title("Интеллектуальный подбор вакансий")

# Ввод текста резюме
resume_input = st.text_area("Введите текст резюме")

# Запрос вакансий по кнопке
if st.button("Поиск вакансий"):
    if resume_input:
        query = 'Python Developer'  # Пример поиска
        num_pages = 1
        result_placeholder = st.empty()
        
        # Получаем вакансии синхронно
        vacancies = get_vacancies_from_headhunter(query, num_pages)
        
        if len(vacancies) > 0:
            # Получаем список подходящих вакансий
            vacancy_matches = predict_vacancy_match(resume_input, vacancies)
            
            st.write("Подходящие вакансии:")
            for match in vacancy_matches:
                st.write(f"**Вакансия**: {match['title']}")
                st.write(f"**Компания**: {match['company']}")
                st.write(f"**Ссылка**: {match['link']}")
                st.write(f"**Оценка соответствия**: {match['match_score']:.2f}")
                st.write("---")
        else:
            st.error("Вакансии не найдены!")
    else:
        st.error("Введите текст резюме!")
