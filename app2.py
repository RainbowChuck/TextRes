import streamlit as st
# Интерфейс Streamlit
st.title("Интеллектуальный подбор вакансий")

# Ввод текста резюме
resume_input = st.text_area("Введите текст резюме")

# Кнопка для поиска вакансий
if st.button("Поиск вакансий"):
    if resume_input:
        # Получаем вакансии с сайта HeadHunter
        vacancies = get_vacancies_from_headhunter('Python Developer', num_pages=1)
        
        # Прогнозируем соответствие резюме и вакансий
        vacancy_matches = predict_vacancy_match(resume_input, vacancies)
        
        # Отображаем результат
        st.write("Подходящие вакансии:")
        
        for match in vacancy_matches:
            st.write(f"Вакансия: {match['title']}")
            st.write(f"Компания: {match['company']}")
            st.write(f"Ссылка: {match['link']}")
            st.write(f"Оценка соответствия: {match['match_score']:.2f}")
            st.write("---")
    else:
        st.error("Введите текст резюме!")
