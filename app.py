import streamlit as st

# Интерфейс
st.title("Интеллектуальный подбор вакансий")

resume_input = st.text_area("Введите текст резюме")

if st.button("Поиск вакансий"):
    if resume_input:
        # Пример вывода вакансий
        st.write("Подходящие вакансии:")
        st.write(f"Вакансия: Python Developer")
        st.write(f"Компания: XYZ Corp.")
        st.write(f"Ссылка: www.example.com")
        st.write(f"Оценка соответствия: 0.85")
    else:
        st.error("Введите текст резюме!")
