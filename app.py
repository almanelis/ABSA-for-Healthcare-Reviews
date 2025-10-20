import streamlit as st
from medreviewanalyzer import MedicalReviewAnalyzer
from medreviewanalyzer.utils.metrics import compute_metrics
import pandas as pd
from io import BytesIO
import ast  # Для безопасного парсинга строк в списки/словари
import json  # Альтернатива, если парсинг как JSON
import plotly.express as px

st.set_page_config(
    page_title="ABSA для отзывов о медицинских организациях",
    page_icon="💬",
    layout="wide"
)

# Заголовок приложения
st.title("💬 Анализ отзывов о медицинских организациях")

# Кэшируем загрузку модели, чтобы не перезапускать при каждом обновлении страницы
@st.cache_resource
def load_analyzer():
    return MedicalReviewAnalyzer(verbose=True)

analyzer = load_analyzer()


# ======== 🩵 Функция подсветки аспектов ========
def highlight_aspects(text: str, aspects: list):
    """Подсвечивает слова в зависимости от их тональности"""
    highlighted_text = text
    for asp in aspects:
        color = {
            "Positive": "#00C853",  # зелёный
            "Neutral": "#9E9E9E",   # серый
            "Negative": "#D50000"   # красный
        }.get(asp["sentiment"], "#000000")

        aspect = asp["aspect"]
        highlighted_text = highlighted_text.replace(
            aspect,
            f"<span style='background-color:{color}33; color:{color}; font-weight:bold;'>{aspect}</span>"
        )
    return highlighted_text


# ======== Вкладки Streamlit ========
tab1, tab2 = st.tabs(["🗣 Анализ одного отзыва", "📊 Анализ массива отзывов"])


# ========== 🗣 Вкладка 1: Один отзыв ==========
with tab1:
    st.subheader("🗣 Анализ одного отзыва")

    review_text = st.text_area(
        "✍️ Введите отзыв",
        height=150,
        placeholder="Например: Отличное место, специалисты внимательные и отзывчивые..."
    )

    if st.button("🔍 Проанализировать отзыв", key="single_review"):
        if review_text.strip():
            with st.spinner("Анализ отзыва..."):
                result = analyzer.analyze_review(review_text)

            st.success("✅ Анализ завершён")

            # Общая тональность
            st.subheader("📊 Общая тональность")
            st.write(f"**{result['sentiment']}**")

            # Подсветка текста
            st.subheader("🩵 Подсветка аспектов в тексте")
            highlighted = highlight_aspects(result["text"], result["aspects"])
            st.markdown(
                f"<div style='font-size:18px; line-height:1.6;'>{highlighted}</div>",
                unsafe_allow_html=True
            )

            # Таблица аспектов
            st.subheader("🩺 Анализ аспектов")
            aspects_df = pd.DataFrame(result['aspects'])
            st.dataframe(aspects_df)

            # Детали
            st.markdown("#### Детали анализа (JSON)")
            st.json(result)
        else:
            st.warning("Введите текст отзыва перед анализом.")


# ========== 📊 Вкладка 2: Таблица отзывов ==========
with tab2:
    st.subheader("📊 Анализ массива отзывов")

    uploaded_file = st.file_uploader(
        "📎 Загрузите Excel или CSV файл с отзывами",
        type=["xlsx", "csv"]
    )

    if uploaded_file is not None:
        with st.spinner("Загрузка данных..."):
            if uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            st.write(f"✅ Загружено {len(df)} отзывов")

        st.info("🧾 Укажите колонку с текстом отзывов:")
        text_column = st.selectbox("Колонка с отзывами", df.columns)

        if st.button("🚀 Начать анализ", key="batch_reviews"):
            with st.spinner("Анализируем отзывы, пожалуйста подождите..."):
                results = analyzer.analyze_dataframe(df=df[[text_column]])

            st.success("✅ Анализ завершён")

            # Проверяем, что results - DataFrame с нужными колонками
            if isinstance(results, pd.DataFrame) and all(col in results.columns for col in ['review_text', 'общая_тональность', 'аспекты']):
                # Построение графиков перед метриками
                st.subheader("📊 Графики анализа")

                # 1. Круговая диаграмма: количество положительных, отрицательных и нейтральных отзывов
                sentiment_counts = results['общая_тональность'].value_counts().reset_index()
                sentiment_counts.columns = ['Тональность', 'Количество']
                # Определяем цвета для тональностей
                color_map = {
                    'Positive': '#00C853',  # зелёный
                    'Neutral': '#9E9E9E',   # серый
                    'Negative': '#D50000'   # красный
                }
                fig1 = px.pie(
                    sentiment_counts,
                    values='Количество',
                    names='Тональность',
                    title='Распределение общей тональности отзывов',
                    color='Тональность',
                    color_discrete_map=color_map
                )
                # Настройка всплывающих подсказок
                fig1.update_traces(
                    hovertemplate='%{label}: %{value} отзывов (%{percent})'
                )
                st.plotly_chart(fig1)

                # Сбор всех аспектов
                all_aspects = []
                for aspects_value in results['аспекты']:
                    if isinstance(aspects_value, list):
                        all_aspects.extend(aspects_value)
                    elif isinstance(aspects_value, str):
                        try:
                            parsed = ast.literal_eval(aspects_value)
                            if isinstance(parsed, list):
                                all_aspects.extend(parsed)
                        except:
                            try:
                                parsed = json.loads(aspects_value)
                                if isinstance(parsed, list):
                                    all_aspects.extend(parsed)
                            except:
                                pass

                if all_aspects:
                    aspects_df = pd.DataFrame(all_aspects)

                    # 2. Столбчатая диаграмма: топ отрицательных категорий
                    negative_categories = aspects_df[aspects_df['sentiment'] == 'Negative']['category'].value_counts().head(10).reset_index()
                    negative_categories.columns = ['Категория', 'Количество']
                    if not negative_categories.empty:
                        fig2 = px.bar(
                            negative_categories,
                            x='Категория',
                            y='Количество',
                            title='Топ отрицательных категорий',
                            color_discrete_sequence=['#D50000']
                        )
                        fig2.update_traces(hovertemplate='Категория: %{x}<br>Количество: %{y}')
                        st.plotly_chart(fig2)

                    # 3. Столбчатая диаграмма: топ положительных категорий
                    positive_categories = aspects_df[aspects_df['sentiment'] == 'Positive']['category'].value_counts().head(10).reset_index()
                    positive_categories.columns = ['Категория', 'Количество']
                    if not positive_categories.empty:
                        fig3 = px.bar(
                            positive_categories,
                            x='Категория',
                            y='Количество',
                            title='Топ положительных категорий',
                            color_discrete_sequence=['#00C853']
                        )
                        fig3.update_traces(hovertemplate='Категория: %{x}<br>Количество: %{y}')
                        st.plotly_chart(fig3)

                # Отображаем отзывы в аккордеонах (expander)
                for i, row in results.iterrows():
                    expander_label = f"💬 Отзыв №{i+1} ({row['общая_тональность']})"
                    expanded = i < 3  # Первые 3 открыты по умолчанию, остальные свернуты
                    with st.expander(expander_label, expanded=expanded):
                        text = row['review_text']
                        sentiment = row['общая_тональность']

                        st.markdown(f"**🧭 Общая тональность:** `{sentiment}`")

                        # Парсинг аспектов
                        aspects_value = row['аспекты']
                        if isinstance(aspects_value, list):
                            aspects_list = aspects_value
                        elif isinstance(aspects_value, str):
                            try:
                                aspects_list = ast.literal_eval(aspects_value)
                            except (ValueError, SyntaxError):
                                try:
                                    aspects_list = json.loads(aspects_value)
                                except json.JSONDecodeError:
                                    aspects_list = []
                                    st.warning(f"⚠️ Не удалось распарсить аспекты для отзыва №{i+1}")
                        else:
                            aspects_list = []
                            st.warning(f"⚠️ Неизвестный тип для аспектов в отзыве №{i+1}")

                        # Подсвеченный текст
                        highlighted = highlight_aspects(text, aspects_list)
                        st.markdown(
                            f"<div style='font-size:17px; line-height:1.6;'>{highlighted}</div>",
                            unsafe_allow_html=True
                        )

                        # Таблица аспектов
                        if aspects_list:
                            aspects_df = pd.DataFrame(aspects_list)
                            st.markdown("**🩺 Аспекты:**")
                            st.table(aspects_df[["aspect", "sentiment", "category", "confidence"]])
                        else:
                            st.write("Аспекты не найдены.")

                # Кнопка для скачивания результатов
                buf = BytesIO()
                with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                    results.to_excel(writer, index=False)
                buf.seek(0)
                st.download_button(
                    label="📥 Скачать результаты (Excel)",
                    data=buf,
                    file_name="analyzed_reviews.xlsx",
                    mime="application/vnd.ms-excel"
                )

            else:
                st.warning("⚠️ Результат анализа не содержит ожидаемых колонок (review_text, общая_тональность, аспекты). Проверьте analyzer.analyze_dataframe.")