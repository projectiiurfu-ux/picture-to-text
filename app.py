import streamlit as st
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import io
import cv2
import re
from datetime import datetime

st.set_page_config(page_title="Мультиязычный OCR")
st.title('Распознавание текста с PaddleOCR')

# Выбор языка в боковой панели
st.sidebar.header("⚙️ Настройки")

# Правильные коды языков для PaddleOCR
language_options = {
    'Японский': 'japan',
    'Китайский (упрощенный)': 'ch',
    'Китайский (традиционный)': 'chinese_cht',
    'Английский': 'en',
    'Корейский': 'korean',
    'Русский (кириллица)': 'cyrillic',
    'Тамильский': 'ta',
    'Телугу': 'te',
    'Каннада': 'ka',
    'Латиница': 'latin',
    'Арабский': 'arabic',
    'Деванагари': 'devanagari'
}

selected_lang_name = st.sidebar.selectbox(
    'Выберите язык распознавания',
    list(language_options.keys())
)
language = language_options[selected_lang_name]


@st.cache_resource
def load_ocr(lang):
    return PaddleOCR(
        lang=lang,
        use_angle_cls=True
    )


def count_characters(text):
    """Подсчет символов в тексте"""
    total_chars = len(text)
    chars_no_spaces = len(text.replace(' ', '').replace('\n', ''))
    words = len(re.findall(r'\b\w+\b', text))
    return total_chars, chars_no_spaces, words


def create_download_text(full_text, text_lines, selected_lang, language_code, stats):
    """Создание текста для скачивания с метаданными"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    download_content = f"""=== РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ ===
Дата: {timestamp}
Язык распознавания: {selected_lang} ({language_code})
Количество строк: {stats['lines']}
Всего символов: {stats['total_chars']}
Символов без пробелов: {stats['chars_no_spaces']}
Количество слов: {stats['words']}
{'=' * 40}

РАСПОЗНАННЫЙ ТЕКСТ:
{full_text}

{'=' * 40}
"""
    return download_content


uploaded_file = st.file_uploader('Выберите изображение', type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    st.image(image, use_column_width=True)

    if st.button('🔍 Распознать'):
        with st.spinner('Распознаю текст...'):
            img_array = np.array(image)

            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

            ocr = load_ocr(language)
            result = ocr.ocr(img_array)

            if result and len(result) > 0 and result[0] is not None:
                text_lines = []
                for line in result[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    text_lines.append(text)

                full_text = '\n'.join(text_lines)

                # Подсчет символов
                total_chars, chars_no_spaces, words_count = count_characters(full_text)

                # Статистика для скачивания
                stats = {
                    'lines': len(text_lines),
                    'total_chars': total_chars,
                    'chars_no_spaces': chars_no_spaces,
                    'words': words_count
                }

                # Отображение статистики
                st.subheader("📊 Статистика распознавания")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📝 Строк", len(text_lines))
                with col2:
                    st.metric("🔤 Всего символов", total_chars)
                with col3:
                    st.metric("📖 Символов (без пробелов)", chars_no_spaces)
                with col4:
                    st.metric("💬 Слов", words_count)

                st.divider()
                st.write('**📄 Полный текст:**')
                st.code(full_text, language='text')

                # Кнопка скачивания
                st.divider()
                download_content = create_download_text(full_text, text_lines, selected_lang_name, language, stats)

                # Генерация имени файла
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recognized_text_{timestamp}.txt"

                st.download_button(
                    label="💾 Скачать результат (TXT)",
                    data=download_content.encode('utf-8'),
                    file_name=filename,
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.warning('⚠️ Текст не найден на изображении')

with st.expander("ℹ️ О модели"):
    st.markdown("""
    **PaddleOCR** с поддержкой множества языков.

    | Язык | Код |
    |------|-----|
    | Японский | japan |
    | Китайский (упрощенный) | ch |
    | Китайский (традиционный) | chinese_cht |
    | Английский | en |
    | Корейский | korean |
    | Русский (кириллица) | cyrillic |
    | Тамильский | ta |
    | Телугу | te |
    | Каннада | ka |
    | Латиница | latin |
    | Арабский | arabic |
    | Деванагари | devanagari |

    - 📍 Определяет вертикальный и горизонтальный текст
    - ⚡ Оптимизирован для быстрого распознавания
    """)

st.markdown("---")
st.caption(f"PaddleOCR | Текущий язык: {selected_lang_name} ({language})")