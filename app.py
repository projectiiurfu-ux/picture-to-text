import streamlit as st
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import io
import cv2

st.set_page_config(page_title="Японский OCR")
st.title('Распознавание японского текста с PaddleOCR')


@st.cache_resource
def load_ocr():
    return PaddleOCR(
        lang='japan',
        use_angle_cls=True
    )


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

            ocr = load_ocr()

            result = ocr.ocr(img_array)

            if result and len(result) > 0 and result[0] is not None:
                text_lines = []
                for line in result[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    text_lines.append(text)

                st.divider()
                st.write('**📄 Полный текст:**')
                st.code('\n'.join(text_lines), language='text')
            else:
                st.warning('⚠️ Текст не найден на изображении')

with st.expander("ℹ️ О модели"):
    st.markdown("""
    **PaddleOCR** с поддержкой японского языка.

    - 🗾 Специализирован на японских иероглифах
    - 📍 Определяет вертикальный и горизонтальный текст
    - ⚡ Оптимизирован для быстрого распознавания
    """)

st.markdown("---")
st.caption("PaddleOCR")
