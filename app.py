import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import torch
import cv2
import numpy as np
from PIL import Image
from model import MultiTaskFaceNet 
from torchvision import transforms

# --- 1. НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="AI Face Analysis System", layout="wide")

# --- 2. ЗАГРУЗКА МОДЕЛИ (Локально в PyCharm) ---
@st.cache_resource
def load_model():
    # На MacBook M2 используем MPS для скорости
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiTaskFaceNet(arch='efficientnet_b0').to(device)
    
    model_path = "best_face_model_m2.pth" # Должен быть в папке проекта
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Ошибка загрузки весов: {e}")
        return None, device

model, device = load_model()

# Инструменты детекции и трансформации
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. ФУНКЦИЯ ОБРАБОТКИ (ДЛЯ ФОТО И ВИДЕО) ---
def process_image(img_array):
    # Копия для отрисовки
    draw_img = img_array.copy()
    gray = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        roi = draw_img[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_pil = Image.fromarray(roi_rgb)
        
        input_tensor = tf(roi_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            age_p, gen_p = model(input_tensor)
            age = int(age_p.item() * 100)
            gender = "Male" if torch.argmax(gen_p, dim=1).item() == 1 else "Female"

        color = (0, 255, 0) if gender == "Male" else (255, 0, 255)
        cv2.rectangle(draw_img, (x, y), (x+w, y+h), color, 3)
        cv2.putText(draw_img, f"{gender}, {age}y", (x, y-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return draw_img

# --- 4. ИНТЕРФЕЙС ---
st.sidebar.title("Настройки")
mode = st.sidebar.selectbox("Выберите режим:", ["Веб-камера", "Загрузить фото"])

st.title("👨‍🔬 Анализ лиц на базе AI")
st.write(f"Устройство: **{device}**")

if mode == "Веб-камера":
    st.subheader("Прямой эфир")
    webrtc_streamer(
        key="camera-mode",
        video_frame_callback=lambda frame: av.VideoFrame.from_ndarray(
            process_image(frame.to_ndarray(format="bgr24")), format="bgr24"
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

else:
    st.subheader("Обработка фотографии")
    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Конвертируем загруженный файл в массив OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Обрабатываем
        with st.spinner('Нейросеть думает...'):
            result_img = process_image(image)
        
        # Показываем результат (переводим обратно в RGB для Streamlit)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Результат анализа")    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- 3. ФУНКЦИЯ ОБРАБОТКИ (ДЛЯ ФОТО И ВИДЕО) ---
def process_image(img_array):
    # Копия для отрисовки
    draw_img = img_array.copy()
    gray = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        roi = draw_img[y:y + h, x:x + w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_pil = Image.fromarray(roi_rgb)

        input_tensor = tf(roi_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            age_p, gen_p = model(input_tensor)
            age = int(age_p.item() * 100)
            gender = "Male" if torch.argmax(gen_p, dim=1).item() == 1 else "Female"

        color = (0, 255, 0) if gender == "Male" else (255, 0, 255)
        cv2.rectangle(draw_img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(draw_img, f"{gender}, {age}y", (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return draw_img


# --- 4. ИНТЕРФЕЙС ---
st.sidebar.title("Настройки")
mode = st.sidebar.selectbox("Выберите режим:", ["Веб-камера", "Загрузить фото"])

st.title("👨‍🔬 Анализ лиц на базе AI")
st.write(f"Устройство: **{device}**")

if mode == "Веб-камера":
    st.subheader("Прямой эфир")
    webrtc_streamer(
        key="camera-mode",
        video_frame_callback=lambda frame: av.VideoFrame.from_ndarray(
            process_image(frame.to_ndarray(format="bgr24")), format="bgr24"
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

else:
    st.subheader("Обработка фотографии")
    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Конвертируем загруженный файл в массив OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Обрабатываем
        with st.spinner('Нейросеть думает...'):
            result_img = process_image(image)

        # Показываем результат (переводим обратно в RGB для Streamlit)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Результат анализа")
