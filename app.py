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
st.set_page_config(
    page_title="AI Face Analyzer | Дипломный проект",
    page_icon="📸",
    layout="wide"
)

# --- 2. КОНФИГУРАЦИЯ СЕТИ (STUN серверы) ---
# Эти серверы помогают установить соединение между твоим браузером и облаком
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
        ]
    }
)

# --- 3. ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_trained_model():
    # Выбираем устройство (на сервере это будет CPU, на твоем Маке — MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model = MultiTaskFaceNet(arch='efficientnet_b0').to(device)
    
    # Файл весов должен лежать в той же папке на GitHub
    model_path = "best_face_model_m2.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, device

model, device = load_trained_model()

# Инициализация детектора лиц (OpenCV) и трансформаций (PyTorch)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. ФУНКЦИЯ ОБРАБОТКИ ВИДЕО ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Детекция лица
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Подготовка области лица
        roi = img[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_pil = Image.fromarray(roi_rgb)
        
        input_tensor = tf(roi_pil).unsqueeze(0).to(device)
        
        # Предсказание возраста и пола
        with torch.no_grad():
            age_p, gen_p = model(input_tensor)
            # Предполагаем, что возраст нормализован 0-1 (умножаем на 100)
            age = int(age_p.item() * 100)
            gender_idx = torch.argmax(gen_p, dim=1).item()
            gender = "Male" if gender_idx == 1 else "Female"

        # Отрисовка
        color = (46, 204, 113) if gender == "Male" else (231, 76, 60) # Зеленый / Красный
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        label = f"{gender}, {age}y"
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
st.title("🤖 Система интеллектуального анализа лиц")
st.markdown(f"Вычисления выполняются на: **{device}**")

# Боковая панель
with st.sidebar:
    st.header("Параметры")
    st.info("Разработка электронного учебника по обработке графики на базе AI.")
    st.write("**Автор:** Нурдаулет")
    st.write("**Модель:** EfficientNet-B0")
    st.divider()
    st.warning("Если камера не загружается, попробуйте сменить Wi-Fi на мобильный интернет.")

# Основной блок видео
webrtc_streamer(
    key="face-recognition",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
