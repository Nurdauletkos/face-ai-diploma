import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av  # Библиотека для работы с видео-фреймами
import torch
import cv2
import numpy as np
from PIL import Image
from model import MultiTaskFaceNet  # Твой файл model.py должен быть рядом
from torchvision import transforms

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(page_title="AI Age & Gender Detector", layout="wide")

# Настройка для работы через интернет (STUN серверы Google)
# Это нужно, чтобы ngrok или облако могли "пробить" путь к камере
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def get_model():
    # Проверяем доступность M2 GPU (MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiTaskFaceNet(arch='efficientnet_b0').to(device)

    # УКАЖИ ПРАВИЛЬНОЕ ИМЯ ФАЙЛА ВЕСОВ ТУТ:
    model_path = "best_face_model_m2.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Ошибка загрузки весов: {e}")
        return None, device


model, device = get_model()

# Инициализация детектора лиц и трансформов
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- ФУНКЦИЯ ОБРАБОТКИ КАДРА ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")  # Конвертируем в формат OpenCV

    # 1. Поиск лиц
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        # 2. Подготовка лица для нейросети
        roi = img[y:y + h, x:x + w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_pil = Image.fromarray(roi_rgb)

        input_tensor = tf(roi_pil).unsqueeze(0).to(device)

        # 3. Предсказание нейросетью
        with torch.no_grad():
            age_p, gen_p = model(input_tensor)
            age = int(age_p.item() * 100)
            gender_idx = torch.argmax(gen_p, dim=1).item()
            gender = "Male" if gender_idx == 1 else "Female"

        # 4. Отрисовка результата
        color = (0, 255, 0) if gender == "Male" else (255, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        label = f"{gender}, {age}y"
        cv2.putText(img, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- ИНТЕРФЕЙС ---
st.title("👨‍🦳 Распознавание Возраста и Пола (AI)")

with st.sidebar:
    st.header("О проекте")
    st.info("Дипломная работа: Использование EfficientNet для анализа лиц в реальном времени.")
    st.write(f"**Device:** {device}")
    st.write("**Разработчик:** Нурдаулет")

st.subheader("Видеопоток с камеры")
webrtc_streamer(
    key="face-analyzer",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},  # Отключаем звук для скорости
)

st.divider()
st.write("Если камера не запускается, проверьте разрешения браузера для localhost или ngrok.")