import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import torch
import cv2
import numpy as np
from PIL import Image
from model import MultiTaskFaceNet # Убедись, что этот файл на GitHub
from torchvision import transforms

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="AI Face Analyzer", layout="wide")

# Базовая конфигурация для работы через интернет
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_trained_model():
    # В облаке обычно нет GPU, поэтому форсируем CPU для стабильности
    # Но оставляем проверку на всякий случай
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model = MultiTaskFaceNet(arch='efficientnet_b0').to(device)
    
    # Файл весов должен лежать в корне репозитория на GitHub
    model_path = "best_face_model_m2.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, device

model, device = load_trained_model()

# Инициализация инструментов
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- ОБРАБОТКА ВИДЕО ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Детекция лиц
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Вырезаем и готовим лицо
        roi = img[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_pil = Image.fromarray(roi_rgb)
        
        input_tensor = tf(roi_pil).unsqueeze(0).to(device)
        
        # Нейросеть делает предсказание
        with torch.no_grad():
            age_p, gen_p = model(input_tensor)
            age = int(age_p.item() * 100)
            gender = "Male" if torch.argmax(gen_p, dim=1).item() == 1 else "Female"

        # Рисуем дизайн
        color = (46, 204, 113) if gender == "Male" else (231, 76, 60) # Зеленый/Красный
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        label = f"{gender}, {age}y"
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- ИНТЕРФЕЙС ---
st.title("📸 AI Face Recognition System")
st.write(f"Работает на: **{device}**")

# Самый стабильный способ вызова стримера для облака
webrtc_streamer(
    key="face-detector",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True, # Позволяет не блокировать основной поток
)

with st.sidebar:
    st.header("Информация")
    st.info("Дипломный проект: Распознавание характеристик лица с помощью EfficientNet-B0.")
    st.markdown("---")
    st.write("**Автор:** Нурдаулет")
