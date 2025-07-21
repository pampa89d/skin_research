import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import time
import os
import pickle
from sklearn.metrics import confusion_matrix

# --- Конфигурация страницы ---
st.set_page_config(
    page_title="Классификация образований на коже",
    page_icon="⚕️",
    layout="wide"
)

# --- Инициализация состояния сессии ---
if 'page' not in st.session_state:
    st.session_state.page = "Классификация изображений"
if 'upload_choice' not in st.session_state:
    st.session_state.upload_choice = None

# --- Загрузка модели ---
model_path = os.path.join("app/models", "model_weights_.pth")
@st.cache_resource
def load_model(model_path='/app/models/model_weights_.pth'):
    """Загружает модель и веса."""
    if not os.path.exists(model_path):
        st.error(f"Файл с весами модели не найден по пути: {model_path}")
        st.error("Пожалуйста, убедитесь, что структура ваших папок соответствует инструкции.")
        st.code("""
        - Главная_папка/
          - папка_со_скриптом/
            - app.py
          - models/
            - model_weights_.pth
        """, language='text')
        st.info("Запускать приложение нужно из 'папки_со_скриптом'.")
        return None
    
    model = models.convnext_tiny()
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- Трансформации для изображений ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- Классы ---
class_names = ['Доброкачественное', 'Злокачественное']

# --- Функция предсказания ---
def predict(image_bytes):
    """Выполняет предсказание для одного изображения."""
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(image_tensor)
        # Применяем Softmax для получения вероятностей
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_probability, predicted_idx = torch.max(probabilities, 1)
        # Конвертируем вероятность в процент
        confidence_percentage = predicted_probability.item() * 100

    end_time = time.time()
    
    prediction_time = end_time - start_time
    predicted_class = class_names[predicted_idx.item()]
    
    return image, predicted_class, prediction_time, confidence_percentage

# --- Главная страница: Классификация (ОБНОВЛЕНА) ---
def main_page():
    st.title("Классификация изображений образований на коже")
    st.write("Загрузите изображение одним из способов ниже, и модель определит его класс.")
    st.subheader('Изображение должно предоставляться в макросъемке участка кожи с подозрением на онкологию')

    # --- СЦЕНАРИЙ 1: ВЫБОР СПОСОБА ЗАГРУЗКИ ---
    if st.session_state.upload_choice is None:
        col1, col2 = st.columns(2)
        if col1.button("Загрузить файл(ы)", use_container_width=True):
            st.session_state.upload_choice = 'file'
            st.rerun()
        if col2.button("Загрузить по ссылке", use_container_width=True):
            st.session_state.upload_choice = 'url'
            st.rerun()

    # --- СЦЕНАРИЙ 2: ВЫБРАНА ЗАГРУЗКА ФАЙЛОВ ---
    elif st.session_state.upload_choice == 'file':
        uploaded_files = st.file_uploader(
            "Выберите одно или несколько изображений...", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )
        if st.button("Назад к выбору"):
            st.session_state.upload_choice = None
            st.rerun()
            
        if uploaded_files:
            st.markdown("---")
            for uploaded_file in uploaded_files:
                image_bytes = uploaded_file.getvalue()
                image, predicted_class, prediction_time, confidence_percentage = predict(image_bytes)
                
                st.image(image, caption=f"Загруженное изображение: {uploaded_file.name}", use_container_width=True)
                st.success(f'''**Предсказанный класс:** {predicted_class}. 
                           \nМодель уверена в своем предсказании на {confidence_percentage:.1f}%''')
                st.info(f"**Время ответа модели:** {prediction_time:.4f} секунд")
                st.markdown("---")

    # --- СЦЕНАРИЙ 3: ВЫБРАНА ЗАГРУЗКА ПО ССЫЛКЕ ---
    elif st.session_state.upload_choice == 'url':
        url = st.text_input("Введите URL изображения:")
        
        col1, col2 = st.columns([3, 1]) # Кнопки разной ширины
        col1.button("Классифицировать по ссылке", key="url_button", on_click=lambda: st.session_state.update(process_url=True))
        col2.button("Назад к выбору", on_click=lambda: st.session_state.update(upload_choice=None))

        if st.session_state.get('process_url', False):
            if url:
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    image_bytes = response.content
                    image, predicted_class, prediction_time = predict(image_bytes)
                    
                    st.markdown("---")
                    st.image(image, caption="Загруженное изображение по ссылке", use_container_width=True)
                    st.success(f"**Предсказанный класс:** {predicted_class}")
                    st.info(f"**Время ответа модели:** {prediction_time:.4f} секунд")
                except requests.exceptions.RequestException as e:
                    st.error(f"Ошибка при загрузке изображения по ссылке: {e}")
            else:
                st.warning("Пожалуйста, введите URL изображения.")
            st.session_state.process_url = False # Сброс состояния после обработки

# --- Страница с информацией о модели ---
def info_page():
    st.title("Информация о модели")
    
    history_path = '/models/fanconic_history.pkl'
    predictions_path = '/models/fanconic_predictions.pkl'

    if not os.path.exists(history_path) or not os.path.exists(predictions_path):
        st.error("Файлы с историей обучения (`fanconic_history.pkl`) или предсказаниями (`fanconic_predictions.pkl`) не найдены в папке `models`.")
        return

    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    def to_cpu(data):
        return [item.cpu().item() if torch.is_tensor(item) else item for item in data]

    train_loss = to_cpu(history.get('train_losses', []))
    val_loss = to_cpu(history.get('valid_losses', [])) 
    train_acc = to_cpu(history.get('train_accs', []))
    val_acc = to_cpu(history.get('valid_accs', []))
    train_f1 = to_cpu(history.get('train_f1_scores', []))
    val_f1 = to_cpu(history.get('valid_f1_scores', []))
    train_time_list = history.get('training_time', [])
    train_time = train_time_list[-1] if train_time_list else 'N/A'
    epochs = range(1, len(train_loss) + 1)

    st.header("Процесс обучения модели")
    st.write(f"**Количество эпох:** {len(epochs)}")
    st.write(f"**Общее время обучения:** {train_time}")
    
    st.subheader("Кривые обучения и метрик")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))
    
    ax1.plot(epochs, train_loss, 'o-', color='g', label='Training loss')
    ax1.plot(epochs, val_loss, 'o-', color='b', label='Validation loss')
    ax1.set_title('Потери (Loss)')
    ax1.set_xlabel('Эпохи')
    ax1.set_ylabel('Потери')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_acc, 'o-', color='g', label='Training accuracy')
    ax2.plot(epochs, val_acc, 'o-', color='b', label='Validation accuracy')
    ax2.set_title('Точность (Accuracy)')
    ax2.set_xlabel('Эпохи')
    ax2.set_ylabel('Точность')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(epochs, train_f1, 'o-', color='g', label='Training F1-score')
    ax3.plot(epochs, val_f1, 'o-', color='b', label='Validation F1-score')
    ax3.set_title('F1-score')
    ax3.set_xlabel('Эпохи')
    ax3.set_ylabel('F1-score')
    ax3.legend()
    ax3.grid(True)
    
    st.pyplot(fig)

    st.header("Состав датасета")
    st.write("Датасет взят с Kaggle: [Skin Cancer: Malignant vs. Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)")
    st.write("- **Обучающая выборка:** 2637 изображений")
    st.write("- **Тестовая (валидационная) выборка:** 660 изображений")
    st.write("- **Распределение по классам:**")
    st.write(f"  ***{class_names[0]}:** 1440 (обучение), 360 (тест)")
    st.write(f"  ***{class_names[1]}:** 1197 (обучение), 300 (тест)")
    
    st.header("Финальные метрики на тестовой выборке")

    with open(predictions_path, 'rb') as f:
        pred = pickle.load(f)

    y_true = pred['y_true']
    y_pred = pred['y_pred']

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="**Validation F1-score**", value=f"{val_f1[-1]:.4f}")
    with col2:
        st.metric(label="**Validation Accuracy**", value=f"{val_acc[-1]:.4f}")

    st.subheader("Матрица ошибок (Confusion Matrix)")
    cm = confusion_matrix(y_true, y_pred)
    
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    ax_cm.set_xlabel('Предсказанные классы')
    ax_cm.set_ylabel('Истинные классы')
    ax_cm.set_title('Матрица ошибок для валидационной выборки')
    st.pyplot(fig_cm)

# --- УПРАВЛЕНИЕ ИНТЕРФЕЙСОМ ---

# --- Боковая панель (Сайдбар) ---
st.sidebar.title("Навигация")

if st.sidebar.button("Классификация изображений", use_container_width=True, key='nav_main'):
    st.session_state.page = "Классификация изображений"
    st.session_state.upload_choice = None # Сбрасываем выбор при переключении на главную
    st.rerun()

if st.sidebar.button("Информация о модели", use_container_width=True, key='nav_info'):
    st.session_state.page = "Информация о модели"
    st.rerun()

# --- Основное содержимое страницы ---
if model is not None:
    if st.session_state.page == "Классификация изображений":
        main_page()
    elif st.session_state.page == "Информация о модели":
        info_page()
        