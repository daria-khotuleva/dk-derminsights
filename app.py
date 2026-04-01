"""
Streamlit-приложение для классификации кожных новообразований.
Современный дизайн: glassmorphism, градиенты, анимации.
Запуск: streamlit run app.py
"""

import json
import base64
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
IMG_SIZE = 224
NUM_CLASSES = 8

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "scc", "vasc"]
MEL_IDX = CLASS_NAMES.index("mel")
BCC_IDX = CLASS_NAMES.index("bcc")
AKIEC_IDX = CLASS_NAMES.index("akiec")
SCC_IDX = CLASS_NAMES.index("scc")

CLASS_INFO = {
    "akiec": {
        "name": "Актинический кератоз",
        "risk": "Предраковое",
        "color": "#FFA500",
        "icon": "🟠",
        "gradient": "linear-gradient(135deg, #f6d365 0%, #fda085 100%)",
        "desc": "Предраковое состояние, вызванное длительным воздействием УФ-лучей. Может перерасти в плоскоклеточную карциному.",
    },
    "bcc": {
        "name": "Базальноклеточная карцинома",
        "risk": "Злокачественное",
        "color": "#FF4444",
        "icon": "🔴",
        "gradient": "linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)",
        "desc": "Самый распространённый вид рака кожи. Растёт медленно, редко метастазирует, но требует лечения.",
    },
    "bkl": {
        "name": "Доброкачественный кератоз",
        "risk": "Доброкачественное",
        "color": "#44AA44",
        "icon": "🟢",
        "gradient": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
        "desc": "Доброкачественное поражение (себорейный кератоз, солнечное лентиго). Как правило, не опасно.",
    },
    "df": {
        "name": "Дерматофиброма",
        "risk": "Доброкачественное",
        "color": "#44AA44",
        "icon": "🟢",
        "gradient": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
        "desc": "Доброкачественная опухоль кожи. Обычно не требует лечения, если не вызывает дискомфорт.",
    },
    "mel": {
        "name": "Меланома",
        "risk": "Злокачественное (опасно!)",
        "color": "#FF0000",
        "icon": "🔴",
        "gradient": "linear-gradient(135deg, #eb3349 0%, #f45c43 100%)",
        "desc": "Наиболее опасный вид рака кожи. Может быстро метастазировать. Раннее обнаружение критически важно!",
    },
    "nv": {
        "name": "Невус (родинка)",
        "risk": "Доброкачественное",
        "color": "#44AA44",
        "icon": "🟢",
        "gradient": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
        "desc": "Обычная доброкачественная родинка. Большинство невусов безопасны, но за ними стоит наблюдать.",
    },
    "scc": {
        "name": "Плоскоклеточная карцинома",
        "risk": "Злокачественное",
        "color": "#FF4444",
        "icon": "🔴",
        "gradient": "linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)",
        "desc": "Второй по распространённости рак кожи. Может метастазировать. Связан с хроническим солнечным воздействием.",
    },
    "vasc": {
        "name": "Сосудистое поражение",
        "risk": "Доброкачественное",
        "color": "#44AA44",
        "icon": "🟢",
        "gradient": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
        "desc": "Ангиомы, ангиокератомы и подобные сосудистые образования. Как правило, не опасны.",
    },
}

BCC_LOCATIONS = ["Лицо", "Нос", "Уши", "Шея", "Голова/скальп"]
MEL_LOCATIONS = ["Спина", "Ноги", "Руки", "Туловище"]
AKIEC_LOCATIONS = ["Лицо", "Голова/скальп", "Руки", "Уши", "Нос"]
ALL_LOCATIONS = [
    "Лицо", "Нос", "Уши", "Голова/скальп", "Шея",
    "Плечи", "Руки", "Кисти рук",
    "Грудь", "Живот", "Спина", "Туловище",
    "Ноги", "Стопы", "Другое",
]


# ═══════════════════════════════════════════
# CSS: Современный дизайн с glassmorphism
# ═══════════════════════════════════════════
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Глобальные стили ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1a3e 40%, #0d1b2a 100%);
}

/* ── Заголовок ── */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem;
    animation: fadeInDown 0.8s ease-out;
}
.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #48c6ef, #6f86d6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.main-header p {
    color: #8892b0;
    font-size: 1.05rem;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Glassmorphism карточки ── */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.glass-card:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(108, 99, 255, 0.3);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
}

/* ── Результат диагностики ── */
.diagnosis-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 2rem;
    text-align: center;
    animation: fadeInUp 0.6s ease-out;
}
.diagnosis-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.diagnosis-confidence {
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.diagnosis-desc {
    color: #8892b0;
    font-size: 0.95rem;
    line-height: 1.6;
    margin-top: 1rem;
}

/* ── Риск-бейдж ── */
.risk-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.risk-danger {
    background: rgba(255, 59, 48, 0.15);
    color: #FF3B30;
    border: 1px solid rgba(255, 59, 48, 0.3);
}
.risk-warning {
    background: rgba(255, 165, 0, 0.15);
    color: #FFA500;
    border: 1px solid rgba(255, 165, 0, 0.3);
}
.risk-safe {
    background: rgba(52, 199, 89, 0.15);
    color: #34C759;
    border: 1px solid rgba(52, 199, 89, 0.3);
}

/* ── Прогресс-бары вероятностей ── */
.prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.6rem;
    gap: 0.8rem;
}
.prob-label {
    min-width: 180px;
    font-size: 0.85rem;
    color: #ccd6f6;
}
.prob-bar-bg {
    flex: 1;
    height: 10px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 1s ease-out;
}
.prob-value {
    min-width: 50px;
    text-align: right;
    font-size: 0.85rem;
    font-weight: 600;
    color: #ccd6f6;
}

/* ── Профиль риска ── */
.risk-meter {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.7rem 1rem;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    margin-bottom: 0.5rem;
    transition: all 0.3s ease;
}
.risk-meter:hover {
    background: rgba(255, 255, 255, 0.06);
}
.risk-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.risk-name {
    flex: 1;
    font-size: 0.85rem;
    color: #ccd6f6;
}
.risk-level {
    font-size: 0.8rem;
    font-weight: 600;
}

/* ── Предупреждение (alert) ── */
.alert-danger {
    background: rgba(255, 59, 48, 0.08);
    border: 1px solid rgba(255, 59, 48, 0.2);
    border-left: 4px solid #FF3B30;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    animation: pulse-border 2s ease-in-out infinite;
}
.alert-warning {
    background: rgba(255, 165, 0, 0.08);
    border: 1px solid rgba(255, 165, 0, 0.2);
    border-left: 4px solid #FFA500;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.alert-success {
    background: rgba(52, 199, 89, 0.08);
    border: 1px solid rgba(52, 199, 89, 0.2);
    border-left: 4px solid #34C759;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.alert-info {
    background: rgba(108, 99, 255, 0.08);
    border: 1px solid rgba(108, 99, 255, 0.2);
    border-left: 4px solid #6C63FF;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12152a 0%, #0d1b2a 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label {
    color: #8892b0 !important;
    font-size: 0.9rem;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(108, 99, 255, 0.3) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(108, 99, 255, 0.6) !important;
    background: rgba(108, 99, 255, 0.05) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
}

/* ── Фототип кожи ── */
.skin-types {
    display: flex;
    justify-content: space-between;
    gap: 6px;
    margin: 8px 0 12px;
}
.skin-circle {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    border: 2px solid rgba(255,255,255,0.15);
    transition: all 0.3s ease;
    cursor: pointer;
}
.skin-circle:hover {
    transform: scale(1.15);
    border-color: #6C63FF;
    box-shadow: 0 0 12px rgba(108, 99, 255, 0.4);
}
.skin-label {
    font-size: 10px;
    color: #8892b0;
    text-align: center;
    margin-top: 4px;
}

/* ── Анимации ── */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-border {
    0%, 100% { border-left-color: #FF3B30; }
    50% { border-left-color: rgba(255, 59, 48, 0.4); }
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Disclaimer badge ── */
.disclaimer {
    background: rgba(255, 165, 0, 0.06);
    border: 1px solid rgba(255, 165, 0, 0.15);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    text-align: center;
    color: #FFA500;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
}

/* ── Скрыть стандартные элементы Streamlit ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 10px;
    padding: 8px 20px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    color: #8892b0;
}
.stTabs [aria-selected="true"] {
    background: rgba(108, 99, 255, 0.15) !important;
    border-color: rgba(108, 99, 255, 0.3) !important;
    color: #6C63FF !important;
}
</style>
"""


# ═══════════════════════════════════════════
# Multimodal model definition (must match training)
# ═══════════════════════════════════════════
SITE_CATEGORIES = [
    "anterior torso", "posterior torso", "lateral torso",
    "upper extremity", "lower extremity",
    "head/neck", "palms/soles", "oral/genital", "unknown",
]
NUM_SITES = len(SITE_CATEGORIES)
META_DIM = 1 + 2 + NUM_SITES  # age + sex(2) + site(9) = 12

# Маппинг локализаций из анкеты → ISIC site categories
LOCATION_TO_SITE = {
    "Лицо": "head/neck", "Нос": "head/neck", "Уши": "head/neck",
    "Голова/скальп": "head/neck", "Шея": "head/neck",
    "Плечи": "upper extremity", "Руки": "upper extremity", "Кисти рук": "upper extremity",
    "Грудь": "anterior torso", "Живот": "anterior torso",
    "Спина": "posterior torso", "Туловище": "anterior torso",
    "Ноги": "lower extremity", "Стопы": "palms/soles",
    "Другое": "unknown",
}


class MultimodalDermModel(nn.Module):
    def __init__(self, num_classes, meta_dim):
        super().__init__()
        efficientnet = models.efficientnet_b0(weights=None)
        self.image_features = efficientnet.features
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        img_feat_dim = 1280

        self.meta_branch = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        combined_dim = img_feat_dim + 32
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, meta):
        x_img = self.image_features(image)
        x_img = self.image_pool(x_img).flatten(1)
        x_meta = self.meta_branch(meta)
        x = torch.cat([x_img, x_meta], dim=1)
        return self.classifier(x)


def encode_metadata(age, sex, location):
    """Кодирует данные из анкеты в вектор для модели."""
    age_norm = float(age) / 85.0
    sex_vec = [1.0, 0.0] if sex == "Мужской" else [0.0, 1.0]
    site = [0.0] * NUM_SITES
    site_name = LOCATION_TO_SITE.get(location, "unknown")
    site_idx = SITE_CATEGORIES.index(site_name) if site_name in SITE_CATEGORIES else SITE_CATEGORIES.index("unknown")
    site[site_idx] = 1.0
    return torch.tensor([age_norm] + sex_vec + site, dtype=torch.float32).unsqueeze(0)


@st.cache_resource
def load_author_photo():
    photo_path = Path(__file__).parent / "author.jpg"
    if photo_path.exists():
        return base64.b64encode(photo_path.read_bytes()).decode()
    return None


@st.cache_resource
def load_model():
    model = MultimodalDermModel(NUM_CLASSES, META_DIM)
    model_path = MODEL_DIR / "best_model.pth"
    if not model_path.exists():
        st.error("Модель не найдена! Запустите: python3 02_train.py")
        st.stop()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_mel_threshold():
    info_path = MODEL_DIR / "model_info.json"
    if info_path.exists():
        with open(info_path) as f:
            return json.load(f).get("mel_threshold", 0.1)
    return 0.1


def compute_risk_factors(age, sex, location, sunburn_history,
                         tanning_bed, sun_exposure, family_history,
                         mole_count, skin_type, lesion_changed):
    mel_risk = 1.0
    bcc_risk = 1.0
    akiec_risk = 1.0

    if age >= 60:
        mel_risk *= 1.4; bcc_risk *= 1.5; akiec_risk *= 1.5
    elif age >= 50:
        mel_risk *= 1.2; bcc_risk *= 1.3; akiec_risk *= 1.3
    elif age < 30:
        mel_risk *= 0.8; bcc_risk *= 0.7

    if sex == "Мужской":
        mel_risk *= 1.1; bcc_risk *= 1.15

    if location in BCC_LOCATIONS:
        bcc_risk *= 1.4
    if location in AKIEC_LOCATIONS:
        akiec_risk *= 1.3
    if location in MEL_LOCATIONS:
        mel_risk *= 1.15

    if sunburn_history == "Часто (5+ сильных ожогов)":
        mel_risk *= 1.5; bcc_risk *= 1.4; akiec_risk *= 1.4
    elif sunburn_history == "Иногда (2-4 ожога)":
        mel_risk *= 1.25; bcc_risk *= 1.2; akiec_risk *= 1.2

    if tanning_bed == "Да, регулярно":
        mel_risk *= 1.6; bcc_risk *= 1.3
    elif tanning_bed == "Да, иногда":
        mel_risk *= 1.3; bcc_risk *= 1.15

    if sun_exposure == "Много (работа на улице / загораю каждый день)":
        bcc_risk *= 1.4; akiec_risk *= 1.5; mel_risk *= 1.2
    elif sun_exposure == "Умеренно":
        bcc_risk *= 1.1; akiec_risk *= 1.15

    if family_history == "Да, меланома у близких родственников":
        mel_risk *= 1.8
    elif family_history == "Да, другой рак кожи":
        mel_risk *= 1.3; bcc_risk *= 1.3

    if mole_count == "Много (50+)":
        mel_risk *= 1.5
    elif mole_count == "Среднее (20-50)":
        mel_risk *= 1.2

    if skin_type.startswith("I —"):
        mel_risk *= 1.5; bcc_risk *= 1.4
    elif skin_type.startswith("II —"):
        mel_risk *= 1.3; bcc_risk *= 1.2
    elif skin_type.startswith("III —"):
        mel_risk *= 1.1

    if lesion_changed == "Да, быстро растёт / меняет цвет / форму":
        mel_risk *= 1.6
    elif lesion_changed == "Да, немного изменилось":
        mel_risk *= 1.2

    return {MEL_IDX: mel_risk, BCC_IDX: bcc_risk, AKIEC_IDX: akiec_risk}


def predict(model, image, mel_threshold, risk_factors, age, sex, location):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image).unsqueeze(0)
    meta_tensor = encode_metadata(age, sex, location)

    with torch.no_grad():
        output = model(img_tensor, meta_tensor)
        probs = torch.softmax(output, dim=1)[0].numpy()

    # Дополнительная коррекция по факторам анкеты (солярий, генетика и т.д.)
    adjusted_probs = probs.copy()
    for class_idx, multiplier in risk_factors.items():
        adjusted_probs[class_idx] *= multiplier
    adjusted_probs = adjusted_probs / adjusted_probs.sum()

    mel_risk = risk_factors.get(MEL_IDX, 1.0)
    effective_threshold = mel_threshold / max(mel_risk, 1.0)

    top_idx = int(np.argmax(adjusted_probs))
    if adjusted_probs[MEL_IDX] >= effective_threshold and top_idx != MEL_IDX:
        top_idx = MEL_IDX

    return probs, adjusted_probs, top_idx, effective_threshold


def generate_gradcam(model, image, target_class, age, sex, location):
    """
    Grad-CAM: показывает, на какие области изображения смотрит модель.
    Возвращает PIL Image с наложенной тепловой картой.
    """
    import matplotlib.cm as cm

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image).unsqueeze(0)
    meta_tensor = encode_metadata(age, sex, location)

    # Hook для захвата активаций и градиентов последнего conv-слоя
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    # Регистрируем hooks на последний блок features
    target_layer = model.image_features[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    model.eval()
    img_tensor.requires_grad_(True)
    output = model(img_tensor, meta_tensor)

    # Backward pass для целевого класса
    model.zero_grad()
    output[0, target_class].backward()

    # Grad-CAM
    grads = gradients["value"]       # (1, C, H, W)
    acts = activations["value"]      # (1, C, H, W)
    weights = grads.mean(dim=[2, 3], keepdim=True)  # Global Average Pooling градиентов
    cam = (weights * acts).sum(dim=1, keepdim=True)  # Взвешенная сумма
    cam = torch.relu(cam)            # ReLU — только положительное влияние
    cam = cam.squeeze().numpy()

    # Нормализация 0-1
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    # Resize до размера изображения
    from PIL import Image as PILImage
    cam_resized = np.array(PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), PILImage.BILINEAR
    )) / 255.0

    # Наложение heatmap на изображение
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0

    # Colormap: jet
    heatmap = cm.jet(cam_resized)[:, :, :3]

    # Blend: 60% изображение + 40% heatmap
    overlay = (img_array * 0.6 + heatmap * 0.4)
    overlay = np.clip(overlay, 0, 1)
    overlay_img = PILImage.fromarray((overlay * 255).astype(np.uint8))

    # Удаляем hooks
    fh.remove()
    bh.remove()

    return overlay_img


def format_risk_level(risk):
    if risk >= 2.5:
        return "Очень высокий", "#FF3B30"
    elif risk >= 1.8:
        return "Высокий", "#FF6B6B"
    elif risk >= 1.3:
        return "Повышенный", "#FFA500"
    elif risk >= 1.0:
        return "Средний", "#8892b0"
    else:
        return "Ниже среднего", "#34C759"


def get_risk_badge_class(top_class):
    if top_class in ("mel", "bcc", "scc"):
        return "risk-danger"
    elif top_class == "akiec":
        return "risk-warning"
    else:
        return "risk-safe"


def render_prob_bars(adj_probs, raw_probs):
    """Рендерит красивые прогресс-бары для каждого класса."""
    sorted_indices = np.argsort(adj_probs)[::-1]
    html = ""
    colors = {
        "mel": "#FF3B30", "bcc": "#FF6B6B", "scc": "#FF6B6B", "akiec": "#FFA500",
        "bkl": "#34C759", "nv": "#34C759", "df": "#34C759", "vasc": "#48c6ef",
    }
    for idx in sorted_indices:
        cls = CLASS_NAMES[idx]
        prob = adj_probs[idx]
        raw = raw_probs[idx]
        color = colors[cls]
        name = CLASS_INFO[cls]["name"]
        delta = prob - raw
        delta_str = ""
        if abs(delta) > 0.005:
            arrow = "+" if delta > 0 else ""
            delta_str = f" <span style='color:#8892b0;font-size:0.75rem'>({arrow}{delta*100:.1f}%)</span>"
        width = max(prob * 100, 1)
        html += f"""
        <div class="prob-row">
            <span class="prob-label">{name}</span>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{width}%;background:{color}"></div>
            </div>
            <span class="prob-value">{prob*100:.1f}%{delta_str}</span>
        </div>"""
    return html


# ═══════════════════════════════════════════
# Main app
# ═══════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="DK DermInsights",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    model = load_model()
    base_mel_threshold = load_mel_threshold()

    # ═══ Sidebar: клиническая анкета ═══
    with st.sidebar:
        st.markdown("### 📋 Клиническая анкета")
        st.caption("Заполните для точной оценки рисков")

        st.markdown("---")
        st.markdown("**Основные данные**")
        sex = st.radio("Пол", ["Мужской", "Женский"], horizontal=True)
        age_range = st.radio(
            "Возраст",
            ["До 18 лет", "18–30 лет", "30–40 лет", "40–50 лет",
             "50–60 лет", "60–70 лет", "70+ лет"],
        )
        age_map = {
            "До 18 лет": 15, "18–30 лет": 25, "30–40 лет": 35,
            "40–50 лет": 45, "50–60 лет": 55, "60–70 лет": 65, "70+ лет": 75,
        }
        age = age_map[age_range]
        location = st.selectbox("Локализация образования", ALL_LOCATIONS)

        st.markdown("---")
        st.markdown("**Солнце и УФ-излучение**")
        sunburn_history = st.select_slider(
            "Солнечные ожоги в прошлом",
            options=["Редко или никогда", "Иногда (2-4 ожога)", "Часто (5+ сильных ожогов)"],
        )
        tanning_bed = st.radio("Солярий", ["Нет", "Да, иногда", "Да, регулярно"])
        sun_exposure = st.radio(
            "Пребывание на солнце",
            ["Мало (в основном в помещении)", "Умеренно",
             "Много (работа на улице / загораю каждый день)"],
        )

        st.markdown("---")
        st.markdown("**Генетика и кожа**")
        family_history = st.radio(
            "Рак кожи у родственников",
            ["Нет / не знаю", "Да, другой рак кожи", "Да, меланома у близких родственников"],
        )
        mole_count = st.radio(
            "Количество родинок",
            ["Мало (менее 20)", "Среднее (20-50)", "Много (50+)"],
        )

        st.markdown("**Фототип кожи**")
        st.markdown(
            '<div class="skin-types">'
            '<div><div class="skin-circle" style="background:#FDEBD0"></div><div class="skin-label">I</div></div>'
            '<div><div class="skin-circle" style="background:#F5CBA7"></div><div class="skin-label">II</div></div>'
            '<div><div class="skin-circle" style="background:#E0B07B"></div><div class="skin-label">III</div></div>'
            '<div><div class="skin-circle" style="background:#C68E4E"></div><div class="skin-label">IV</div></div>'
            '<div><div class="skin-circle" style="background:#8D5524"></div><div class="skin-label">V</div></div>'
            '<div><div class="skin-circle" style="background:#4A2912"></div><div class="skin-label">VI</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        skin_type = st.radio(
            "Ваш фототип",
            [
                "I — очень светлая, всегда обгорает",
                "II — светлая, легко обгорает",
                "III — средняя, иногда обгорает",
                "IV — смуглая, редко обгорает",
                "V-VI — тёмная, почти не обгорает",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Образование**")
        lesion_changed = st.radio(
            "Изменилось ли образование?",
            ["Нет, стабильное", "Да, немного изменилось",
             "Да, быстро растёт / меняет цвет / форму"],
        )

    risk_factors = compute_risk_factors(
        age, sex, location, sunburn_history,
        tanning_bed, sun_exposure, family_history,
        mole_count, skin_type, lesion_changed,
    )

    # ═══ Header ═══
    st.markdown(
        '<div class="main-header">'
        '<h1>DK DermInsights</h1>'
        '<p>Интеллектуальная система анализа кожных новообразований.<br>'
        'Загрузите фото и заполните анкету — модель оценит изображение '
        'с учётом ваших индивидуальных факторов риска.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="disclaimer">'
        '⚠️ Это учебный проект, не медицинский инструмент. '
        'Для постановки диагноза обратитесь к дерматологу.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ═══ Main content ═══
    col1, spacer, col2 = st.columns([5, 0.5, 5])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📷 Загрузите дерматоскопическое изображение")
        uploaded_file = st.file_uploader(
            "Перетащите файл или нажмите для выбора",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Профиль риска
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Ваш профиль риска")
        risk_labels = {MEL_IDX: "Меланома", BCC_IDX: "Баз. карцинома", AKIEC_IDX: "Акт. кератоз"}
        risk_html = ""
        for idx, name in risk_labels.items():
            risk = risk_factors[idx]
            level, color = format_risk_level(risk)
            risk_html += f"""
            <div class="risk-meter">
                <div class="risk-dot" style="background:{color}"></div>
                <span class="risk-name">{name}</span>
                <span class="risk-level" style="color:{color}">{level} (x{risk:.1f})</span>
            </div>"""
        st.markdown(risk_html, unsafe_allow_html=True)

        warnings = []
        if risk_factors[MEL_IDX] >= 1.5:
            warnings.append("Повышенный риск меланомы по анамнезу")
        if risk_factors[BCC_IDX] >= 1.5:
            warnings.append("Повышенный риск базальноклеточной карциномы")
        if lesion_changed == "Да, быстро растёт / меняет цвет / форму":
            warnings.append("Быстрое изменение образования — тревожный признак")
        if family_history == "Да, меланома у близких родственников":
            warnings.append("Семейная история меланомы — значимый фактор")

        if warnings:
            warn_html = '<div class="alert-warning"><strong>На что обратить внимание:</strong><ul style="margin:0.5rem 0 0;padding-left:1.2rem">'
            for w in warnings:
                warn_html += f"<li style='margin-bottom:0.3rem;color:#ccd6f6;font-size:0.9rem'>{w}</li>"
            warn_html += "</ul></div>"
            st.markdown(warn_html, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if uploaded_file is not None:
            raw_probs, adj_probs, top_idx, eff_threshold = predict(
                model, image, base_mel_threshold, risk_factors, age, sex, location
            )

            # Всегда показываем класс с наибольшей вероятностью
            argmax_idx = int(np.argmax(adj_probs))
            top_class = CLASS_NAMES[argmax_idx]
            info = CLASS_INFO[top_class]
            badge_class = get_risk_badge_class(top_class)

            # Проверяем порог меланомы для предупреждения
            mel_prob = adj_probs[MEL_IDX]
            mel_threshold_triggered = mel_prob >= eff_threshold and argmax_idx != MEL_IDX

            # Diagnosis card — всегда реальный топ-класс
            st.markdown(
                f'<div class="diagnosis-card">'
                f'<div class="diagnosis-title">{info["icon"]} {info["name"]}</div>'
                f'<div class="diagnosis-confidence" style="background:{info["gradient"]};'
                f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
                f'background-size:200% 200%;animation:gradientShift 3s ease infinite">'
                f'{adj_probs[argmax_idx]*100:.1f}%</div>'
                f'<span class="risk-badge {badge_class}">{info["risk"]}</span>'
                f'<div class="diagnosis-desc">{info["desc"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Предупреждение о меланоме — отдельный блок
            if mel_threshold_triggered:
                st.markdown(
                    f'<div class="alert-danger" style="margin-top:1rem">'
                    f'<strong>⚠️ Риск меланомы: {mel_prob*100:.1f}%</strong><br>'
                    f'Хотя наиболее вероятный диагноз — {info["name"]}, '
                    f'вероятность меланомы повышена с учётом ваших факторов риска. '
                    f'<strong>Рекомендуется обследование у дерматолога</strong> для исключения меланомы.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Grad-CAM визуализация
            st.markdown('<div class="glass-card" style="margin-top:1rem">', unsafe_allow_html=True)
            st.markdown("#### 🔍 Grad-CAM: на что смотрит модель")
            st.markdown(
                '<div style="color:#8892b0;font-size:0.85rem;margin-bottom:0.8rem">'
                'Тепловая карта показывает области, которые повлияли на решение модели. '
                '<span style="color:#FF3B30">Красные зоны</span> — максимальное внимание, '
                '<span style="color:#3B82F6">синие</span> — минимальное.</div>',
                unsafe_allow_html=True,
            )
            try:
                gradcam_img = generate_gradcam(model, image, argmax_idx, age, sex, location)
                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    st.image(image.resize((IMG_SIZE, IMG_SIZE)), caption="Оригинал", use_container_width=True)
                with gcol2:
                    st.image(gradcam_img, caption=f"Grad-CAM: {info['name']}", use_container_width=True)
            except Exception as e:
                st.markdown(f'<div style="color:#8892b0">Grad-CAM недоступен: {e}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div class="glass-card" style="margin-top:1rem">', unsafe_allow_html=True)
            st.markdown("#### Вероятности всех классов")
            st.markdown(render_prob_bars(adj_probs, raw_probs), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Recommendation
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 💡 Рекомендация")
            if top_class == "mel":
                st.markdown(
                    '<div class="alert-danger">'
                    '🔴 <strong>Обратитесь к дерматологу как можно скорее.</strong><br>'
                    'Меланома — серьёзное заболевание. Раннее обнаружение спасает жизни.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            elif top_class in ("bcc", "akiec", "scc"):
                st.markdown(
                    '<div class="alert-warning">'
                    '🟠 <strong>Рекомендуется визит к дерматологу.</strong><br>'
                    f'{info["name"]} хорошо поддаётся лечению при раннем обнаружении.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            elif risk_factors[MEL_IDX] >= 1.5 or risk_factors[BCC_IDX] >= 1.5:
                st.markdown(
                    '<div class="alert-info">'
                    '🟡 Образование выглядит доброкачественным, но ваш профиль риска повышен. '
                    'Рекомендуется регулярный осмотр у дерматолога.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="alert-success">'
                    '🟢 Образование, вероятнее всего, доброкачественное. '
                    'Тем не менее, при любых изменениях обратитесь к врачу.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown(
                '<div class="glass-card" style="text-align:center;padding:4rem 2rem">'
                '<div style="font-size:4rem;margin-bottom:1rem">🔬</div>'
                '<div style="font-size:1.2rem;color:#8892b0;margin-bottom:0.5rem">'
                'Загрузите изображение для анализа</div>'
                '<div style="font-size:0.9rem;color:#5a6178">'
                'Поддерживаемые форматы: JPG, PNG<br>'
                'Рекомендуется: дерматоскопические снимки</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ═══ Нижняя секция: информация ═══
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["👩‍⚕️ Об авторе и проекте", "📚 О заболеваниях", "🧠 О модели", "📋 Как работает анкета", "🛠 Технологии"])

    with tab1:
        author_b64 = load_author_photo()
        author_img_tag = (
            f'<img src="data:image/jpeg;base64,{author_b64}" alt="Дарья Хотулева" '
            if author_b64 else '<img src="" alt="Дарья Хотулева" '
        )
        st.markdown(
            '<div class="glass-card">'
            '<div style="display:flex;gap:2rem;align-items:flex-start;flex-wrap:wrap">'

            # Left: photo + name
            '<div style="flex:0 0 auto;text-align:center">'
            + author_img_tag +
            'style="width:140px;height:140px;border-radius:50%;object-fit:cover;'
            'border:3px solid rgba(108,99,255,0.4);margin:0 auto 1rem;display:block;'
            'box-shadow:0 0 20px rgba(108,99,255,0.2)">'
            '<div style="font-size:1.2rem;font-weight:700;color:#ccd6f6">Дарья Хотулева</div>'
            '<div style="color:#6C63FF;font-size:0.85rem;font-weight:500;margin-top:0.3rem">'
            'Врач-дерматолог &nbsp;→&nbsp; Data Scientist</div>'
            '</div>'

            # Right: story
            '<div style="flex:1;min-width:280px">'
            '<div style="font-size:1.3rem;font-weight:700;color:#ccd6f6;margin-bottom:1rem">'
            'От дерматоскопа к нейросетям</div>'
            '<div style="color:#8892b0;line-height:1.8;font-size:0.95rem">'

            'Я окончила <strong style="color:#ccd6f6">Дальневосточный государственный '
            'медицинский университет</strong> (ДВГМУ, Хабаровск), '
            'а ординатуру по дерматовенерологии прошла в Москве в '
            '<strong style="color:#ccd6f6">ФГБУ ГНЦДК</strong> — одном из ведущих '
            'дерматологических центров страны. Там же начала клиническую практику: '
            'дерматоскопия, диагностика новообразований, работа с пациентами.<br><br>'

            'После переезда из России я решила сменить вектор карьеры, '
            'но сохранить то, что мне всегда было важно — '
            '<strong style="color:#ccd6f6">помогать людям</strong>. '
            'Так я пришла в Data Science: сейчас учусь в Португалии и параллельно '
            'осваиваю машинное обучение самостоятельно.<br><br>'

            'Этот проект — точка пересечения моих двух профессий. '
            'Я знаю, как выглядит меланома под дерматоскопом, знаю, что '
            'dysplastic nevus и меланому путают даже опытные врачи, и знаю, '
            'какие клинические факторы повышают риск. '
            'Именно поэтому модель здесь — не просто классификатор картинок: '
            'она учитывает <strong style="color:#ccd6f6">анамнез пациента</strong>, '
            'как это делает дерматолог на приёме.'
            '</div>'
            '</div>'
            '</div>'

            # Bottom: highlights
            '<div style="display:flex;gap:1rem;margin-top:1.5rem;flex-wrap:wrap;width:100%">'
            '<div style="flex:1;min-width:160px;background:rgba(108,99,255,0.08);'
            'border-radius:12px;padding:1rem;text-align:center">'
            '<div style="font-size:1.5rem;font-weight:700;color:#6C63FF">ДВГМУ</div>'
            '<div style="color:#8892b0;font-size:0.8rem;margin-top:0.3rem">Медицинское образование<br>Хабаровск</div>'
            '</div>'
            '<div style="flex:1;min-width:160px;background:rgba(108,99,255,0.08);'
            'border-radius:12px;padding:1rem;text-align:center">'
            '<div style="font-size:1.5rem;font-weight:700;color:#6C63FF">ГНЦДК</div>'
            '<div style="color:#8892b0;font-size:0.8rem;margin-top:0.3rem">Ординатура и практика<br>Москва</div>'
            '</div>'
            '<div style="flex:1;min-width:160px;background:rgba(108,99,255,0.08);'
            'border-radius:12px;padding:1rem;text-align:center">'
            '<div style="font-size:1.5rem;font-weight:700;color:#6C63FF">DS</div>'
            '<div style="color:#8892b0;font-size:0.8rem;margin-top:0.3rem">Data Science<br>Португалия</div>'
            '</div>'
            '<div style="flex:1;min-width:160px;background:rgba(108,99,255,0.08);'
            'border-radius:12px;padding:1rem;text-align:center">'
            '<div style="font-size:1.5rem;font-weight:700;color:#6C63FF">AI + Med</div>'
            '<div style="color:#8892b0;font-size:0.8rem;margin-top:0.3rem">Технологии<br>для здоровья</div>'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with tab2:
        cols = st.columns(3)
        for i, (cls, ci) in enumerate(CLASS_INFO.items()):
            with cols[i % 3]:
                badge = get_risk_badge_class(cls)
                st.markdown(
                    f'<div class="glass-card" style="min-height:180px">'
                    f'<div style="font-size:0.95rem;font-weight:600;margin-bottom:0.5rem">'
                    f'{ci["icon"]} {ci["name"]}</div>'
                    f'<span class="risk-badge {badge}" style="font-size:0.7rem">{ci["risk"]}</span>'
                    f'<div style="color:#8892b0;font-size:0.85rem;margin-top:0.8rem;line-height:1.5">'
                    f'{ci["desc"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with tab3:
        st.markdown(
            '<div class="glass-card">'

            '<div style="color:#6C63FF;font-size:1.1rem;font-weight:700;margin-bottom:1rem">'
            'Мультимодальная архитектура</div>'
            '<div style="color:#8892b0;line-height:1.7;margin-bottom:1.5rem">'
            'Модель работает как настоящий дерматолог — анализирует <strong style="color:#ccd6f6">'
            'и изображение, и данные пациента</strong> одновременно. '
            'Две ветки нейросети обрабатывают разные типы данных, '
            'затем объединяются для финального решения.</div>'

            '<div style="background:rgba(108,99,255,0.06);border-radius:12px;padding:1.2rem;'
            'margin-bottom:1.5rem;font-family:monospace;font-size:0.85rem;color:#ccd6f6;line-height:1.8">'
            '┌─────────────────┐ &nbsp;&nbsp;&nbsp; ┌──────────────────┐<br>'
            '│ &nbsp; Изображение &nbsp; │ &nbsp;&nbsp;&nbsp; │ &nbsp; Метаданные &nbsp;&nbsp;&nbsp; │<br>'
            '│ &nbsp; (224 x 224) &nbsp;&nbsp; │ &nbsp;&nbsp;&nbsp; │ age, sex, site &nbsp; │<br>'
            '└────────┬────────┘ &nbsp;&nbsp;&nbsp; └────────┬─────────┘<br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; │ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; │<br>'
            '&nbsp; EfficientNet-B0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MLP (12→64→32)<br>'
            '&nbsp; (1280 признаков) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (32 признака)<br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; │ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; │<br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └───────── Concat ─────────┘<br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; │<br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Классификатор (256→8)<br>'
            '</div>'

            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem">'
            '<div>'
            '<div style="color:#6C63FF;font-weight:600;margin-bottom:0.3rem">Визуальная ветка</div>'
            '<div style="color:#ccd6f6">EfficientNet-B0 (pretrained ImageNet)</div>'
            '</div>'
            '<div>'
            '<div style="color:#6C63FF;font-weight:600;margin-bottom:0.3rem">Ветка метаданных</div>'
            '<div style="color:#ccd6f6">MLP: возраст + пол + локализация (12 признаков)</div>'
            '</div>'
            '<div>'
            '<div style="color:#6C63FF;font-weight:600;margin-bottom:0.3rem">Датасет</div>'
            '<div style="color:#ccd6f6">ISIC 2019 + HAM10000 + ISIC 2020 — 31 331 дерматоскопическое изображение + метаданные пациентов</div>'
            '</div>'
            '<div>'
            '<div style="color:#6C63FF;font-weight:600;margin-bottom:0.3rem">Классы</div>'
            '<div style="color:#ccd6f6">8 типов новообразований (включая SCC)</div>'
            '</div>'
            '<div>'
            '<div style="color:#6C63FF;font-weight:600;margin-bottom:0.3rem">Accuracy</div>'
            '<div style="color:#ccd6f6">77.7% (argmax) / 75.8% (с порогом меланомы)</div>'
            '</div>'
            '<div>'
            '<div style="color:#6C63FF;font-weight:600;margin-bottom:0.3rem">Melanoma Recall</div>'
            '<div style="color:#ccd6f6">78.1% (с threshold tuning, порог 0.21)</div>'
            '</div>'
            '</div>'

            '<div style="margin-top:1.5rem;color:#8892b0;line-height:1.7">'
            '<strong style="color:#6C63FF">Этапы обучения:</strong><br>'
            '1. Transfer learning — EfficientNet-B0 предобучен на ImageNet (14 млн фото)<br>'
            '2. Дообучение на ISIC 2019 + HAM10000 + ISIC 2020 — 31 331 дерматоскопическое фото из клиник Австрии, Австралии, Испании и США<br>'
            '3. Мультимодальное обучение — к визуальным признакам добавлены возраст, пол и локализация образования<br>'
            '4. Обучение на GPU (NVIDIA RTX 4070) — 30 эпох, batch size 64, mixed precision (AMP)<br>'
            '5. Threshold tuning — подбор порога чувствительности для меланомы (приоритет: не пропустить опасный диагноз)<br>'
            '6. Постобработка по анкете — дополнительная коррекция по факторам риска (солярий, генетика, фототип кожи)'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with tab4:
        st.markdown(
            '<div class="glass-card">'
            '<div style="color:#ccd6f6;line-height:1.8">'
            'Анкета рассчитывает <strong>профиль риска</strong> на основе клинических данных:<br><br>'
            '<strong style="color:#FF6B6B">Факторы риска меланомы:</strong><br>'
            '• Солярий регулярно → x1.6 &nbsp; • Меланома у родственников → x1.8<br>'
            '• 50+ родинок → x1.5 &nbsp; • Светлая кожа (I тип) → x1.5<br>'
            '• Быстрое изменение образования → x1.6 &nbsp; • 5+ ожогов → x1.5<br><br>'
            '<strong style="color:#FFA500">Факторы риска баз. карциномы:</strong><br>'
            '• Локализация на лице/носу/ушах → x1.4<br>'
            '• Хроническое солнечное воздействие → x1.4<br>'
            '• Светлая кожа, возраст 60+ → x1.5<br><br>'
            'Множители корректируют вероятности модели и <strong>снижают порог '
            'чувствительности к меланоме</strong> для пациентов из группы риска.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with tab5:
        st.markdown(
            '<div class="glass-card">'

            '<div style="color:#6C63FF;font-size:1.1rem;font-weight:700;margin-bottom:1rem">'
            'Технологический стек проекта</div>'

            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem">'

            # Язык и ML-фреймворк
            '<div>'
            '<div style="color:#6C63FF;font-weight:600;margin-bottom:0.5rem">🐍 Язык и ML-фреймворк</div>'
            '<div style="color:#ccd6f6;line-height:1.8;font-size:0.9rem">'
            '<strong>Python 3</strong> — основной язык проекта<br>'
            '<strong>PyTorch</strong> — обучение нейросети, inference<br>'
            '<strong>torchvision</strong> — EfficientNet-B0, аугментации, transforms<br>'
            '<strong>CUDA + AMP</strong> — GPU-ускорение и mixed precision на RTX 4070'
            '</div>'
            '</div>'

            # Компьютерное зрение
            '<div>'
            '<div style="color:#6C63FF;font-weight:600;margin-bottom:0.5rem">👁 Компьютерное зрение</div>'
            '<div style="color:#ccd6f6;line-height:1.8;font-size:0.9rem">'
            '<strong>EfficientNet-B0</strong> — backbone, transfer learning с ImageNet<br>'
            '<strong>Grad-CAM</strong> — визуализация областей внимания модели<br>'
            '<strong>Pillow (PIL)</strong> — загрузка и обработка изображений<br>'
            '<strong>Data Augmentation</strong> — flip, rotation, color jitter, affine'
            '</div>'
            '</div>'

            # Data Science
            '<div>'
            '<div style="color:#6C63FF;font-weight:600;margin-bottom:0.5rem">📊 Data Science</div>'
            '<div style="color:#ccd6f6;line-height:1.8;font-size:0.9rem">'
            '<strong>Pandas</strong> — обработка метаданных пациентов, EDA<br>'
            '<strong>NumPy</strong> — числовые операции, threshold tuning<br>'
            '<strong>scikit-learn</strong> — train/test split, classification report, '
            'confusion matrix, ROC-AUC, label binarization'
            '</div>'
            '</div>'

            # Визуализация
            '<div>'
            '<div style="color:#6C63FF;font-weight:600;margin-bottom:0.5rem">📈 Визуализация</div>'
            '<div style="color:#ccd6f6;line-height:1.8;font-size:0.9rem">'
            '<strong>Matplotlib</strong> — графики обучения, confusion matrix<br>'
            '<strong>Seaborn</strong> — EDA: распределения, корреляции, heatmaps<br>'
            '<strong>Streamlit</strong> — интерактивное веб-приложение'
            '</div>'
            '</div>'

            # Деплой и инфраструктура
            '<div>'
            '<div style="color:#6C63FF;font-weight:600;margin-bottom:0.5rem">🚀 Деплой и инфраструктура</div>'
            '<div style="color:#ccd6f6;line-height:1.8;font-size:0.9rem">'
            '<strong>Streamlit Cloud / HuggingFace Spaces</strong> — хостинг приложения<br>'
            '<strong>NVIDIA RTX 4070</strong> — обучение модели (30 эпох, ~75 мин)<br>'
            '<strong>SSH</strong> — удалённый запуск обучения на GPU-сервере'
            '</div>'
            '</div>'

            # Методология
            '<div>'
            '<div style="color:#6C63FF;font-weight:600;margin-bottom:0.5rem">🧬 Методология</div>'
            '<div style="color:#ccd6f6;line-height:1.8;font-size:0.9rem">'
            '<strong>Transfer Learning</strong> — дообучение предобученной сети<br>'
            '<strong>Multimodal Learning</strong> — изображение + метаданные<br>'
            '<strong>Threshold Tuning</strong> — оптимизация порога для меланомы<br>'
            '<strong>WeightedRandomSampler</strong> — балансировка классов'
            '</div>'
            '</div>'

            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown(
        '<div style="text-align:center;padding:2rem 0 1rem;color:#5a6178;font-size:0.8rem">'
        'DK DermInsights v1.0 &nbsp;|&nbsp; '
        'Дарья Хотулева &nbsp;|&nbsp; '
        'Врач-дерматолог &amp; Data Scientist &nbsp;|&nbsp; '
        'Не является медицинским устройством'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
