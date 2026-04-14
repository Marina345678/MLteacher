import gradio as gr
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
IMAGE_SIZE = 512
CLASS_NAMES = ['Edema', 'Fibrin', 'Granulation tissue', 'Healthy skin', 'Necrosis']
CLASS_COLORS = {
    0: (255, 0, 0),  # Edema - Синий
    1: (0, 255, 255),  # Fibrin - Желтый
    2: (0, 0, 255),  # Granulation - Красный
    3: (0, 255, 0),  # Healthy skin - Зеленый
    4: (0, 0, 0)  # Necrosis - Черный
}

# Загрузка модели
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
    activation='softmax2d'
)
model.load_state_dict(torch.load(r"D:\MLteacher\3\best_wound_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()


def process_image(input_image):
    """Основная функция для Gradio"""
    # Конвертация из PIL в numpy
    if isinstance(input_image, np.ndarray):
        image_rgb = input_image
    else:
        image_rgb = np.array(input_image)

    original_h, original_w = image_rgb.shape[:2]

    # Трансформации
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    input_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(DEVICE)

    # Инференс
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Ресайз маски
    pred_mask = cv2.resize(pred_mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # Создание цветной маски
    vis_mask = np.zeros((original_h, original_w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        vis_mask[pred_mask == class_id] = color

    # Расчет процентов
    total_pixels = original_h * original_w
    percentages = {}
    for class_id, name in enumerate(CLASS_NAMES):
        count = np.sum(pred_mask == class_id)
        percent = (count / total_pixels) * 100
        percentages[name] = percent

    # Определение стадии
    granulation = percentages['Granulation tissue']
    fibrin = percentages['Fibrin']
    necrosis = percentages['Necrosis']

    if necrosis > 10:
        stage = " КРИТИЧЕСКАЯ: Обширный некроз"
    elif necrosis > 3:
        stage = " СТАДИЯ НЕКРОЗА"
    elif fibrin > granulation and fibrin > 20:
        stage = "СТАДИЯ ФИБРИНА (воспаление)"
    elif granulation > 30:
        stage = " СТАДИЯ ГРАНУЛЯЦИИ (активное заживление)"
    else:
        stage = "СТАДИЯ ЭПИТЕЛИЗАЦИИ"

    # Формирование текста результата
    result_text = f"{stage}\n\n"
    result_text += "Состав раны:\n"
    for name, perc in percentages.items():
        result_text += f"- {name}: {perc:.1f}%\n"

    return vis_mask, result_text


# Создание интерфейса
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(label="Загрузите фото раны"),
    outputs=[
        gr.Image(label="Сегментация"),
        gr.Textbox(label="Результат анализа", lines=10)
    ],
    title="Анализ заживления раны",
    description="Загрузите фотографию раны, и ИИ определит стадию заживления и состав тканей",
    examples=[]
)

if __name__ == "__main__":
    iface.launch(share=True)  # share=True даст публичную ссылку