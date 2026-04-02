import gradio as gr
import cv2
import json
import numpy as np
import os
from pipeline import DrawingPipeline

# ===== INIT MODEL =====
# Tự động kiểm tra CPU/GPU để tránh crash trên Spaces
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pipeline = DrawingPipeline(
    config='custom_ds_config.py',
    checkpoint='epoch_23.pth',
    device=device
)


# ===== DRAW BBOX (An toàn hơn) =====
def draw_boxes(image, result):
    img = image.copy()
    colors = {
        "Note": (0, 255, 0),  # Xanh lá
        "PartDrawing": (255, 0, 0),  # Xanh dương
        "Table": (0, 0, 255)  # Đỏ
    }

    # Kiểm tra xem result có đúng cấu trúc không để tránh lỗi dict
    objects = result.get("objects", [])

    for obj in objects:
        cls = obj.get("class", "Unknown")
        bbox = obj.get("bbox", {})

        # Lấy tọa độ, nếu thiếu thì bỏ qua thay vì crash
        try:
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            color = colors.get(cls, (255, 255, 255))  # Trắng nếu không thuộc 3 loại trên

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, cls, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except (KeyError, TypeError):
            continue

    return img


# ===== MAIN FUNCTION =====
def process(image):
    if image is None:
        return None, "No image uploaded", ""

    # Gradio truyền ảnh dạng RGB (Numpy), MMDetection/OpenCV cần BGR
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Lưu ảnh tạm thời để pipeline xử lý
    temp_path = "temp_input.jpg"
    cv2.imwrite(temp_path, img_bgr)

    try:
        # Chạy pipeline
        result = pipeline.process_image(temp_path, output_dir="output")

        # Vẽ bbox
        vis_img_bgr = draw_boxes(img_bgr, result)
        vis_img_rgb = cv2.cvtColor(vis_img_bgr, cv2.COLOR_BGR2RGB)

        # Chuẩn bị JSON panel
        json_text = json.dumps(result, indent=2, ensure_ascii=False)

        # Chuẩn bị OCR panel
        ocr_text = ""
        for obj in result.get("objects", []):
            cls = obj.get("class")
            if cls in ["Note", "Table"]:
                content = obj.get("ocr_content", "No content")
                obj_id = obj.get("id", "?")
                ocr_text += f"--- [{cls} #{obj_id}] ---\n{content}\n\n"

        if not ocr_text:
            ocr_text = "No OCR content found for Note or Table."

        return vis_img_rgb, json_text, ocr_text

    except Exception as e:
        return image, f"Error during processing: {str(e)}", ""


# ===== UI =====
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛠️ Drawing Detection & OCR")
    gr.Markdown("Upload bản vẽ kỹ thuật để nhận diện **Note**, **PartDrawing**, và **Table**.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Input Drawing")
            btn = gr.Button("Start Detection", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Visualization")

    with gr.Row():
        json_output = gr.Textbox(label="Data (JSON)", lines=15)
        ocr_output = gr.Textbox(label="OCR Content", lines=15)

    btn.click(
        fn=process,
        inputs=input_image,
        outputs=[output_image, json_output, ocr_output]
    )

if __name__ == "__main__":
    demo.launch(share = True)