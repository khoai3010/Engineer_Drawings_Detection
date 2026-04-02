import os
import json
import cv2
import numpy as np
import easyocr
import torch
from mmdet.apis import init_detector, inference_detector


class DrawingPipeline:
    def __init__(self, config, checkpoint, device='cpu'):
        # Tự động chuyển sang CPU nếu không có CUDA (Hugging Face Spaces)
        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'

        self.model = init_detector(config, checkpoint, device=device)
        self.reader = easyocr.Reader(['en'])
        self.classes = ['Note', 'PartDrawing', 'Table']

    def preprocess_for_ocr(self, img):
        """Cải thiện chất lượng ảnh cho OCR"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Sử dụng Adaptive Threshold thay vì cứng 150 để linh hoạt hơn
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh

    def extract_table_structure(self, crop):
        """Chuyển kết quả OCR thành cấu trúc bảng (hàng/cột)"""
        ocr_result = self.reader.readtext(crop)
        if not ocr_result:
            return []

        words = []
        for res in ocr_result:
            # res format: [ [[x1,y1], [x2,y1], [x2,y2], [x1,y2]], text, conf ]
            bbox, text, conf = res
            x, y = bbox[0][0], bbox[0][1]
            words.append((x, y, text))

        # Sắp xếp theo y để nhóm hàng
        words.sort(key=lambda x: x[1])

        rows = []
        current_row = []
        row_threshold = 15

        for word in words:
            if not current_row:
                current_row.append(word)
                continue

            prev_y = current_row[-1][1]
            if abs(word[1] - prev_y) < row_threshold:
                current_row.append(word)
            else:
                rows.append(current_row)
                current_row = [word]

        if current_row:
            rows.append(current_row)

        table = []
        for row in rows:
            row.sort(key=lambda x: x[0])  # Sắp xếp mỗi hàng theo x
            table.append([w[2] for w in row])

        return table

    def process_image(self, image_path, output_dir='output', conf_thresh=0.5):
        os.makedirs(output_dir, exist_ok=True)

        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"Cannot read image: {image_path}", "objects": []}

        # ===== INFERENCE =====
        result = inference_detector(self.model, image_path)

        # Truy cập pred_instances an toàn (MMDet v3)
        pred_instances = result.pred_instances.cpu().numpy()
        bboxes = pred_instances.bboxes
        scores = pred_instances.scores
        labels = pred_instances.labels

        json_output = {
            "image": os.path.basename(image_path),
            "objects": []
        }

        obj_id = 1
        h, w = img.shape[:2]

        for i in range(len(bboxes)):
            score = float(scores[i])
            if score < conf_thresh:
                continue

            x1, y1, x2, y2 = bboxes[i]
            cls_id = int(labels[i])

            # Đảm bảo class id nằm trong danh sách (Tránh lỗi Index out of range)
            if cls_id >= len(self.classes):
                continue
            cls_name = self.classes[cls_id]

            # Clip tọa độ
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(w, int(x2)), min(h, int(y2))

            crop = img[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                continue

            # ===== LƯU CROP (Tùy chọn) =====
            # crop_path = os.path.join(output_dir, f"{obj_id}.jpg")
            # cv2.imwrite(crop_path, crop)

            # ===== XỬ LÝ OCR THEO LOẠI =====
            ocr_content = ""
            if cls_name == "Note":
                proc_img = self.preprocess_for_ocr(crop)
                ocr_res = self.reader.readtext(proc_img)
                ocr_content = " ".join([r[1] for r in ocr_res])

            elif cls_name == "Table":
                table_data = self.extract_table_structure(crop)
                # Biến table thành chuỗi để hiển thị dễ hơn trong JSON/OCR Panel
                ocr_content = "\n".join([" | ".join(row) for row in table_data])

            # Thêm vào danh sách output
            json_output["objects"].append({
                "id": obj_id,
                "class": cls_name,
                "confidence": score,
                "bbox": {"x1": ix1, "y1": iy1, "x2": ix2, "y2": iy2},
                "ocr_content": ocr_content
            })
            obj_id += 1

        return json_output