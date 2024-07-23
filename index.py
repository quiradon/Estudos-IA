import cv2
from ultralytics import YOLO
import os
import json

# Carregar o modelo YOLOv8 pré-treinado aqui!
model = YOLO("yolov8n.pt")

# Definir o nível de precisão mínimo para considerar uma detecção válida
min_confidence = 0.2

input_folder = "in"
output_folder = "out"
files = os.listdir(input_folder)
processed_count = 0
detections = {}
for file_name in files:
    file_path = os.path.join(input_folder, file_name)
    image = cv2.imread(file_path)
    if image is not None:
        results = model(image)
        image_detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if confidence > min_confidence:
                    # Converte as coordenadas para lista antes de adicionar ao dicionário
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{result.names[cls_id]} {confidence:.2f}"
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    image_detections.append({"class": result.names[cls_id], "confidence": confidence, "bbox": [x1, y1, x2, y2]})
        detections[file_name] = image_detections
        cv2.imwrite(os.path.join(output_folder, file_name), image)
        processed_count += 1
        print(f"Imagem {processed_count}/{len(files)} processada: {file_name}")

with open("detections.json", "w") as json_file:
    json.dump(detections, json_file, indent=4)
print(f"Processamento concluído. Total de imagens processadas: {processed_count}. Resultados salvos em 'detections.json'")