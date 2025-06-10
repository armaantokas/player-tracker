from ultralytics import YOLO
import torch

class Detector:
    def __init__(self, model_path='weights/best.pt', device=None):
        self.model = YOLO(model_path)
        self.device = 'cpu' # Change Device Here
        self.model.to(self.device)

    def detect(self, frame, conf_thresh=0.6):
        results = self.model(frame)[0]
        detections = []
        for det in results.boxes.data:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if int(cls) == 2 and conf > conf_thresh: # In My Pretrained Model, Player Class = 2, change accordingly
                detections.append([int(x1), int(y1), int(x2), int(y2), float(conf)])
        return detections
