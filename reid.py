import torch
import torchvision
models = torchvision.models
T = torchvision.transform
import numpy as np
import cv2

class ReID:
    def __init__(self, device=None):
        self.device = 'cpu' # You can change device here
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Identity()
        self.model.to(self.device).eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.embeddings = {}

    def extract_embedding(self, image):
        with torch.no_grad():
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            emb = self.model(tensor).cpu().numpy().flatten()
            return emb / np.linalg.norm(emb)

    def match(self, new_emb, threshold=0.5):
        best_id = None
        best_sim = -1

        for track_id, emb in self.embeddings.items():
            sim = np.dot(new_emb, emb)
            if sim > threshold and sim > best_sim:
                best_sim = sim
                best_id = track_id

        return best_id

    def register(self, emb, track_id):
        self.embeddings[track_id] = emb
