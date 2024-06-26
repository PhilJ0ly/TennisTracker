import torch 
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        # only on first frame as camera not moving

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image_tensor = self.transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        kps = outputs.squeeze().cpu().numpy()
        original_h, original_w = img_rgb.shape[:2]

        scale_x, scale_y = 230.0, 235.0
        x_shift, y_shift = 20, 27

        kps[::2] *= original_w/scale_x
        kps[::2] += x_shift

        kps[1::2] *= original_h/scale_y
        kps[1::2] += y_shift

        return kps
    
    def draw_kps(self, image, kps):
        for i in range(0, len(kps), 2):
            x = int(kps[i])
            y = int(kps[i+1])

            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            cv2.circle(image, (x,y), 5, (255, 0, 0), -1)
        return image
    
    def draw_kps_video(self, video_frames, kps):
        output_frames = []
        for frame in video_frames:
            frame = self.draw_kps(frame, kps)
            output_frames.append(frame)
        return output_frames