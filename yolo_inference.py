# Philippe Joly

from ultralytics import YOLO

model = YOLO('models/key_pts.pt')

result = model.track('input_video/input_video.mp4', conf=0.2, save=True)

# print(result)
# print("\nBoxes:")
# for box in result[0].boxes:
#     print(box)
