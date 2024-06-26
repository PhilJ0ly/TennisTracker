# Philippe Joly

from ultralytics import YOLO

model = YOLO('models/yolov5_best.pt')

result = model.predict('input_video/image.png', conf=0.2, save=True)

# print(result)
# print("\nBoxes:")
# for box in result[0].boxes:
#     print(box)
