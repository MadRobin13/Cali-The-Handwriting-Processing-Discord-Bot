from ultralytics import YOLO
import pandas as pandas
import matplotlib.pyplot as plt
import numpy as np

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco8.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
results = model("./bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format

print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
print(results.boxes)  # print results
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

# plt.imshow(np.squeeze(results.plot()))
# plt.show()