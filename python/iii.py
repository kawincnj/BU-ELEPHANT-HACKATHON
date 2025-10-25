from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Export the model to TorchScript format
model.export(format="torchscript")  # creates 'yolo11n.torchscript'

# Load the exported TorchScript model
torchscript_model = YOLO("yolo11n.torchscript")

# Run inference
results = torchscript_model("https://ultralytics.com/images/bus.jpg")