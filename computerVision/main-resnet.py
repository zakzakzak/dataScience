import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# load model pretrained
model = models.resnet50(pretrained=True)
model.eval()

# transform sesuai ImageNet
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# load label ImageNet (opsional)
labels = open("imagenet_classes.txt").read().splitlines()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)

    _, predicted = outputs.max(1)
    label = labels[predicted.item()]

    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2)

    cv2.imshow("Webcam AI", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()