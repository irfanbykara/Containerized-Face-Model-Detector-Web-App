from django.shortcuts import render,redirect
import torch
from torch.utils.data import Dataset, DataLoader
from django.core.files.uploadedfile import InMemoryUploadedFile
from torchvision import transforms
from PIL import Image
from deepface import DeepFace
from efficientnet_pytorch import EfficientNet
import cv2
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import os
from django.contrib import messages #import messages

from .models import PredictedImage
import tempfile

class EffNetCNN(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # 선학습 모델에서는 특징 추출 부분만 사용하고
        # (N,3,380,280)->(N,1792,12,12)
        self.features = backbone

        # (N,1792,12,12)->(N,1792,1,1)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = torch.nn.Sequential(
            ##############################################
            # WRITE YOUR CODE HERE
            # 여기서 self.features가 출력하는 모양을 flatten 했을 때
            # 노드 수를 계산하여
            # 계산된 노드수->100으로 가는 완전 연결층을 구성하기
            torch.nn.Linear(backbone._fc.in_features, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, 2),

            torch.nn.LogSoftmax(dim=-1)
            ##############################################
        )

    def forward(self, x):
        self.fmap = self.features.extract_features(x) # (N,3,300,300)->(N,1920,9,9)

        N = self.fmap.shape[0]
        x = self.avg_pool(self.fmap).reshape(N,-1) # (N,1920,9,9)->(N,1920,1,1)->(N,1920)
        x = self.classifier(x) #(N,1920)->(N,100)

        return x

# Create your views here.
def home(request):

    if request.method == "POST":
        data = request.POST
        uploaded_image = request.FILES["images"]
        # Create a temporary file on disk if the uploaded file is an InMemoryUploadedFile
        if isinstance(uploaded_image, InMemoryUploadedFile):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_image.read())
            temp_file.close()
            img_path = temp_file.name
        else:
            img_path = uploaded_image.temporary_file_path()
        print(img_path)
        if data.get("selection") == "extreme":
            path_selection = "main/vaelsys_best_final.pth"
        else:
            path_selection = "main/vaelsys_best.pth"

        effnetb4 = EfficientNet.from_pretrained('efficientnet-b4')
        model = EffNetCNN(effnetb4)
        model.to("cpu")
        
        try:
            detected_face = DeepFace.extract_faces(img_path=img_path, detector_backend="DeepFace")
            detected_face = detected_face[0]["face"]
            # Convert the NumPy array to a PIL Image
            detected_face = Image.fromarray(cv2.cvtColor(detected_face * 255, cv2.COLOR_RGB2BGR))
        except:
            #print("This image has no face!")
            messages.warning(request, "Esta imagen no tiene cara!")
            return redirect("home")

        test_transform = transforms.Compose([
            transforms.Resize((380, 380)),  # Resize the image to a specific size
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])


        checkpoint = torch.load(path_selection, map_location=torch.device('cpu'))

        # If the checkpoint contains the model state dictionary directly
        model.load_state_dict(checkpoint)
        # image = Image.open(detected_face).convert('RGB')

        # Apply transformations
        image = test_transform(detected_face)
        model.eval()
        # Forward pass through the model
        with torch.no_grad():
            output = model(image.unsqueeze(0).to("cpu"))
           # probs = torch.exp(output.data)
            _, predicted = torch.max(output.data, 1)
            predicted_label = predicted.item()


        new_image_instance = PredictedImage(image=uploaded_image.temporary_file_path(),name=str(uploaded_image), predicted_label=predicted_label)
        new_image_instance.save()

        context = {"instance":new_image_instance}
        return render(request,"main/home.html",context)

    return render(request,"main/home.html")