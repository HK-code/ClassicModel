import torch
import torchvision
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# 加载 CIFAR10 数据集
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# 加载 CLIP 模型和处理器
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 为 CIFAR10 类别生成文本描述
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'hip', 'truck']
text_descriptions = [f"a photo of a {c}" for c in classes]

# 在推理时使用模型
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(testloader):
        # print(torch.max(images), torch.min(images))
        images = images.to(device)
        labels = labels.to(device)

        # 对图像进行预处理
        inputs = processor(images=images, return_tensors="pt").to(device)

        # 计算图像的特征
        image_features = model.get_image_features(**inputs)

        # 将文本描述转换为模型可接受的格式并计算文本特征
        text_inputs = processor(text=text_descriptions, return_tensors="pt").to(device)
        text_features = model.get_text_features(**text_inputs)

        # 计算图像和文本的相似度
        similarity_scores = image_features @ text_features.t()

        # 获取预测的类别
        _, predicted = torch.max(similarity_scores, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {(100 * correct / total):.2f}%")