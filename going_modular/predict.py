import torch
import matplotlib.pyplot as plt
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_image(model, img, class_names):
    """
    Predict the class of an image and displays the image, with title as the predicted class
    also prints out the probability of each class
    """
    transformimg = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    model.eval()
    with torch.inference_mode():
        img = transformimg(img).to(device)
        img = img.to(device)
        pred_logits = model(img.unsqueeze(0))
        pred_label = pred_logits.argmax(dim=1)
        pred_prob = torch.softmax(pred_logits, dim=1)[0][pred_label].item()
        plt.imshow(img.permute(1, 2, 0).cpu())
        plt.title(f"Predicted: {class_names[pred_label.item()]}")
        plt.axis("off")
        plt.show()

        for class_name, prob in zip(class_names, torch.softmax(pred_logits, dim=1)[0]):
            print(f"{class_name}: {prob:.4f}")