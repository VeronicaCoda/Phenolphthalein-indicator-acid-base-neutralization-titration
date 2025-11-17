import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from resnet import resnet34


class MyDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def white_balance(img_pil):
    img_np = np.array(img_pil)
    b, g, r = cv2.split(img_np)
    m, n = img_np.shape[:2]
    sum_ = b + g + r
    hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
    Y = 765
    num = 0
    ratio = 0.01
    key = next((Y for Y in range(765, -1, -1) if (num := num + hists[Y]) > m * n * ratio / 100), 0)

    mask = sum_ >= key
    sum_b, sum_g, sum_r = b[mask].sum(), g[mask].sum(), r[mask].sum()
    time = mask.sum()

    avg_b, avg_g, avg_r = sum_b / time, sum_g / time, sum_r / time
    maxvalue = float(np.max(img_np))

    img_np = np.clip(img_np * [maxvalue / avg_b, maxvalue / avg_g, maxvalue / avg_r], 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            # transforms.Lambda(lambda img: white_balance(img)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        "val": transforms.Compose([
            # transforms.Lambda(lambda img: white_balance(img)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, "pythonProject/Picture_Train", "data_cam3")
    print(image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    classes, class_to_idx = datasets.folder.find_classes(image_path)
    imgs, labels = [], []
    for class_name in classes:
        class_index = class_to_idx[class_name]
        class_dir = os.path.join(image_path, class_name)
        for root, _, fnames in sorted(os.walk(class_dir)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if datasets.folder.is_image_file(path):
                    imgs.append(path)
                    labels.append(class_index)

    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}
    with open('idx_to_class.json', 'w') as json_file:
        json.dump(idx_to_class, json_file, indent=4)

    best_fold_acc = 0.0
    best_fold_model_state = None
    best_model_path = './Best_Model_Weightscam3.pth'

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(imgs, labels)):
        print(f"Fold {fold + 1}")
        print("-------")
        print(f"Train index size: {len(train_idx)}, Val index size: {len(val_idx)}")
        print(f"Labels size: {len(labels)}")

        train_imgs = [imgs[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_imgs = [imgs[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        train_dataset = MyDataset(train_imgs, train_labels, transform=data_transform["train"])
        val_dataset = MyDataset(val_imgs, val_labels, transform=data_transform["val"])

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        net = resnet34(num_classes=2)
        net.to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
        epochs = 10
        best_acc = 0.0
        save_path = f'./2_Color_Model_Phenolphthalein_Fold{fold + 1}.pth'
        train_steps = len(train_loader)
        early_stop_patience = 3
        no_improve_epochs = 0

        for epoch in range(epochs):
            net.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, label = data
                optimizer.zero_grad()
                outputs = net(images.to(device))
                loss = loss_function(outputs, label.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] "

            net.eval()
            acc = 0.0
            with torch.no_grad():
                val_bar = tqdm(val_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_label = val_data
                    outputs = net(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_label.to(device)).sum().item()
            val_accurate = acc / len(val_dataset)
            print(f'[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f}  val_accuracy: {val_accurate:.3f}')

            scheduler.step(val_accurate)

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= early_stop_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        if best_acc > best_fold_acc:
            best_fold_acc = best_acc
            best_fold_model_state = net.state_dict()
            torch.save(best_fold_model_state, best_model_path)
            print(f'Best model saved with accuracy: {best_fold_acc:.3f}')

    print('Finished Training')


if __name__ == '__main__':
    main()