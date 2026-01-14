import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns

class MasterplanClassifier:
    def __init__(self, data_path, model_save_path='masterplan_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")
        
        self.data_path = data_path
        self.model_save_path = model_save_path
        
        # Data transforms with augmentation for training
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # No augmentation for validation/test
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.load_data()
        self.build_model()
    
    def load_data(self):
        """Load and prepare datasets"""
        train_path = os.path.join(self.data_path, 'train')
        val_path = os.path.join(self.data_path, 'val')
        test_path = os.path.join(self.data_path, 'test')
        
        # Load datasets
        self.train_dataset = datasets.ImageFolder(train_path, transform=self.train_transform)
        self.val_dataset = datasets.ImageFolder(val_path, transform=self.val_transform)
        self.test_dataset = datasets.ImageFolder(test_path, transform=self.val_transform)
        
        print(f"\n📊 Dataset Loaded:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")
        print(f"  Classes: {self.train_dataset.classes}")
        
        # REDUCED BATCH SIZE TO PREVENT MEMORY ERROR
        batch_size = 8
        
        # On Windows, num_workers > 0 can cause issues; set to 0 or 1
        num_workers = 1 if os.name == 'nt' else 2
        pin_memory = True if self.device.type == 'cuda' else False
        
        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, 
                                       num_workers=num_workers, pin_memory=pin_memory)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, 
                                     num_workers=num_workers, pin_memory=pin_memory)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, 
                                      num_workers=num_workers, pin_memory=pin_memory)
    
    def build_model(self):
        """Build ResNet-18 model with updated syntax"""
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-10]:
            param.requires_grad = False
        
        # Modify final layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # 2 classes: masterplan, not_masterplan
        )
        
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{running_loss/(pbar.n+1):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        return running_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self, loader):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        return running_loss / len(loader), accuracy, all_preds, all_labels
    
    def train(self, epochs=30):
        """Full training loop"""
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        best_val_acc = 0.0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(epochs):
            print(f"\n📅 Epoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, _, _ = self.validate(self.val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_names': self.train_dataset.classes
                }, self.model_save_path)
                print(f"  ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Plot training history
        self.plot_history(train_losses, val_losses, train_accs, val_accs)
        
        return train_losses, val_losses, train_accs, val_accs
    
    def plot_history(self, train_losses, val_losses, train_accs, val_accs):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(train_accs, label='Train Acc')
        ax2.plot(val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        print("\n📊 Training history saved to 'training_history.png'")
    
    def test(self):
        """Test the model"""
        print("\n🧪 Testing model...")
        
        if not os.path.exists(self.model_save_path):
            print(f"⚠️ No saved model found at {self.model_save_path}. Using current model.")
        else:
            checkpoint = torch.load(self.model_save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_acc, all_preds, all_labels = self.validate(self.test_loader)
        
        print(f"\n📈 Test Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        
        # Classification report
        class_names = self.train_dataset.classes
        print("\n📋 Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("📊 Confusion matrix saved to 'confusion_matrix.png'")

if __name__ == "__main__":
    DATA_PATH = "processed_data"
    MODEL_SAVE_PATH = "masterplan_model.pth"
    
    print("🏗️ Masterplan Classifier Training")
    print("=" * 50)
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Processed data not found at '{DATA_PATH}'")
        print("Please run preprocess.py first!")
    else:
        classifier = MasterplanClassifier(DATA_PATH, MODEL_SAVE_PATH)
        classifier.train(epochs=30)
        classifier.test()
        print("\n✅ Training complete!")
        print(f"Model saved to: {MODEL_SAVE_PATH}")