"""
Model Trainer Module

Unified model-training utilities for computer vision, NLP, and tabular domains.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision.models as models
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import inspect
from typing import Dict, Tuple, Optional
import time
from tqdm import tqdm


class CVModelTrainer:
    """Computer-vision training utilities."""
    
    @staticmethod
    def create_resnet50(num_classes: int = 100):
        """ResNet-50 model adapted to CIFAR-sized inputs."""
        model = models.resnet50(weights=None)
        # CIFAR-100 uses 32x32 inputs, so the ImageNet stem is too aggressive.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    @staticmethod
    def train(model, train_loader, val_loader, hyperparameters: Dict, device='cuda') -> Dict:
        """Train a CV model and return best validation metrics and history."""
        model = model.to(device)
        
        lr = hyperparameters.get('learning_rate', 0.01)
        batch_size = hyperparameters.get('batch_size', 64)
        weight_decay = hyperparameters.get('weight_decay', 1e-4)
        optimizer_name = hyperparameters.get('optimizer', 'Adam')
        epochs = hyperparameters.get('epochs', 10)
        momentum = hyperparameters.get('momentum', 0.9)
        label_smoothing = hyperparameters.get('label_smoothing', 0.0)
        dropout_rate = hyperparameters.get('dropout_rate', 0.0)
        scheduler_name = hyperparameters.get('scheduler', 'None')
        warmup_steps = int(hyperparameters.get('warmup_steps', 0))
        
        if dropout_rate > 0:
            if isinstance(model.fc, nn.Linear):
                in_features = model.fc.in_features
                out_features = model.fc.out_features
                model.fc = nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(in_features, out_features)
                )
            model = model.to(device)

        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        scheduler = None
        if scheduler_name == 'StepLR':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
        elif scheduler_name == 'CosineAnnealing':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
            
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        start_time = time.time()
        
        best_val_acc = 0.0
        global_step = 0
        history = {'train_loss': [], 'val_acc': [], 'val_loss': []}
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Warmup
                if global_step < warmup_steps:
                    warmup_percent = global_step / warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr * warmup_percent
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                global_step += 1
                
                pbar.set_postfix({'loss': loss.item()})
            
            # Scheduler step
            if scheduler:
                if scheduler_name == 'ReduceLROnPlateau':
                    # Validation accuracy is calculated below, so we step it there
                    pass 
                else:
                    scheduler.step()

            val_acc, val_loss = CVModelTrainer.evaluate(model, val_loader, device)
            
            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(val_acc)
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
        
        training_time = time.time() - start_time
        
        return {
            'accuracy': best_val_acc,
            'loss': val_loss,
            'training_time': training_time,
            'history': history,
            'model_state_dict': best_model_state
        }
    
    @staticmethod
    def evaluate(model, data_loader, device='cuda') -> Tuple[float, float]:
        """Evaluate classification accuracy and loss."""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(data_loader)
        
        return accuracy, avg_loss


class NLPModelTrainer:
    """Text-classification training utilities based on Hugging Face Trainer."""
    
    @staticmethod
    def train(
        train_dataset,
        val_dataset,
        hyperparameters: Dict,
        model_name='distilbert-base-uncased',
        num_labels: Optional[int] = None
    ) -> Dict:
        """Fine-tune a sequence-classification model and return validation metrics."""
        lr = hyperparameters.get('learning_rate', 2e-5)
        batch_size = hyperparameters.get('batch_size', 16)
        weight_decay = hyperparameters.get('weight_decay', 0.01)
        epochs = hyperparameters.get('epochs', 3)
        
        # Infer class count from dataset labels if not explicitly provided.
        if num_labels is None:
            try:
                label_feature = train_dataset.features.get("label")
                if hasattr(label_feature, "num_classes"):
                    num_labels = int(label_feature.num_classes)
                else:
                    num_labels = int(len(set(train_dataset["label"])))
            except Exception:
                num_labels = 2

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
        # Accuracy metric definition
        import numpy as np
        from sklearn.metrics import accuracy_score
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {"accuracy": accuracy_score(labels, predictions)}
        
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        ta_kwargs = {
            "output_dir": "./results",
            "learning_rate": lr,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": epochs,
            "weight_decay": weight_decay,
            "logging_dir": "./logs",
            "logging_steps": 100,
            "save_strategy": "no",
            "report_to": []
        }
        # Support both older/newer transformers argument names.
        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        if "evaluation_strategy" in ta_params:
            ta_kwargs["evaluation_strategy"] = "epoch"
        else:
            ta_kwargs["eval_strategy"] = "epoch"
        training_args = TrainingArguments(**ta_kwargs)
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        eval_results = trainer.evaluate()
        
        return {
            'accuracy': eval_results.get('eval_accuracy', 0.0),
            'loss': eval_results.get('eval_loss', 0.0),
            'training_time': training_time,
            'history': trainer.state.log_history, # HuggingFace Trainer logs
            'model_state_dict': None # HuggingFace handles saving differently, skipping for now or return model.state_dict()
        }


class TabularModelTrainer:
    """Simple MLP trainer for tabular classification tasks."""
    
    @staticmethod
    def create_mlp(input_dim: int, hidden_dims: list = [128, 64], num_classes: int = 2):
        """Build a feed-forward classifier."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def train(X_train, y_train, X_val, y_val, hyperparameters: Dict, device='cuda') -> Dict:
        """Train a tabular MLP and return best validation metrics and history."""
        lr = hyperparameters.get('learning_rate', 0.001)
        batch_size = hyperparameters.get('batch_size', 64)
        weight_decay = hyperparameters.get('weight_decay', 1e-4)
        epochs = hyperparameters.get('epochs', 50)
        
        input_dim = X_train.shape[1]
        model = TabularModelTrainer.create_mlp(input_dim).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)
        
        start_time = time.time()
        
        start_time = time.time()
        
        best_val_acc = 0.0
        history = {'train_loss': [], 'val_acc': [], 'val_loss': []}
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                _, predicted = torch.max(val_outputs.data, 1)
                val_acc = (predicted == y_val_tensor).sum().item() / len(y_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
            
            history['train_loss'].append(loss.item())
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
        
        training_time = time.time() - start_time
        
        return {
            'accuracy': best_val_acc,
            'loss': val_loss,
            'training_time': training_time,
            'history': history,
            'model_state_dict': best_model_state
        }
