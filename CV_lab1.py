import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt


# 读取数据与数据预处理
def load_cifar10(data_dir='D:/cifar-10-python/cifar-10-batches-py'):
    #获取数据
    X_train = []
    y_train = []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        with open(batch_file, 'rb') as f:
            data = np.load(f, allow_pickle=True, encoding='bytes')
            X_train.append(data[b'data'])
            y_train.append(data[b'labels'])
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train).astype(np.int64)

    test_file = os.path.join(data_dir, 'test_batch')
    with open(test_file, 'rb') as f:
        data = np.load(f, allow_pickle=True, encoding='bytes')
        X_test = data[b'data']
        y_test = np.array(data[b'labels']).astype(np.int64)

    # 处理数据与标准化
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0

    mean = np.mean(X_train, axis=(0, 1, 2))
    std = np.std(X_train, axis=(0, 1, 2))

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42) #划分测试集与验证集

    return X_train, X_val, X_test, y_train, y_val, y_test

# 模型定义
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, activation='relu', reg_lambda=0.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 10
        self.activation_type = activation  # 激活函数类型
        self.reg_lambda = reg_lambda       # 正则化参数

        # 初始化
        if activation == 'relu':
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            self.W2 = np.random.randn(hidden_size, self.output_size) * np.sqrt(2.0 / hidden_size)
        elif activation == 'sigmoid':
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
            self.W2 = np.random.randn(hidden_size, self.output_size) * np.sqrt(1.0 / hidden_size)
        else:
            raise ValueError("Unsupported activation function")

        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(self.output_size)

    # 激活函数（这里实现了relu与sigmoid）
    def activation(self, z):
        if self.activation_type == 'relu':
            return np.maximum(0, z)
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        else:
            raise ValueError("Unsupported activation function")

    # 激活函数对应导数
    def activation_derivative(self, z):
        if self.activation_type == 'relu':
            return (z > 0).astype(float)
        elif self.activation_type == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        else:
            raise ValueError("Unsupported activation function")

    # 前向传播
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        
        # Softmax
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    # 反向传播
    def backward(self, X, y):
        num_samples = X.shape[0]
        y_one_hot = np.eye(self.output_size)[y]

        delta3 = self.probs - y_one_hot
        dW2 = (self.a1.T @ delta3) / num_samples + self.reg_lambda * self.W2
        db2 = np.sum(delta3, axis=0) / num_samples

        delta2 = (delta3 @ self.W2.T) * self.activation_derivative(self.z1)
        dW1 = (X.T @ delta2) / num_samples + self.reg_lambda * self.W1
        db1 = np.sum(delta2, axis=0) / num_samples
        
        return dW1, db1, dW2, db2

    # 计算交叉熵损失
    def compute_loss(self, X, y):
        probs = self.forward(X)
        corect_logprobs = -np.log(probs[range(len(y)), y])
        data_loss = np.sum(corect_logprobs)
        data_loss += 0.5 * self.reg_lambda * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return data_loss / len(y)

    def evaluate(self, X, y):
        probs = self.forward(X)
        preds = np.argmax(probs, axis=1)
        return np.mean(preds == y)

# 保存模型权重
def save_weights_to_txt(model, file_path):

    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        # 写入W1
        f.write(f"W1_shape: {' '.join(map(str, model.W1.shape))}\n")
        np.savetxt(f, model.W1.flatten(), fmt='%.6f')  # 保留6位小数

        # 写入b1
        f.write(f"b1_shape: {' '.join(map(str, model.b1.shape))}\n")
        np.savetxt(f, model.b1.flatten(), fmt='%.6f')

        # 写入W2
        f.write(f"W2_shape: {' '.join(map(str, model.W2.shape))}\n")
        np.savetxt(f, model.W2.flatten(), fmt='%.6f')

        # 写入b2
        f.write(f"b2_shape: {' '.join(map(str, model.b2.shape))}\n")
        np.savetxt(f, model.b2.flatten(), fmt='%.6f')
class Trainer:
    def __init__(self, model, X_train, y_train, X_val, y_val, 
                 learning_rate=1e-3, lr_decay=0.95, decay_interval=10):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.learning_rate = learning_rate      # 学习率
        self.lr_decay = lr_decay                # 学习率衰减速率
        self.decay_interval = decay_interval    # 衰减周期

        # 最佳模型与其参数
        self.best_val_acc = 0.0
        self.best_params = None

        # loss曲线与accuracy曲线准备
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train(self, num_epochs=100, batch_size=64):
        num_samples = self.X_train.shape[0]
        iterations_per_epoch = num_samples // batch_size
        
        for epoch in range(num_epochs):
            if epoch % self.decay_interval == 0 and epoch != 0:
                self.learning_rate *= self.lr_decay
                print(f"Learning rate decayed to {self.learning_rate:.6f}")

            # 打乱数据，增强鲁棒性， 防止过拟合
            permutation = np.random.permutation(num_samples)
            X_shuffled = self.X_train[permutation]
            y_shuffled = self.y_train[permutation]

            for i in range(iterations_per_epoch):
                start = i * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                self.model.forward(X_batch)
                dW1, db1, dW2, db2 = self.model.backward(X_batch, y_batch)

                self.model.W1 -= self.learning_rate * dW1
                self.model.b1 -= self.learning_rate * db1
                self.model.W2 -= self.learning_rate * dW2
                self.model.b2 -= self.learning_rate * db2

            train_loss = self.model.compute_loss(self.X_train, self.y_train)
            val_loss = self.model.compute_loss(self.X_val, self.y_val)
            val_acc = self.evaluate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            print(f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_params = {
                    'W1': self.model.W1.copy(),
                    'b1': self.model.b1.copy(),
                    'W2': self.model.W2.copy(),
                    'b2': self.model.b2.copy()
                }

        self.model.W1 = self.best_params['W1']
        self.model.b1 = self.best_params['b1']
        self.model.W2 = self.best_params['W2']
        self.model.b2 = self.best_params['b2']
    
    def evaluate(self):
        probs = self.model.forward(self.X_val)
        preds = np.argmax(probs, axis=1)
        return np.mean(preds == self.y_val)

def hyperparameter_search(X_train, X_val, y_train, y_val, activate = 'relu'):
    best_acc = 0
    best_params = {}
    
    for hidden_size in [128, 256, 512]:
        for lr in [0.001, 0.0005, 0.0001]:
            for reg_lambda in [0.01, 0.001, 0.0001]:
                print(f"\nTraining with hidden_size={hidden_size}, lr={lr}, reg={reg_lambda}")
                model = NeuralNetwork(3072, hidden_size, activate, reg_lambda)
                trainer = Trainer(model, X_train, y_train, X_val, y_val,
                                 learning_rate=lr, lr_decay=0.95, decay_interval=10) # 在此处调节学习率衰减参数与其周期！
                trainer.train(num_epochs=100, batch_size=64)
                
                if trainer.best_val_acc > best_acc:
                    best_acc = trainer.best_val_acc
                    best_params = {
                        'hidden_size': hidden_size,
                        'lr': lr,
                        'reg_lambda': reg_lambda,
                        'model': model,
                        'trainer': trainer
                    }
    return best_params

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_cifar10()

    print("Starting hyperparameter search...")
    best = hyperparameter_search(X_train, X_val, y_train, y_val, activate='relu') # 在此处修改激活函数

    print("\nBest parameters:")
    print(f"Hidden size: {best['hidden_size']}")
    print(f"Learning rate: {best['lr']}")
    print(f"Regularization: {best['reg_lambda']}")
    print(f"Validation Accuracy: {best['model'].evaluate(X_val, y_val):.4f}")

    save_weights_to_txt(best['model'], r'D:\CV\ReLU_weights.txt')
    print("模型权重已保存")

    probs = best['model'].forward(X_test)
    preds = np.argmax(probs, axis=1)
    test_acc = np.mean(preds == y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    best_trainer = best['trainer']

    # loss曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(best_trainer.train_losses, label='Train Loss')
    plt.plot(best_trainer.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(best_trainer.val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.show()

