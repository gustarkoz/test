import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

df = pd.read_excel('training_data.xlsx')
#将第一列为零的数据删除并令存到新的dataframe中
#第一列为零的数据是测试数据，不参与训练
df_train = df[df['Y'] != 0]
X = df.iloc[:, 1:].values
Y = df.iloc[:, 0].values
df_test = df[df['Y'] == 0]
X_1 = df_test.iloc[:, 1:].values
scaler_X = MinMaxScaler((0, 1))
X_scaled = scaler_X.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = Net()

optimizer = optim.Adam(model.parameters(), lr=0.01)


criterion = nn.MSELoss()

X_train_tensor = torch.Tensor(X_train).float()
Y_train_tensor = torch.Tensor(Y_train).float().unsqueeze(1)
X_test_tensor = torch.Tensor(X_test).float()
Y_test_tensor = torch.Tensor(Y_test).float().unsqueeze(1)

num_epochs = 100
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_Y = Y_train_tensor[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    loss = criterion(outputs, Y_test_tensor)
    mse = loss.item()
    rmse = np.sqrt(mse)
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
#将测试数据放入模型中进行预测
#先将X_1进行归一化
X_1_scaled = scaler_X.transform(X_1)
#将X_1转换为tensor
X_test_tensor_1 = torch.Tensor(X_1_scaled ).float()
outputs_1 = model(X_test_tensor_1)
#将outputs_1和x_1合并
df_test.loc[:, 'Y'] = outputs_1.detach().numpy()
#将df_test保存到新的excel文件中
df_test.to_excel('test_data.xlsx', index=False)


