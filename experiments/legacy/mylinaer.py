import torch

import matplotlib.pyplot as plt
import random

def create_data(w, b, data_num):
    x = torch.normal(0,1,(data_num, len(w)))
    y = torch.matmul(x, w) + b

    noise = torch.normal(0, 0.01, y.shape)
    y = y + noise
    return x, y

num = 500
true_w = torch.tensor([8.1, 2, 2, 4])
true_b = torch.tensor([1.1])
X, Y = create_data(true_w, true_b, num)




plt.scatter(X[:,0].detach().numpy(), Y.detach().numpy(), 1)
plt.show()

def data_provider(data, label, batch_size):
    length = len(label)
    indices = list(range(length))
    random.shuffle(indices)
    for each in range(0, length, batch_size):
        get_indices = indices[each : each+batch_size]      # 想一下如果超了怎么半。
        get_data = data[get_indices]
        get_label = label[get_indices]
        yield get_data, get_label

batch_size = 15

for batch_x, batch_y in data_provider(X, Y, 15):
    print(batch_x, batch_y)
    break

def Fun(x, w, b):
    pred_y = torch.matmul(x, w) + b
    return pred_y

def maeloss(y, pred_y):
    return abs(y-pred_y).sum()/len(y)

def sgd(params, lr):
    with torch.no_grad():
        for param in params:
            param -= param.grad* lr
            # param.grad.zero_()
            param.grad= torch.zeros(param.shape)



lr = 0.001
w_0 = torch.normal(1, 0.01, true_w.shape, requires_grad=True)
b_0 = torch.rand(1, requires_grad=True)
print(w_0, b_0)
epochs = 30

for epoch in range(epochs):
    data_loss = 0
    for batch_x, batch_y in data_provider(X, Y,batch_size):
        pred = Fun(batch_x, w_0, b_0)
        loss = maeloss(batch_y, pred)
        loss.backward()
        sgd([w_0, b_0], 0.03)
        data_loss += loss
    print("epoch %03d : loss:%.6f" % (epoch, data_loss/(num)))
print("原来函数值",true_w, true_b)
print("预测函数值", w_0, b_0)

idx = 0
plt.plot(X[:, idx].detach().numpy(), X[:, idx].detach().numpy()*w_0[idx].detach().numpy()+b_0.detach().numpy(), label="pre")
plt.scatter(X[:, idx].detach().numpy(), Fun(X, w_0, b_0).detach().numpy(), 1)
plt.show()
