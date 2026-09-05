import torch
import torch.nn as nn




y = torch.tensor([11.7,23,20],dtype=float)

softmax = nn.Softmax(dim=-1)
CEloss = nn.CrossEntropyLoss()

y_hat = torch.tensor([1, 0, 0],dtype=float)

print(softmax(y))
#
# y= y.unsqueeze(0)
# y_hat = y_hat.unsqueeze(0)
# print(softmax(y))
# #
# print(CEloss(y, y_hat))
# #
# y = softmax(y)
# print(CEloss(y, y_hat))