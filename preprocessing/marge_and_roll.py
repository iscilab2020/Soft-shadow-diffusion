import torch 
import scipy
import numpy as np
import glob

path ="./Objects3"
to = "./torch_object"

data = glob.glob(f"{path}/**.mat")

np.random.shuffle(data)
train = data[:-3000]
tests = data[-3000:]

datas = []
j = 1
for i in range(len(train)):

    f=scipy.io.loadmat(train[i])
    datas.append(f)

    if len(datas) == 10000:

        torch.save(datas, f"{to}/train_{j}.pt")
        datas=[]
        j+=1
        print("saved", j)


if len(datas) > 0:
    torch.save(datas, f"{to}/train_{j}.pt")
    print("saved", j)

datas = []
j = 1
for i in range(len(tests)):

    f=scipy.io.loadmat(tests[i])
    datas.append(f)

    if len(datas) == 10000:

        torch.save(datas, f"{to}/test_{j}.pt")
        datas=[]
        j+=1
        print("saved", j)


if len(datas) > 0:
    torch.save(datas, f"{to}/test_{j}.pt")
    print("saved", j)




