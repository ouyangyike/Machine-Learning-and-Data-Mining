import numpy as np
import utils
import matplotlib.pyplot as plt
from run_knn import run_knn

rate = np.zeros((5,1))
for k in [1,3,5,7,9]:
    train_data,train_label = utils.load_train()
    valid_data,valid_label = utils.load_valid()
    valid_label_temp = run_knn(k,train_data,train_label,valid_data)
    j=0
    for i in range(0,49):
        if valid_label[i]==valid_label_temp[i]:
            j+=1
    rate[(k-1)/2]=(j*1.0)/50
k = [1,3,5,7,9]
plt.plot(k,rate)
plt.title('Plot of classification rate vs. k')
plt.xlabel('k')
plt.ylabel('classification rate')
plt.xlim(0,10)
plt.ylim(0.90,1.00)
plt.show()
