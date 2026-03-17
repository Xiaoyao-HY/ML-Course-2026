import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from sklearn.neighbors import KNeighborsClassifier

print("="*60)
print("实验一：参数估计（MLE与MAP）")
print("="*60)

data = torch.tensor([1.,1.,0.,1.,0.])
p_mle = torch.mean(data)
alpha = 2
beta = 2
p_map = (torch.sum(data)+alpha-1)/(len(data)+alpha+beta-2)
print("MLE =",p_mle.item())
print("MAP =",p_map.item())

print("\n" + "="*60)
print("可视化分析：MLE vs MAP")
print("="*60)

data = np.array([1,1,0,1,0])
N = len(data)
sum_x = np.sum(data)
p = np.linspace(0,1,100)
likelihood = p**sum_x*(1-p)**(N-sum_x)
prior = beta_dist.pdf(p,2,2)
posterior = p**(sum_x+1)*(1-p)**(N-sum_x+1)
p_mle = sum_x/N
p_map = (sum_x+1)/(N+2)

plt.figure(figsize=(10,6))
plt.plot(p,likelihood,label="Likelihood")
plt.plot(p,prior,label="Prior")
plt.plot(p,posterior,label="Posterior")
plt.axvline(p_mle,color='r',linestyle='--',label="MLE")
plt.axvline(p_map,color='g',linestyle='--',label="MAP")
plt.legend()
plt.title("MLE vs MAP - Bernoulli Parameter Estimation")
plt.grid()
plt.savefig("mle_vs_map.png", dpi=150, bbox_inches='tight')
plt.close()
print("图像已保存: mle_vs_map.png")

print("\n" + "="*60)
print("实验二：KNN分类")
print("="*60)

X = np.array([[1,2],[2,3],[3,3],[6,5],[7,7],[8,6]])
y = np.array([0,0,0,1,1,1])

xx,yy = np.meshgrid(np.linspace(0,10,200),
                    np.linspace(0,10,200))

fig, axes = plt.subplots(1, 3, figsize=(15,5))

for idx, k in enumerate([1,3,5]):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X,y)

    Z = model.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)

    axes[idx].contourf(xx,yy,Z,alpha=0.3)
    axes[idx].scatter(X[:,0],X[:,1],c=y, edgecolors='k', s=100)
    axes[idx].set_title(f"K = {k}")
    axes[idx].set_xlim(0,10)
    axes[idx].set_ylim(0,10)
    axes[idx].set_xlabel('X1')
    axes[idx].set_ylabel('X2')

plt.suptitle("KNN Classification Boundaries", fontsize=14)
plt.tight_layout()
plt.savefig("knn_classification.png", dpi=150, bbox_inches='tight')
plt.close()
print("图像已保存: knn_classification.png")

print("\n" + "="*60)
print("实验三：梯度下降法")
print("="*60)

print("\n--- 梯度下降优化 ---")
x = torch.tensor([5.0],requires_grad=True)
lr = 0.1

for i in range(20):
    y = x**2+2*x+1
    y.backward()
    with torch.no_grad():
        x -= lr*x.grad
    x.grad.zero_()

print(f"最优 x = {x.item()}, 最小 y = {y.item()}")

print("\n--- 损失曲线 ---")
loss = []
x = torch.tensor([5.0],requires_grad=True)

for i in range(20):
    y = x**2+2*x+1
    loss.append(y.item())
    y.backward()
    with torch.no_grad():
        x -= 0.1*x.grad
    x.grad.zero_()

plt.figure(figsize=(10,6))
plt.plot(loss,marker='o')
plt.title("Loss Curve - Gradient Descent")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.savefig("loss_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("图像已保存: loss_curve.png")

print("\n--- 优化路径可视化 ---")
x_vals = np.linspace(-5,5,100)
y_vals = x_vals**2+2*x_vals+1

x_path = []
x = torch.tensor([5.0],requires_grad=True)

for i in range(10):
    y = x**2+2*x+1
    x_path.append(x.item())
    y.backward()
    with torch.no_grad():
        x -= 0.3*x.grad
    x.grad.zero_()

plt.figure(figsize=(10,6))
plt.plot(x_vals,y_vals)
plt.scatter(x_path,[xx**2+2*xx+1 for xx in x_path],color='r', zorder=5)
for i, xv in enumerate(x_path):
    plt.annotate(str(i), (xv, xv**2+2*xv+1), textcoords="offset points", xytext=(0,10), ha='center')
plt.title("Gradient Descent Path")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.savefig("gradient_path.png", dpi=150, bbox_inches='tight')
plt.close()
print("图像已保存: gradient_path.png")

print("\n" + "="*60)
print("实验完成！所有图像已保存。")
print("="*60)
