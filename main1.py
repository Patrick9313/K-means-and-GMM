import numpy as np
import matplotlib.pyplot as plt
data=[-0.39, 0.12, 0.94, 1.67, 1.76, 2.44, 3.72, 4.28, 4.92, 5.53,
        0.06, 0.48, 1.01, 1.68, 1.80, 3.25, 4.12, 4.60, 5.28, 6.22]

y = np.array(data)

# 绘制直方图，一句话就可以，不要忘记把频率归一化
num_bins = 8
bin_width = 1
bins = [i*bin_width-1 for i in range(num_bins+1)]

# 绘制直方图，并归一化纵坐标
plt.hist(data, bins=bins, density=True)

# 添加坐标轴标签
plt.title("Histogram of data")
plt.xlabel("y")
plt.ylabel("p")

# 显示图形
plt.show()

# 参数初始化
N = len(y)
# 随机选取数据集中两个点作为初始化的均值
mu1 = 2.44
mu2 = 4.60
s1 = np.sum((y - np.mean(y)) ** 2) / N
s2 = s1
pi = 0.5

# EM迭代
count = 0  ## 当前迭代次数
gamma = 1. * np.zeros(N)  ## 隐变量
num_iters = 20  ## 迭代次数

ll = 1. * np.zeros(num_iters)  ## 存储对数似然值


# 定义一个高斯概率密度函数
def gaussian_pdf(v, mu, s):
    p = 1 / (np.sqrt(2*np.pi)*s) * np.exp(-0.5*((v-mu)/s)**2)
    return p



while count < num_iters:
        count = count + 1

        # E-step
        for i in range(N):
                p1 = gaussian_pdf(y[i], mu1, np.sqrt(s1))
                p2 = gaussian_pdf(y[i], mu2, np.sqrt(s2))
                gamma[i] = p1 * pi / (p1 * pi + (1 - pi) * p2)

        # M-step
        N1 = np.sum(gamma)
        N2 = N - N1
        mu1 = np.sum(gamma * y) / N1
        mu2 = np.sum((1 - gamma) * y) / N2
        s1 = np.sum(gamma * (y - mu1) ** 2) / N1
        s2 = np.sum((1 - gamma) * (y - mu2) ** 2) / N2
        pi = N1 / N


        # 计算每一次对数似然函数值
        ll[count-1] = np.sum(np.log(pi*gaussian_pdf(y, mu1, np.sqrt(s1)) + (1-pi)*gaussian_pdf(y, mu2, np.sqrt(s2))))


print(mu1, mu2, s1, s2, pi)
x = np.arange(-1,7,0.1)
p = pi*gaussian_pdf(x,mu1,s1) + (1-pi)*gaussian_pdf(x,mu2,s2)
plt.plot(x,p,'k',linewidth=4)
plt.xlabel("y")
plt.ylabel("p")
plt.title("Estimated Gaussian mixture density")
plt.figure()
plt.plot(ll,'ko-')
plt.xlabel("iters")
plt.ylabel("log-likelihood")
plt.show()