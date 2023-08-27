import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  ## 读取图片
img = np.array(Image.open("lena.png")) ##将图像读入并转换成numpy矩阵
print(img.shape) ## 打印图像矩阵的大小
## 显示原始输入图像
plt.figure(figsize=(8,8))
plt.imshow(img)
plt.title('Original Input Image')
plt.axis('off')
plt.show() ## 若没有这一步，将不会显示
fig = plt.figure("3-D Plot of the first row pixels")
ax = plt.axes(projection='3d')
for px in img[0]:
    ax.scatter3D(*px, c = np.array([px])/255)
ax.set_xlabel("Red")
ax.set_ylabel("Green")
ax.set_zlabel("Blue")
ax.dist = 11
plt.tight_layout()
plt.show()
## 步骤一，随机初始化k个聚类中心
def initialization(img, k):
    """
    Parameters:
        img - 输入图像
        k - 初始聚类中心数量
    Returns:
        init_mu: (k,3) - 随机初始的k个聚类中心
    Tips:
    可以从原图中随机采样k个像素点作为聚类中心;
    """
    h, w, _ = img.shape
    np.random.seed(42)
    sample_idx = np.random.choice(h * w, size=k, replace=False)
    init_mu = img.reshape((-1, 3))[sample_idx].astype(np.float32)
    return init_mu

## 步骤二，完成k-means主体算法，迭代的为每个样本点分配聚类中心，更新聚类中心为其样本点的均值
def kmeans(img,k,iter_num):
    """
    Parameters:
        img - 输入图像
        k - 初始聚类中心数量
        iter_num - k-means迭代次数
    Returns:
        mu: (k,3) - 完成迭代后聚类中心
    Tips:
    参考教学ppt的k-means算法流程，有一点需要注意，对于这种样本点特别多的k-means聚类，很可能算法收敛到所有聚类簇都无须更新的时间较久。为此我们设置了一定的迭代轮数，让算法在完成这么多次迭代后自动结束;
    """
    h, w = img.shape[:2]
    mu = initialization(img,k)
    for iter in range(iter_num):
        pass
    return mu
## 步骤三，将图像中的每个像素点替换为与其所属的聚类中心的值，进行压缩
def compression(img,k,mu):
    """
    Parameters:
        img - 输入图像
        k - 初始聚类中心数量
        mu: (k,3) - k-means聚类获取的k个聚类中心
    Returns:
        new_img - 压缩后的图像
    Tips:
        将原图的每个像素点和k个聚类中心算距离，将其值替换为离它最近的聚类中心的均值;
    """
    h,w = img.shape[:2]
    img = img.reshape((-1,3))
    new_img = np.zeros_like(img)
    # 计算每个像素点和k个聚类中心的距离
    dist = np.linalg.norm(img[:,np.newaxis,:]-mu,axis=2)
    # 找到距离最近的聚类中心索引
    labels = np.argmin(dist,axis=1)
    # 将像素点的值替换为其所属的聚类中心的均值
    for i in range(k):
        new_img[labels==i] = mu[i]
    return new_img.reshape((h,w,3)).astype(np.uint8)

k_list = [2,4,8,16,32]
iter_num = 30
plt.figure(figsize=(10,10*len(k_list)),dpi=200)
for idx,k in enumerate(k_list):
    plt.subplot(1,len(k_list),idx+1)
    mu = kmeans(img,k,iter_num)
    new_img = compression(img,k,mu)
    plt.imshow(new_img)
    plt.title('k={}'.format(k))
    plt.axis('off')
    plt.tight_layout()
plt.show()