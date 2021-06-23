import PIL.Image as image
from sklearn import preprocessing
import numpy as np
from sklearn.cluster import KMeans
from skimage import color

# 加载图像，并对数据进行规范化
def load_data(filepath):
    f = open(filepath, 'rb')
    data = []
    img = image.open(f)
    width, height = img.size
    for x in range(width):
        for y in range(height):
            c1, c2, c3 = img.getpixel((x, y))
            data.append([c1, c2, c3])
    f.close()
    mm = preprocessing.MinMaxScaler()
    data = mm.fit_transform(data)
    return np.mat(data), width, height

img, width, height = load_data('C:/Users/O-O RT/Desktop/数据分析相关书籍/项目/使用K-Means对图像进行分割/蓝鲸.jpg')
kmeans = KMeans(n_clusters=8)
kmeans.fit(img)
label = kmeans.predict(img)
label = label.reshape([width, height])
# 将聚类标识矩阵转化为不同颜色的矩阵
label_color = (color.label2rgb(label)*255).astype(np.uint8)
label_color = label_color.transpose(1, 0, 2)
images = image.fromarray(label_color)
images.save('C:/Users/O-O RT/Desktop/数据分析相关书籍/项目/使用K-Means对图像进行分割/蓝鲸3.jpg')

