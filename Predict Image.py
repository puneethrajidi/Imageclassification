import pickle
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
model=pickle.load(open('imgclassification_model.p','rb'))
flat_data=[]
Types=['bikes','cars']
url=input('Enter url')
img=imread(url)
img_resized=resize(img,(150,150,3))
flat_data.append(img_resized.flatten())
flat_data=np.array(flat_data)
print(img.shape)
plt.imshow(img_resized)
y_out=model.predict(flat_data)
print(Types[y_out[0]])


