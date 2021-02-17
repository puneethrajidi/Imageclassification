import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from PIL import Image

output=[]
images=[]
flattered_data=[]
DIR='C:\\Users\\punee\\Desktop\\Images'
Types=['bikes','cars']
for type in Types:
  Label=Types.index(type)
  path=os.path.join(DIR,type)
  for img in os.listdir(path):
    im_array=imread(os.path.join(path,img))
    img_resized=resize(im_array,(150,150,3))
    flattered_data.append(img_resized.flatten())
    images.append(img_resized)
    output.append(Label)
    
flattered_data=np.array(flattered_data)
output=np.array(output)
images=np.array(images)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(flattered_data,output,test_size=0.7,random_state=109)
from sklearn.model_selection import GridSearchCV
from sklearn import svm
param_grid={'C':[0.1,1,10,100,1000],'kernel':['linear','rbf','poly'],'gamma':[1,0.1,0.01,0.001,0.0001]}
svc=svm.SVC(probability=True)
var=GridSearchCV(svc,param_grid)
var.fit(x_train,y_train)
y_pred=var.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))



