
# coding: utf-8

# # AlexNet(CNN)을 이용한 흉부 엑스레이 이상 탐지
# ## 1. 라이브러리 호출
# ## 2. 데이터 불러오기
# ## 3. 데이터 샘플링
# ## 4. 데이터 전처리
# ## 5. 모델링
# ## 6. 모델 평가

# ## 1. 라이브러리 호출

# In[108]:


#Python 3.6.7 버전에서 작성 되었습니다.
#자료는 https://www.kaggle.com/nih-chest-xrays/data 에서 images_001.zip ~  images_005.zip까지
#다운로드 한 뒤에 샘플링을 거쳐 구성되었습니다.

#기본 라이브러리 묶음
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

#이미지 핸들링 라이브러리
from PIL import Image
#모델 저장용 라이브러리
import joblib


# In[ ]:


#OHE 라이브러리
from keras.utils import np_utils
#데이터셋 분리 
from sklearn.model_selection import train_test_split
#모델 평가
from sklearn.metrics import classification_report


# In[ ]:


#딥러닝 모델 구성 관련 라이브러리 묶음
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


# ## 2. 데이터 불러오기
# 
# 샘플로 받은 이미지 파일은 레이블이 없기 때문에 전체 이미지에 대한 메타 데이터를 갖고 있는 파일을 참고하여 샘플 데이터의 레이블을 찾아준다.

# In[26]:


#X-Ray 사진들에 메타 데이터를 정리한 파일 읽어오기
total_img_info=pd.read_csv("./datasets/Data_Entry_2017(1).csv")


# In[27]:


#이미지 파일의 파일명을 전부 불러온다.
file_list=os.listdir("./datasets/img/")


# In[28]:


#모든 이미지 데이터에 대한 메타 데이터 중에서 이미지 파일의 이름만 가져온다.
total_list=total_img_info["Image Index"].values.tolist()


# In[38]:


#모든 메타 데이터 중 현재 다운 받은 파일(images_001~005.zip)의 이름만 골라낸다.
dataset=set(total_list).intersection(set(file_list))


# In[39]:


#리스트 형태로 바꿔준다.
dataset_list=list(dataset_list)


# In[41]:


#샘플 데이터의 이름과 레이블로 구성된 데이터 프레임을 만들어준다.
df=total_img_info[total_img_info["Image Index"].isin(dataset_list)].copy()


# In[43]:


#쉬운 컬럼명으로 변경한다.
df.columns=["Image","Label"]


# In[44]:


#나중에 사용하기 위해 따로 저장한다.
df.to_csv("dataset.csv", index=False)


# In[45]:


#레이블의 분포를 살펴본다.
df.Label.value_counts()


# In[61]:


#상위 5개의 레이블만 가져온다.
df_pre=df[(df["Label"]=="No Finding")|(df["Label"]=="Infiltration")|(df["Label"]=="Atelectasis")|(df["Label"]=="Effusion")|(df["Label"]=="Nodule")].copy()


# ## 3. 데이터 샘플링

# In[62]:


#층화 추출을 위한 함수
def stratified_sample_df(df, col, n_samples):
    """
    파이썬에는 층화 추출 함수가 없어 직접 만든 함수
    레이블 칼럼을 입력하면 레이블 종류별 값을 기준으로
    최소값에 해당하는 값만큼 각각의 레이블을 추출한다.
    """
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


# In[63]:


#층화 추출
df_stry=stratified_sample_df(df_pre, "Label", 2000)


# In[64]:


df_stry["Label"].value_counts()


# ## 4. 데이터 전처리

# In[76]:


#파일 경로와 레이블 값만 따로 담아준다.
file_path=df_stry["Image"].values
labels=df_stry["Label"].values


# In[77]:


labels[labels=="No Finding"]=0
labels[labels=="Nodule"]=1
labels[labels=="Atelectasis"]=2
labels[labels=="Effusion"]=3
labels[labels=="Infiltration"]=4


# In[79]:


labels=labels.astype("int32")


# In[89]:


img_list=[]

for i in range(len(df_stry)):
    #이미지를 불러온다.
    img_read =Image.open("./datasets/img/"+file_path[i])
    gray=img_read.convert(mode="RGB")
    print(i)
    #1024 이미지를 224 이미지로 축소한다.
    img_resize = gray.resize((224, 224), Image.ANTIALIAS)
    #이미지를 배열로 만들어준다.
    img_array = np.asarray(img_resize, dtype="float32")
    img_list.append(img_array)


# In[101]:


#이미지 확인
imgplot = plt.imshow(img_list[0]/255)
plt.show();


# In[117]:


#레이블을 더미화 시켜준다.
dummy_y = np_utils.to_categorical(labels)


# In[118]:


#훈련과 테스트 셋으로 데이터를 나눠준다.
x_train, x_test, y_train, y_test = train_test_split(img_list, dummy_y, test_size=0.3, random_state=523, stratify=labels)


# In[119]:


#0과 1사이의 값으로 만들어준다.
x_train = np.array(x_train).reshape(len(x_train), 224*224*3).astype('float32') / 255.0
x_test = np.array(x_test).reshape(len(x_test), 224*224*3).astype('float32') / 255.0


# In[120]:


#딥러닝 모델에 맞게 차원을 수정해준다.
x_tr=x_train.reshape(-1, 224,224,3)
x_te=x_test.reshape(-1, 224,224,3)


# ## 5. 모델링

# In[149]:


# AlexNet 모델
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*1,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(5))
model.add(Activation('softmax'))


# In[150]:


model.summary()


# In[151]:


# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[152]:


# 모델 학습
hist = model.fit(x_tr, y_train, epochs=1000, batch_size=32, validation_split=.2)


# In[153]:


# 모델 학습과정 살펴보기
get_ipython().run_line_magic('matplotlib', 'inline')

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')


acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')


loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()


# ## 6. 모델 평가

# In[154]:


#모델 평가하기
loss_and_metrics = model.evaluate(x_te, y_test, batch_size=32)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)


# In[155]:


#모델을 이용하여 흉부사진을 분류한다.
y_pred=model.predict_classes(x_te)


# In[156]:


#결과 값을 OHE 해야 비교해볼 수 있다.
dummy_pred = np_utils.to_categorical(y_pred)


# In[157]:


#F1 스코어를 살펴본다.
print(classification_report(y_test, dummy_pred ))


# In[158]:


# 모델을 저장한다.
filename = 'alexnet_cnn.sav'
joblib.dump(model, filename)


# In[148]:


#혹시 반복하게 된다면 가중치 리셋을 위한 코드
#model.reset_states()

