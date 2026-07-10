import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import albumentations as A #type: ignore
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0 #type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Input, GlobalAveragePooling2D

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CATEGORIES = ['benign', 'malignant', 'normal']

images = []
labels = []
data_path = os.path.join(path, 'Dataset_BUSI_with_GT')

for category in CATEGORIES:
    folder_path = os.path.join(data_path, category)
    for image_name in os.listdir(folder_path):
        if image_name.endswith('.png') and '_mask' not in image_name:
            image_path = os.path.join(folder_path, image_name)
            images.append(image_path)
            labels.append(category)
print(f'Numer of Images Samples: {len(images)}')
print(f'Numer of Labels Samples: {len(images)}')

data = {'Images': images, 'Labels': labels}
data = pd.DataFrame(data)
print(data.head()) #data.head()

label2idx = {cat: idx for idx, cat in enumerate(CATEGORIES)}
print(label2idx)

plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

ax = sns.countplot(data=data, x='Labels', palette='viridis')
plt.title("Count of Each Category")
plt.xlabel("Category")
plt.ylabel("Count")

plt.show()

\

train, test = train_test_split(data, test_size = 0.2, shuffle = True, stratify = data['Labels'],random_state = 42)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])
all_images = []
all_labels = []

for category in CATEGORIES:
    original_images = train[train['Labels'] == category]['Images'].values.tolist()
    augmented_images = []

    print(f'Current Category: {category}')

    for img_path in tqdm(original_images):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        augmented_images.append(image)

    while len(augmented_images) < 1000:
        img_path = random.choice(original_images)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        augmented = transform(image=image)
        aug_img = augmented['image']
        aug_img = cv2.resize(aug_img, (224, 224))
        augmented_images.append(aug_img)

    all_images.extend(augmented_images)
    all_labels.extend([label2idx[category]] * len(augmented_images))

all_images = np.array(all_images)
all_labels = np.array(all_labels)

print(f'Final dataset shape: {all_images.shape}')
print(f'Labels shape: {all_labels.shape}')


encoder = OneHotEncoder(sparse_output=False)
all_labels = encoder.fit_transform(all_labels.reshape((-1, 1)))


input_shape = (224, 224, 3)

def create_model(input_shape):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = create_model(input_shape)


history = model.fit(all_images, all_labels, epochs=50, batch_size=32, validation_split=0.2)


model.save('model_v3.keras')