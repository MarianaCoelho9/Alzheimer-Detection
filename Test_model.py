import numpy as np
import os
import nibabel as nib
import pandas as pd
import tensorflow as tf
from tensorflow import keras

#In[1]:

# - import test arrays (select path)
X_test = np.load("/content/drive/MyDrive/TESE/Dados/teste_image_emci.npy")
y_test = np.load("/content/drive/MyDrive/TESE/Dados/teste_label_emci.npy")

print ("Dados de teste:", len(X_test))


#In[2]:
test_image = X_test
test_label = y_test
print(test_image.shape)


#In[3]:
batch_size = 3
test_loader = tf.data.Dataset.from_tensor_slices((X_test, y_test))

def test_preprocessing(volume, label):
    """Process test data by only adding a channel and covert to rgb."""
    volume = tf.expand_dims(volume, axis=3)
    volume=tf.image.grayscale_to_rgb(volume)
    return volume, label

# - do not shuffle the test dataset 
test_dataset = (
    test_loader.map(test_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

#In[4]:


# - import the trained model
final_model = keras.models.load_model('/content/drive/MyDrive/TESE/Final_class/seresnet152_final_emci.h5')

final_model.evaluate(test_dataset)
final_predict = final_model.predict(test_dataset)

print(final_predict)

predict = (final_predict > 0.5).astype('int')

print(predict)
print(test_label)

# - create an array with the desired labels 
labels_multi=np.array(["CN", "EMCI", "LMCI", "AD"])
labels_bi=np.array(["CN", "AD"])

#In[5]:
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

#MULTICLASS
ConfusionMatrixDisplay.from_predictions(test_label.argmax(axis=1), predict.argmax(axis=1), display_labels=labels_multi, cmap=plt.cm.Blues)

#BINARYCLASS
ConfusionMatrixDisplay.from_predictions(test_label, predict,display_labels=labels_bi, cmap=plt.cm.Blues)



#In[6]:
import matplotlib.pyplot as plt

def plot_roc_curve(true_y, y_prob):

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr, label='Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(test_label, final_predict)
print(f'Model AUC score: {roc_auc_score(test_label, final_predict)}')




#In[7]:

history = pd.read_csv('/content/drive/MyDrive/TESE/Final_class/seresnet152_lmci___.csv')
print(history)


#In[8]:

# - Loss plot
plt.plot(history['loss'], label= 'Training Loss')
plt.plot(history['val_loss'], label= 'Validation Loss')
plt.legend()
plt.show()

# - Accuracy plot
plt.plot(history['binary_accuracy'], label= 'Training Accuracy')
plt.plot(history['val_binary_accuracy'], label= 'Validation Accuracy')
plt.legend()
plt.show()


#In[9]:

#MULTICLASS
print("Test Accuracy : {}".format(accuracy_score(test_label, predict)))
print("\nClassification Report :")
print(classification_report(test_label, predict, target_names=['CN', 'EMCI', 'LMCI','AD']))

#BINARYCLASS
print("Test Accuracy : {}".format(accuracy_score(test_label, predict)))
print("\nClassification Report :")
print(classification_report(test_label, predict, target_names=['CN', 'AD']))

