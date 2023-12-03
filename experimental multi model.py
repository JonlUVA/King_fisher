import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate, Dense,BatchNormalization, Dropout
from tensorflow.keras.models import Model


import numpy as np 
import pandas as pd 

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Conv2D,Add,MaxPooling2D, Dense, BatchNormalization,Input,Flatten, Dropout,GlobalMaxPooling2D,Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random



#https://www.kaggle.com/code/chekoduadarsh/starters-guide-convolutional-xgboost/notebook
#https://www.kaggle.com/code/pedrolucasbritodes/bird-image-classification-cnn-89-accuracy/notebook
#https://stats.stackexchange.com/questions/404809/is-it-advisable-to-use-output-from-a-ml-model-as-a-feature-in-another-ml-model


#creating directory path

directory_path = 'train_organised'

#creating training and validation set
train_data, val_data  = tf.keras.utils.image_dataset_from_directory(
    directory = directory_path,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    image_size=(224, 224),
    shuffle=True,
    validation_split=0.2,
    subset="both",
    seed=42069,
    batch_size = 32
    )



# changing the data to a hot-endcoded (dont know why but XGboost doesnt work without it)
num_classes = len(train_data.class_names)
# Define a function to convert integer labels to one-hot encoded labels within the dataset pipeline
def preprocess_data(image, label):
    # Convert label to one-hot encoded format
    label = tf.one_hot(label, num_classes)
    return image, label

# Map the preprocess_data function to both training and validation datasets
train_data = train_data.map(preprocess_data)
val_data = val_data.map(preprocess_data)




# Load the InceptionV3 model with pre-trained weights
inception_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
# Freeze the InceptionV3 model layers
for layer in inception_model.layers:
    layer.trainable = False

# Load the ResNet50 model with pre-trained weights
resnet_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg',
    classes=200,
    weights='imagenet'
)
# Freeze the ResNet50 model layers
for layer in resnet_model.layers:
    layer.trainable = False

# Get the output layers from both models
inception_output = inception_model.output
resnet_output = resnet_model.output

# Global Average Pooling for InceptionV3

inception_output = tf.keras.layers.Reshape((1, 1, -1))(inception_output)
inception_output = GlobalAveragePooling2D()(inception_output)

# Reshape the output of ResNet50 before Global Average Pooling
resnet_output = tf.keras.layers.Reshape((1, 1, -1))(resnet_output)
resnet_output = GlobalAveragePooling2D()(resnet_output)

# Concatenate the outputs of InceptionV3 and ResNet50
merged_output = Concatenate()([inception_output, resnet_output])

# Add Dense layers for classification
merged_output = Dense(512, activation='relu')(merged_output)
# Add Dense, BatchNormalization, and Dropout layers
merged_output = Dense(512, activation='relu')(merged_output)
merged_output = BatchNormalization()(merged_output)
merged_output = Dropout(0.3)(merged_output)

merged_output = Dense(256, activation='relu')(merged_output)
merged_output = BatchNormalization()(merged_output)
merged_output = Dropout(0.3)(merged_output)

# Output layer
output = Dense(200, activation='softmax')(merged_output)  # Replace 'num_classes' with your number of output classes



# Create a functional model using the inputs from both models and the concatenated output
model = Model(inputs=[inception_model.input, resnet_model.input], outputs=output)


# Assuming you have datasets 'train_data' and 'val_data' prepared
tf.data.Dataset.from_tensor_slices(train_data)
for x,y in train_data:
    type(x)

type(train_data)
# Fit the model with the prepared datasets
history = model.fit(
    [train_data, train_data],
    epochs=18,
    validation_data=([val_data, val_data]),
    batch_size=32
)


# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

# here is the data augmentation, this is what needs work i believe.
# I dont know how to see how much data it is making? 
# i think we need to really up the game here

datagen = ImageDataGenerator(
    #rotation_range=15,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range = 15,
    horizontal_flip = True,
    zoom_range = 0.20)
#look at aug from group chat 


# actually really confused here? idk if ive done this right lol
aug = datagen.flow_from_directory(directory_path)

model.fit_generator(aug,aug, steps_per_epoch= 122)


train_generator = datagen.flow_from_directory(
    directory=directory_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Assuming you're using categorical labels
    shuffle=True,
    seed=42
)

val_generator = datagen.flow_from_directory(
    directory=directory_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # No need to shuffle validation data
)

model.summary()
# this is supposed to help with over-fitting
#Setting the early_stop to avoid overfitting
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=3,
    min_delta=0.001,
    restore_best_weights=True,)



# idk if this is supposed to have the aug data or the train data
history = model.fit(
    x = train_data,
    #train_data,
    epochs=18,
    #steps_per_epoch=len(train_data),
    steps_per_epoch=len(train_data),
    validation_data=val_generator,
    validation_steps=int(0.25 * len(val_data)),
    callbacks=[early_stop],
)


def extract_labels(image, label):
    return label

# Apply the 'extract_labels' function to the dataset to extract only labels
labels_dataset = train_data.map(extract_labels)





print("Number of layers in the base model: ", len(model.layers))

# need to find out why this person has used 260 layers, might need to change this 

# Fine-tune from this layer onwards
fine_tune_at = 260


# Freeze all the layers before the `fine_tune_at` layer
for layer in model.layers[:fine_tune_at]:
    layer.trainable = False
   
base_learning_rate = 0.0001     
# Now that we will allow some layers to be unfreezed, it's better to decrease the learning rate to avoid dramatic changes in those 
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate/100)

# Compile the model again
model.compile(
    #loss = "sparse_categorical_crossentropy",
    loss="categorical_crossentropy",
    optimizer=adam_optimizer,
    metrics=["accuracy"]
)


early_stop = tf.keras.callbacks.EarlyStopping(
    patience=4,
    min_delta=0.001,
    restore_best_weights=True,)


fine_tune_epochs = 12
total_epochs =  18 + fine_tune_epochs

history_fine = model.fit(aug,
                        #train_data,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         steps_per_epoch=len(train_data),
                         validation_data=val_data,
                         validation_steps=int(0.25 * len(val_data)),
                         callbacks=[early_stop])



from keras.models import Model
layer_name='output-layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

intermediate_output = intermediate_layer_model.predict(train_data) 

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import tensorflow as tf

# Assuming intermediate_output is your feature data

# Define the XGBoost model

xgbmodel = XGBClassifier(objective='multi:softprob', num_class=200)
# Define the parameter grid for GridSearchCV
param_grid = {
    'learning_rate': [0.1, 0.0001],  # Learning rate
    'max_depth': [3, 5, 7, 25],        # Maximum depth of a tree
    'n_estimators': [100, 200, 300,400] # Number of boosting rounds
    # Add more parameters to tune based on your requirements
}

# Create GridSearchCV instance
# cv = k-folds 
grid_search = GridSearchCV(estimator=xgbmodel, param_grid=param_grid, cv=10, scoring='accuracy')

def extract_labels(image, label):
    return label

# Apply the 'extract_labels' function to the dataset to extract only labels
labels_dataset = train_data.map(extract_labels)

# Convert the labels dataset into a NumPy array for inspection or further processing
labels = list(labels_dataset.as_numpy_iterator())  # Convert labels dataset to a list

el = []
for x in labels:
    for y in x:
        el.append(np.where(y == 1)[0][0])

# Convert intermediate_output to a NumPy array or pandas DataFrame (assuming it's your feature data)
# Ensure that intermediate_output and el have the same number of samples
# Your data preprocessing steps might vary, so adjust accordingly

labels_dataset2 = val_data.map(extract_labels)
val_intermediate_output = intermediate_layer_model.predict(val_data) 
# Convert the labels dataset into a NumPy array for inspection or further processing
labels2 = list(labels_dataset2.as_numpy_iterator())  # Convert labels dataset to a list

intermediate_output_np = np.array(intermediate_output)  # Convert to NumPy array if it's not already

# Perform grid search for hyperparameter tuning
#grid_search.fit(intermediate_output_np, el)

# Get the best parameters
#best_params = grid_search.best_params_
#print("Best Parameters:", best_params)

# You can use these best parameters to initialize your XGBoost model
#best_xgbmodel = XGBClassifier(objective='multi:softprob', num_class=200, **best_params)
# Old model


# Define the XGBoost classifier with enhanced parameters for boosting
best_xgbmodel = XGBClassifier(
    objective='multi:softprob',
    num_class=200,
    learning_rate=0.1,  # Adjusted learning rate for better performance
    max_depth=8,  # Increased max depth for more complex trees
    n_estimators=500,  # Increased number of estimators for better learning
    booster='gbtree',  # Using gradient boosted trees
    subsample=0.8,  # Utilize 80% of the samples for each boosting iteration
    colsample_bytree=0.8,  # Use 80% of features for constructing each tree
    gamma=0.1,  # Regularization parameter to control tree complexity
    min_child_weight=1,  # Minimum sum of instance weight (hessian) needed in a child
    reg_alpha=0.001,  # L1 regularization term on weights
    reg_lambda=1,  # L2 regularization term on weights
    scale_pos_weight=1,  # Control the balance of positive and negative weights
    random_state=42  # Set a seed for reproducibility
)


# Train the XGBoost model with the best parameters
best_xgbmodel.fit(intermediate_output_np, el)

# Evaluate the model on the validation data (similar preprocessing for validation data is required)
# Assuming val_intermediate_output represents validation feature data
val_intermediate_output = intermediate_layer_model.predict(val_data) 
val_intermediate_output_np = np.array(val_intermediate_output)  # Convert to NumPy array if needed

el2 = list()
for x in labels2:
    for y in x:
        el2.append(np.where(y == 1)[0][0])
# Perform predictions on the validation data
predictions = best_xgbmodel.predict(val_intermediate_output_np)




# Evaluate accuracy or other metrics as needed
accuracy = np.mean(predictions == el2)
print("Validation accuracy with best parameters:", accuracy)


history_df = pd.DataFrame(history.history)
# Start the plot at epoch 1
history_df.loc[1:, ['loss', 'val_loss']].plot()
history_df.loc[1:, ['accuracy', 'val_accuracy']].plot()

print(("Best Validation Loss: {:1f}" +\
      "\nBest Validation Accuracy: {:1f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_accuracy'].max()))

history_df_ft = pd.DataFrame(history_fine.history)
# Start the plot at epoch 1
history_df_ft.loc[1:, ['loss', 'val_loss']].plot()
history_df_ft.loc[1:, ['accuracy', 'val_accuracy']].plot()

print(("Best Validation Loss: {:1f}" +\
      "\nBest Validation Accuracy: {:1f}")\
      .format(history_df_ft['val_loss'].min(), 
              history_df_ft['val_accuracy'].max()))
    
# Following the TensorFlow documentation, we will save the metrics in some variables and then plot it using plt. 

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']


# Plot the metrics

num_epochs = len(history_df)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.7, 1])
plt.plot([num_epochs-1,num_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([num_epochs-1,num_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()




# Print model summary
model.summary()
