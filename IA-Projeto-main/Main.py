import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os


# Caminho para o diretório de dados
data_dir = 'test'

# Definir geradores de dados com e sem data augmentation
datagen_no_aug = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
datagen_aug = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Carregar dados de treino e validação (sem data augmentation)
train_generator_no_aug = datagen_no_aug.flow_from_directory(
    data_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

validation_generator_no_aug = datagen_no_aug.flow_from_directory(
    data_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

# Carregar dados de treino e validação (com data augmentation)
train_generator_aug = datagen_aug.flow_from_directory(
    data_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

validation_generator_aug = datagen_no_aug.flow_from_directory(
    data_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)


def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model


# Treinamento sem data augmentation
model_s_no_aug = create_model()
model_s_no_aug.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

history_no_aug = model_s_no_aug.fit(
    train_generator_no_aug,
    epochs=10,
    validation_data=validation_generator_no_aug
)

# Treinamento com data augmentation
model_s_aug = create_model()
model_s_aug.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

history_aug = model_s_aug.fit(
    train_generator_aug,
    epochs=10,
    validation_data=validation_generator_aug
)

# Avaliar o modelo sem data augmentation
test_loss_no_aug, test_acc_no_aug = model_s_no_aug.evaluate(validation_generator_no_aug, verbose=2)
print(f'Test accuracy without data augmentation: {test_acc_no_aug}')

# Avaliar o modelo com data augmentation
test_loss_aug, test_acc_aug = model_s_aug.evaluate(validation_generator_aug, verbose=2)
print(f'Test accuracy with data augmentation: {test_acc_aug}')


# Função para plotar gráficos de acurácia e perda
def plot_history(histories, titles):
    for i, history in enumerate(histories):
        plt.figure(figsize=(12, 4))
        # Acurácia
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Treinamento')
        plt.plot(history.history['val_accuracy'], label='Validação')
        plt.title(f'Acurácia - {titles[i]}')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend()

        # Perda
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Treinamento')
        plt.plot(history.history['val_loss'], label='Validação')
        plt.title(f'Perda - {titles[i]}')
        plt.xlabel('Época')
        plt.ylabel('Perda')
        plt.legend()

        plt.show()

# Plotar os gráficos
plot_history([history_no_aug, history_aug], ['Sem Data Augmentation', 'Com Data Augmentation'])

# Salvar o modelo
model_s_no_aug.save('model_no_aug.h5')
model_s_aug.save('model_s_aug.h5')



