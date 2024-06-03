import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os


# Caminho para o modelo salvo
model_path = 'model_no_aug.h5'

# Carregar o modelo
model = tf.keras.models.load_model(model_path)


# Caminho para a nova imagem
image_path = '..\\test\\000_airplane\\twinjet_s_000413.png'

# Carregar a imagem
img = image.load_img(image_path, target_size=(32, 32))

# Converter a imagem para um array de numpy
img_array = image.img_to_array(img)

# Expandir as dimensões da imagem para que corresponda ao formato esperado pelo modelo
img_array = np.expand_dims(img_array, axis=0)

# Normalizar a imagem
img_array /= 255.0


# Fazer a previsão
# Define the variable "train_generator_no_aug"
train_generator_no_aug = ...

predictions = model.predict(img_array)

# Obter a classe com a maior probabilidade
predicted_class = np.argmax(predictions[0])

# Mapear o índice da classe para o nome da classe
# Certifique-se de que você tenha o mapeamento de índices de classe de quando você treinou o modelo
class_indices = {v: k for k, v in train_generator_no_aug.class_indices.items()}
predicted_class_name = class_indices[predicted_class]

print(f'Predicted class: {predicted_class_name}')


# Carregar o gerador de dados de teste
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=(32, 32),
    batch_size=32,
    class_mode='sparse'
)

# Avaliar o modelo
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f'Test accuracy: {test_acc}')
