# import tensorflow as tf
# from tensorflow.keras import layers, models

# def MobileNetV2(input_shape=(224, 224, 3), num_classes=1000):
#     input_img = layers.Input(shape=input_shape)

#     x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(input_img)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = inverted_res_block(x, 16, 1, 1)

#     x = inverted_res_block(x, 24, 2, 6)
#     x = inverted_res_block(x, 24, 1, 6)

#     x = inverted_res_block(x, 32, 2, 6)
#     x = inverted_res_block(x, 32, 1, 6)
#     x = inverted_res_block(x, 32, 1, 6)

#     x = inverted_res_block(x, 64, 2, 6)
#     x = inverted_res_block(x, 64, 1, 6)
#     x = inverted_res_block(x, 64, 1, 6)
#     x = inverted_res_block(x, 64, 1, 6)

#     x = inverted_res_block(x, 96, 1, 6)
#     x = inverted_res_block(x, 96, 1, 6)
#     x = inverted_res_block(x, 96, 1, 6)

#     x = inverted_res_block(x, 160, 2, 6)
#     x = inverted_res_block(x, 160, 1, 6)
#     x = inverted_res_block(x, 160, 1, 6)

#     x = inverted_res_block(x, 320, 1, 6)

#     x = layers.Conv2D(1280, (1, 1), padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dense(num_classes, activation='softmax')(x)

#     model = models.Model(inputs=input_img, outputs=x)
    
#     return model

# def inverted_res_block(x, filters, stride, expansion):
#     input_shape = x.shape[-1]
#     x = layers.Conv2D(input_shape * expansion, (1, 1), padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.DepthwiseConv2D((3, 3), strides=(stride, stride), padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.Conv2D(filters, (1, 1), padding='same')(x)
#     x = layers.BatchNormalization()(x)

#     if stride == 1 and input_shape == filters:
#         x = layers.Add()([x, input_img])

#     return x

import tensorflow as tf
from tensorflow.keras import layers, models

def inverted_res_block(x, filters, stride, expansion):
    # Store input for residual connection
    shortcut = x
    
    # Expansion phase
    x = layers.Conv2D(expansion * int(x.shape[-1]), 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    
    # Depthwise Convolution
    x = layers.DepthwiseConv2D(3, stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    
    # Projection
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    # Residual connection if stride=1 and input/output channels match
    if stride == 1 and shortcut.shape[-1] == filters:
        return layers.Add()([shortcut, x])
    return x

def MobileNetV2(input_shape=(224, 224, 3), num_classes=1000):
    inputs = layers.Input(shape=input_shape)
    
    # First layer
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    
    # Inverted Residual Blocks
    x = inverted_res_block(x, 16, 1, 1)
    
    x = inverted_res_block(x, 24, 2, 6)
    x = inverted_res_block(x, 24, 1, 6)
    
    x = inverted_res_block(x, 32, 2, 6)
    x = inverted_res_block(x, 32, 1, 6)
    x = inverted_res_block(x, 32, 1, 6)
    
    x = inverted_res_block(x, 64, 2, 6)
    x = inverted_res_block(x, 64, 1, 6)
    x = inverted_res_block(x, 64, 1, 6)
    x = inverted_res_block(x, 64, 1, 6)
    
    x = inverted_res_block(x, 96, 1, 6)
    x = inverted_res_block(x, 96, 1, 6)
    x = inverted_res_block(x, 96, 1, 6)
    
    x = inverted_res_block(x, 160, 2, 6)
    x = inverted_res_block(x, 160, 1, 6)
    x = inverted_res_block(x, 160, 1, 6)
    
    x = inverted_res_block(x, 320, 1, 6)
    
    # Last layer
    x = layers.Conv2D(1280, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    
    # Global pooling and prediction
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, x, name='mobilenetv2')
    return model