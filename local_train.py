from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import GlobalMaxPooling2D, Dropout
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam

image_shape = (512,512)

def get_model():
    inp = Input((image_shape[0], image_shape[1], 3))

    base_model = ResNet152(weights='imagenet',
                             include_top=False,
                             input_shape=(image_shape[0], image_shape[1], 3),
                             pooling='avg')(inp)

    x = Dense(2048, activation='relu')(base_model)
    x = Dropout(0.4)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(228, activation=sigmoid)(x)
    model = Model(inputs=inp, outputs=out)

    #insert metrics here
    adam = Adam(lr=0.001, clipnorm=0.001)
    binacc = BinaryAccuracy(name="binary_accuracy")

    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=[binacc])
    model.summary()

    return model

get_model()