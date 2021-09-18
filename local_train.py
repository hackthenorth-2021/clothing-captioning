from model import get_model
from generator import get_generator

train_gen = get_generator("C:/Users/Brand/Desktop/data/train", "train.json")
model = get_model()
model.fit(x=train_gen, epochs=10)