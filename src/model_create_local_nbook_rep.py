from load_model import load_saved_model
from model_create import *

train_generator = create_master_train_generator('../data/asl_alphabet_train')
print(train_generator[0][0][0].shape)
val_generator = create_val_generator('../data/asl_alphabet_validation')
print(val_generator[0][0][0].shape)

classifier = load_saved_model('../models/model19.keras')
print(classifier.summary())

print(classifier.evaluate_generator(val_generator))
