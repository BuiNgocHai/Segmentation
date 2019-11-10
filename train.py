import os
import matplotlib.pyplot as plt
import argparse
import time
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from callbacks import TrainCheck

#from mobilenet import *
from PTIT_model import *
from generator import data_generator

model_name = "mobile_net"

TRAIN_BATCH = 12
VAL_BATCH = 1
lr_init = 1e-4
lr_decay = 5e-4
vgg_path = None

# Use only 3 classes.
labels = ['background','road','traffic','car']

model = build_ptit((256,320,3), num_classes=len(labels),
                   lr_init=lr_init, lr_decay=lr_decay)
try:
    model.load_weights('mobile/mobile-032-0.88355.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

model.summary()

# Define callbacks
weight_path = "pit_model/pitmodel-{epoch:03d}-{val_focal_loss_fixed:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath=weight_path,
                             verbose=0,
                             monitor='val_focal_loss_fixed',
                             save_best_only=False,
                             save_weights_only=False, mode='auto', period=1)

tensorboard = TensorBoard(log_dir="logs/unet{}".format(time.time()),
                              batch_size=TRAIN_BATCH, write_images=True)

train_check = TrainCheck(output_path='./img', model_name=model_name)

# training
history = model.fit_generator(data_generator('../city/data_mix2.h5', TRAIN_BATCH, 'train'),
                              steps_per_epoch=88299 // TRAIN_BATCH,
                              validation_data=data_generator('../city/data_mix2.h5', VAL_BATCH, 'val'),
                              validation_steps=500 // VAL_BATCH,
                              callbacks=[checkpoint, tensorboard],
                              epochs=2000,
                              verbose=1,initial_epoch = 1)
