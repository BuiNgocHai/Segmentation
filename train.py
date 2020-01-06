import os
import argparse
import time
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from callbacks import TrainCheck

#from mobilenet import *
from PTIT_model import *
from generator import data_generator

model_name = "mobile_net"

TRAIN_BATCH = 64
VAL_BATCH = 64
lr_init = 1e-3
lr_decay = 5e-4
vgg_path = None

# Use only 3 classes.
labels = ['background','road','traffic','car']

model = build_ptit((256,320,3), num_classes=len(labels),
                   lr_init=lr_init, lr_decay=lr_decay)
try:
    model.load_weights('pit_model/mobile-032-0.88355.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

model.summary()

# Define callbacks
weight_path = "model/pitmodel_1road-{epoch:03d}-{val_iou:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath=weight_path,
                             verbose=0,
                             monitor='val_iou',
                             save_best_only=False,
                             save_weights_only=False, mode='auto', period=1)

tensorboard = TensorBoard(log_dir="/logs/pitmodel_1road{}".format(time.time()),
                              batch_size=TRAIN_BATCH, write_images=True)

train_check = TrainCheck(output_path='./img', model_name=model_name)

# training
history = model.fit_generator(data_generator('../city/data_mix4_1road.h5', TRAIN_BATCH, 'train'),
                              steps_per_epoch=92130 // TRAIN_BATCH,
                              validation_data=data_generator('../city/data_mix4_1road.h5', VAL_BATCH, 'val'),
                              validation_steps=1525 // VAL_BATCH,
                              callbacks=[checkpoint,train_check, tensorboard],
                              epochs=2000,
                              verbose=1,initial_epoch = 1)
