import os
import datetime
import time
from pathlib import Path

import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # This makes training considerably faster
tf.config.experimental.set_lms_enabled(True)

from model import scnn_model, loss_mse, CmbDataset

# Data locations
SIM_DIR = Path("/tmp/outputnops512_updateddust_processed")

# Network parameters
NSIDE = 512
FILTER_MAPS = 6
POLY_ORDER = 9
LENGTH_SCALE = 1e-4

# Training parameters
EPOCHS = 400
BATCH_SIZE = 1
N_TRAIN = 800

planck_map_freqs = [70, 100, 143, 217, 353, 545, 857]


start_time_str = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
start_time = time.time()

os.makedirs(f"logs/{start_time_str}")

train_ds = (
    CmbDataset(NSIDE, planck_map_freqs, 0, N_TRAIN, SIM_DIR, True)
    .prefetch(2 * BATCH_SIZE)
    .batch(BATCH_SIZE)
)

# Compile model
model = scnn_model(
    NSIDE, FILTER_MAPS, POLY_ORDER, LENGTH_SCALE, N_TRAIN, len(planck_map_freqs)
)
model.compile(
    loss=loss_mse,
    optimizer=tf.keras.optimizers.Adam(amsgrad=True),
)

model.summary()
print("Model compiled:", time.time() - start_time, "seconds")

# Train model
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    f"logs/{start_time_str}/weights" + "_{epoch:03d}", save_weights_only=True
)
h = model.fit(train_ds, epochs=EPOCHS, verbose=1, callbacks=[cp_callback])

# Save model
model.save_weights(f"logs/{start_time_str}/model_trained", save_format="tf")

print("Training complete:", time.time() - start_time, "seconds")
