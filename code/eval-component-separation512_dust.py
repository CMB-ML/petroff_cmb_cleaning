import time
from pathlib import Path
import numpy as np
import healpy as hp
import astropy.io.fits

import tensorflow as tf

tf.compat.v1.disable_eager_execution()  # This makes training considerably faster
tf.config.experimental.set_lms_enabled(True)
import tensorflow_datasets as tfds

from model import scnn_model, loss_mse, CmbDataset

start_time_str = "2020-02-27-15-30-45"
start_time = time.time()

BASE_PATH = Path("logs") / start_time_str

MODEL_FILENAME = str(BASE_PATH / "model_trained")


# Data locations
PLANCK_MAP_DIR = Path("/data/planck/")
SIM_DIR = Path("/data/users/petroff/psm-sims/outputnops512_updateddust_processed")
FIELD = 0
MIN_MAX_FILENAME = Path(
    "/data/users/petroff/psm-sims/outputnops512_processed_2/normalization.npz"
)
INPUT_MIN_MAX_FILENAME = SIM_DIR / "normalization.npz"

# Network parameters
NSIDE = 512
FILTER_MAPS = 6
POLY_ORDER = 9
LENGTH_SCALE = 1e-4
BATCH_SIZE = 1
N_TRAIN = 800
N_TEST = 200

ERROR_CALC_NUM = 100

planck_map_freqs = [70, 100, 143, 217, 353, 545, 857]

planck_map_names = [
    "LFI_SkyMap_070-BPassCorrected_1024_R3.00_full.fits",
    "HFI_SkyMap_100_2048_R3.01_full.fits",
    "HFI_SkyMap_143_2048_R3.01_full.fits",
    "HFI_SkyMap_217_2048_R3.01_full.fits",
    "HFI_SkyMap_353_2048_R3.01_full.fits",
    "HFI_SkyMap_545_2048_R3.01_full.fits",
    "HFI_SkyMap_857_2048_R3.01_full.fits",
]


npz = np.load(MIN_MAX_FILENAME)
x_max = npz["x_max"]
y_max = npz["y_max"]

test_ds = (
    CmbDataset(NSIDE, planck_map_freqs, N_TRAIN, N_TEST, SIM_DIR, False)
    .prefetch(2 * BATCH_SIZE)
    .batch(BATCH_SIZE)
)

model = scnn_model(
    NSIDE, FILTER_MAPS, POLY_ORDER, LENGTH_SCALE, N_TRAIN, len(planck_map_freqs)
)
model.compile(
    loss=loss_mse,
    optimizer=tf.keras.optimizers.Adam(amsgrad=True),
    experimental_run_tf_function=False,
)
model.load_weights(MODEL_FILENAME)

print("Loaded data and model:", time.time() - start_time, "seconds")


# Detailed evaluation of test example
# test_ds_iter = iter(CmbDataset(NSIDE, planck_map_freqs, 999, 1, SIM_DIR, False))
test_ds_iter = tfds.as_numpy(
    CmbDataset(NSIDE, planck_map_freqs, 999, 1, SIM_DIR, False)
)
maps = next(test_ds_iter)[0]
maps *= np.load(INPUT_MIN_MAX_FILENAME)["x_max"][:, None] / x_max[:, None]
tiled_maps = np.tile(maps, [ERROR_CALC_NUM, 1, 1])
result_comp = model.predict(tiled_maps, batch_size=BATCH_SIZE, verbose=1)
# np.savez_compressed(BASE_PATH / 'test0999_dm8_raw.npz', result_comp)
results = result_comp[:, :, 0] + np.random.normal(
    scale=np.sqrt(np.exp(-result_comp[:, :, 1]))
)
full_result = np.stack(
    [np.mean(results, axis=0), -np.log(np.var(results, axis=0))], axis=1
)
# Un-normalize result, switching back to T_cmb (uK)
full_result[..., 0] = (full_result[..., 0] - 0.5) * y_max * 2
hp.write_map(
    BASE_PATH / "test0999_dm8.fits",
    hp.reorder(full_result[:, 0], n2r=True),
    coord="G",
    column_units="uK",
)
hp.write_map(
    BASE_PATH / "test0999_dm8_err.fits",
    hp.reorder(np.sqrt(np.exp(-full_result[:, 1])) * y_max * 2, n2r=True),
    coord="G",
    column_units="uK",
)


instr_beams = np.append(
    astropy.io.fits.open(PLANCK_MAP_DIR / "LFI_RIMO_R3.31.fits")[
        "FREQUENCY_MAP_PARAMETERS"
    ].data["FWHM"],
    astropy.io.fits.open(PLANCK_MAP_DIR / "HFI_RIMO_R3.00.fits")["MAP_PARAMS"].data[
        "FWHM"
    ],
)
instr_beams = np.deg2rad(instr_beams / 60)[2:]

# Look at Planck maps
print("Evaluating Planck maps")
planck_maps = []
for i, pmn in enumerate(planck_map_names):
    planck_maps.append(
        hp.reorder(
            hp.alm2map(
                hp.smoothalm(
                    hp.map2alm(
                        hp.read_map(PLANCK_MAP_DIR / pmn, verbose=False, field=FIELD),
                        use_pixel_weights=True,
                    ),
                    beam_window=hp.gauss_beam(np.deg2rad(13.1 / 60), 3 * NSIDE)
                    / hp.gauss_beam(instr_beams[i], 3 * NSIDE),
                ),
                NSIDE,
                pixwin=True,
            ),
            r2n=True,
        )
    )

# Correct CIB monopole to match simulations
# From From Planck Collaboration III (2018)
cib_monopoles = np.array([0.0030, 0.0079, 0.033, 0.13, 0.35, 0.64])
# From fit to CIB component of PSM simulations
cib_monopoles -= np.array([0.0032, 0.010, 0.032, 0.10, 0.26, 0.55])
# From Planck Collaboration III (2018) and
# https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/UC_CC_Tables
conversion_factors = np.array([244.1, 371.7, 483.7, 287.5, 58.04, 2.268])
cib_monopoles /= conversion_factors  # Now in K_cmb
# Planck 545GHz and 857GHz maps aren't in K_cmb
planck_maps[5] /= conversion_factors[4]
planck_maps[6] /= conversion_factors[5]
# Subtract CIB monopole from maps
for i in range(1, 7):
    planck_maps[i] -= cib_monopoles[i - 1]

planck_maps = np.array(planck_maps)
planck_maps = np.tile(planck_maps, [ERROR_CALC_NUM, 1, 1])

# Normalize data
planck_maps = planck_maps / x_max[None, :, None]

# Evaluate
planck_result = model.predict(planck_maps, batch_size=BATCH_SIZE, verbose=1)
results = planck_result[:, :, 0] + np.random.normal(
    scale=np.sqrt(np.exp(-planck_result[:, :, 1]))
)
planck_result = np.stack(
    [np.mean(results, axis=0), -np.log(np.var(results, axis=0))], axis=1
)

# Un-normalize result, switching back to T_cmb (uK)
planck_result[..., 0] = (planck_result[..., 0] - 0.5) * y_max * 2

hp.write_map(
    BASE_PATH / "planck.fits",
    hp.reorder(planck_result[:, 0], n2r=True),
    coord="G",
    column_units="uK",
)
hp.write_map(
    BASE_PATH / "planck_err.fits",
    hp.reorder(np.sqrt(np.exp(-planck_result[:, 1])) * y_max * 2, n2r=True),
    coord="G",
    column_units="uK",
)


print("Finished evaluation:", time.time() - start_time, "seconds")
