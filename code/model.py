import numpy as np
import healpy as hp
import tensorflow as tf

import scnn.layers
import scnn.dropout


def scnn_model(nside, filter_maps, poly_order, length_scale, n_train, input_channels):
    """
    Construct neural network.
    """
    # Check parameters
    assert (
        nside > 0 or nside & (nside - 1) == 0
    ), "NSIDE must be a positive power of two"
    assert filter_maps > 0
    assert poly_order > 0

    # Construct nsides list
    nsides = []
    i = nside
    while i >= 1:
        nsides.append(i)
        i = i // 2
    nsides += reversed(nsides)

    # Construct filter maps and Chebyshev polynomial order lists
    filters = [filter_maps] * len(nsides) + [2]  # Result + uncertainty outputs
    poly_k = [poly_order] * len(nsides)

    n = n_train
    wd = length_scale ** 2.0 / n
    dd = 2.0 / n

    num_downscales = len(nsides) // 2

    x_input = []
    downscale = []

    # Construct inputs, one per input frequency
    full_input = tf.keras.layers.Input(shape=(input_channels, hp.nside2npix(nside)))
    for j in range(input_channels):
        x_input.append(full_input[:, j])

    # Construct coarsening / pooling layers, separate for each frequency
    for i in range(num_downscales):
        downscale.append([])
        for j in range(input_channels):
            if i != 0:
                # Pool all but first layer
                downscale[i].append(scnn.layers.GraphPool(4)(downscale[i - 1][j]))
            else:
                downscale[i].append(
                    tf.keras.layers.Reshape((hp.nside2npix(nside), 1))(x_input[j])
                )
            downscale[i][j] = scnn.dropout.SpatialConcreteDropout(
                scnn.layers.SphereConvolution(
                    filter_size=filters[i], poly_k=poly_k[i], nside=nsides[i]
                ),
                weight_regularizer=wd,
                dropout_regularizer=dd,
            )(downscale[i][j])
            downscale[i][j] = tf.keras.layers.Activation("elu")(downscale[i][j])
            downscale[i][j] = scnn.dropout.SpatialConcreteDropout(
                scnn.layers.SphereConvolution(
                    filter_size=filters[i], poly_k=poly_k[i], nside=nsides[i]
                ),
                weight_regularizer=wd,
                dropout_regularizer=dd,
            )(downscale[i][j])
            downscale[i][j] = tf.keras.layers.Activation("elu")(downscale[i][j])

    # Construct scaling / biasing for adding to upscaled layers, separate for each frequency
    for i in range(len(downscale)):
        for j in range(input_channels):
            downscale[i][j] = scnn.dropout.SpatialConcreteDropout(
                scnn.layers.LinearCombination(),
                weight_regularizer=wd,
                dropout_regularizer=dd,
            )(downscale[i][j])

    # Add separate fully-coarsened frequency layers into a single layer for upscaling
    upscale = tf.keras.layers.add(downscale[-1])
    upscale = tf.keras.layers.Activation("elu")(upscale)
    upscale = tf.keras.layers.UpSampling1D(4)(upscale)

    # Construct upscaling / skip connection layers
    for i in range(1, num_downscales):
        upscale = scnn.dropout.SpatialConcreteDropout(
            scnn.layers.SphereConvolution(
                filter_size=filters[num_downscales + i],
                poly_k=poly_k[num_downscales + i],
                nside=nsides[num_downscales + i],
            ),
            weight_regularizer=wd,
            dropout_regularizer=dd,
        )(upscale)
        upscale = tf.keras.layers.add([upscale] + downscale[-1 - i])  # Skip connection
        upscale = tf.keras.layers.Activation("elu")(upscale)
        upscale = scnn.dropout.SpatialConcreteDropout(
            scnn.layers.SphereConvolution(
                filter_size=filters[num_downscales + i],
                poly_k=poly_k[num_downscales + i],
                nside=nsides[num_downscales + i],
            ),
            weight_regularizer=wd,
            dropout_regularizer=dd,
        )(upscale)
        upscale = tf.keras.layers.Activation("elu")(upscale)
        if i < num_downscales - 1:
            # Upsample all but last layer
            upscale = tf.keras.layers.UpSampling1D(4)(upscale)

    # Final convolution layer
    upscale = scnn.dropout.SpatialConcreteDropout(
        scnn.layers.SphereConvolution(
            filter_size=filters[-1], poly_k=poly_k[-1], nside=nsides[-1]
        ),
        weight_regularizer=wd,
        dropout_regularizer=dd,
    )(upscale)

    return tf.keras.models.Model(inputs=full_input, outputs=upscale)


def loss_mse(labels, predictions):
    sq_err = tf.math.square(
        predictions[:, :, 0] - tf.reshape(labels, tf.shape(input=predictions[:, :, 0]))
    )  # Reshape needed due to incorrect assumption of shape
    # Combine with log variance: Kendall & Gal 2017
    log_var = -predictions[:, :, 1]
    return 0.5 * tf.math.reduce_mean(
        input_tensor=tf.math.multiply(sq_err, tf.math.exp(-log_var)) + log_var, axis=-1
    )


class CmbDataset(tf.data.Dataset):
    def __new__(cls, nside, freqs, first_example, n_examples, base_dir, is_random):
        def _generator():
            # Load data
            ids = np.arange(n_examples)
            if is_random:
                np.random.shuffle(ids)  # Randomize order each epoch
            for i in ids:
                y_raw = hp.read_map(
                    base_dir / f"sim{first_example + i:04}/cmb.fits",
                    verbose=False,
                    dtype=np.float32,
                    nest=True,
                )
                x_raw = []
                for j, freq in enumerate([f"{freq:03d}" for freq in freqs]):
                    x_raw.append(
                        hp.read_map(
                            base_dir / f"sim{first_example + i:04}/{freq}.fits",
                            verbose=False,
                            dtype=np.float32,
                            nest=True,
                        )
                    )
                x_raw = np.array(x_raw)
                yield (x_raw, y_raw)

        return tf.data.Dataset.from_generator(
            _generator,
            output_types=(tf.dtypes.float32, tf.dtypes.float32),
            output_shapes=(
                (len(freqs), hp.nside2npix(nside)),
                (hp.nside2npix(nside),),
            ),
        )
