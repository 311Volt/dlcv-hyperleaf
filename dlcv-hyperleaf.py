import datetime
import tensorflow as tf
import keras
import pandas as pd
import os
import sys
import time
import timeit
import PIL.Image as img
import PIL.ImageSequence as imgseq
import numpy as np
import tifffile

DSDIR = None

def dlcv_net():
	return keras.models.Sequential([
		keras.layers.Input(shape=(48, 352, 204)),
		keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.MaxPool2D(pool_size=(2, 2)),
		keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.MaxPool2D(pool_size=(1, 2)),
		keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.MaxPool2D(pool_size=(1, 2)),
		keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.MaxPool2D(pool_size=(1, 2)),
		keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Flatten(),
		keras.layers.Dense(24),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Dense(32),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Dense(32),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Dense(24),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.Dense(8),
	])


def map_grain_weight(raw_weight):
	return np.log2(raw_weight) - 10


def unmap_grain_weight(mapped_weight):
	return np.exp2(mapped_weight + 10)


def path_to_image_id(img_id):
	img_id_str = "{:05d}".format(int(img_id))
	return os.path.join(DSDIR, "images", img_id_str + ".tiff")

def read_image(img_id):
	imgpath = path_to_image_id(img_id)
	return (np.transpose(
		tifffile.imread(imgpath),
		axes=(1, 2, 0)
	) / 65535.0).astype(np.float32)
	# return np.transpose(
	# 	np.array(list(imgseq.Iterator(img.open(imgpath))), dtype=np.float32),
	# 	axes=(1, 2, 0)
	# ) / 65535.0

def map_dataset_row(row: np.array):
	t0 = timeit.default_timer()
	# print(row)
	# print(split_arr.shape)
	image_id = int(row[0])
	raw_grain_weight = float(row[1])
	mapped_grain_weight = map_grain_weight(raw_grain_weight)
	stomatal_conductance = float(row[2])
	phi_ps2 = float(row[3])
	fertilizer = float(row[4])
	prob_heerup = float(row[5])
	prob_kvium = float(row[6])
	prob_rembrandt = float(row[7])
	prob_sheriff = float(row[8])

	img = read_image(image_id)
	result = np.array([
			mapped_grain_weight, 
			stomatal_conductance, phi_ps2, fertilizer, 
			prob_heerup, prob_kvium, prob_rembrandt, prob_sheriff
		], dtype=np.float32)
	t1 = timeit.default_timer()
	# print("reading took {:.2f} ms".format(1000.0 * (t1-t0)))

	return img, result
			
def map_dataset_row_tf(split_arr):
	ret = tf.numpy_function(
		func=map_dataset_row,
		inp=[split_arr],
		Tout=[np.float32, np.float32]
	)
	ret[0].set_shape((48, 352, 204))
	ret[1].set_shape((8,))
	return ret

def prepare_submission(out_fn, sample_submission: pd.DataFrame, model: keras.Model):
	output_data = sample_submission.copy(deep=True)
	for idx, row in output_data.iterrows():
		print(idx)
		input = np.array([read_image(row['ImageId'])])
		if input.shape[1] != 48:
			continue
		pred = model.predict(input)[0]
		output_data.at[idx, 'GrainWeight'] = unmap_grain_weight(float(pred[0]))
		output_data.at[idx, 'Gsw'] = float(pred[1])
		output_data.at[idx, 'PhiPS2'] = float(pred[2])
		output_data.at[idx, 'Fertilizer'] = float(pred[3])
		output_data.at[idx, 'Heerup'] = float(pred[4])
		output_data.at[idx, 'Kvium'] = float(pred[5])
		output_data.at[idx, 'Rembrandt'] = float(pred[6])
		output_data.at[idx, 'Sheriff'] = float(pred[7])
	
	output_data.to_csv(out_fn)
	




def entry(in_mdl=None):

	keras.mixed_precision.set_global_policy('mixed_float16')
	
	if in_mdl is not None:
		mdl = keras.models.load_model(in_mdl)
		prepare_submission("submission.csv", pd.read_csv(os.path.join(DSDIR, "sample_submission.csv")), mdl)
		exit()

	train_csv = pd.read_csv(os.path.join(DSDIR, "train.csv"))
	
	train_meta = train_csv.iloc[:1440]
	val_meta = train_csv.iloc[1440:]

	batch_size = 8

	print(train_meta)
	print(val_meta)
	print(train_meta.to_numpy())
	print(val_meta.to_numpy())

	dataset_train = tf.data.Dataset \
		.from_tensor_slices(
			train_meta.to_numpy()
		) \
		.shuffle(len(train_meta), reshuffle_each_iteration=True) \
		.map(
			map_func=map_dataset_row_tf,
			num_parallel_calls=tf.data.AUTOTUNE
		) \
		.batch(batch_size, drop_remainder=True)
	
	dataset_val = tf.data.Dataset \
		.from_tensor_slices(
			val_meta.to_numpy()
		) \
		.shuffle(len(val_meta), reshuffle_each_iteration=True) \
		.map(
			map_func=map_dataset_row_tf,
			num_parallel_calls=tf.data.AUTOTUNE
		) \
		.batch(batch_size, drop_remainder=True)

	# dataset_val = tf.data.Dataset \
	# 	.from_generator(
	# 		lambda: dataset_generator(DSDIR, val_meta),
	# 		output_signature=(
	# 			tf.TensorSpec(shape=(48, 352, 204), dtype=tf.float32),
	# 			tf.TensorSpec(shape=(8,), dtype=tf.float32)
	# 		)
	# 	) \
	# 	.take(len(val_meta)) \
	# 	.prefetch(tf.data.AUTOTUNE) \
	# 	.cache() \
	# 	.batch(batch_size, drop_remainder=True)

	model = dlcv_net()

	model.compile(
		loss=keras.losses.MeanSquaredError(),
		optimizer=keras.optimizers.Adam(learning_rate=0.0004)
	)

	model.summary()

	savecb = keras.callbacks.ModelCheckpoint(
        "saved-model-epoch{epoch:03d}-{val_loss:.10f}.keras",
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='min'
    )

	logs = "logs/" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
	tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = (1, 70))


	model.fit(dataset_train, validation_data=dataset_val, epochs=600, callbacks=[savecb])
	model.save("final.keras")



if __name__ == "__main__":
	DSDIR = "/run/media/volt/36E62701E626C14B/dataset_hyperleaf/"
	print(sys.argv)
	if len(sys.argv) > 1:
		entry("dlcvnet.keras")

	entry()
	#entry("/run/media/volt/d0p1_misc/dataset_hyperleaf")
