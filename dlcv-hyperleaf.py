import tensorflow as tf
import keras
import pandas as pd
import os
import sys
import PIL.Image as img
import PIL.ImageSequence as imgseq
import numpy as np
import tifffile

def dlcv_net():
	return keras.models.Sequential([
		keras.layers.Input(shape=(48, 352, 204)),
		keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same'),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
		keras.layers.ReLU(),
		keras.layers.MaxPool2D(pool_size=(2, 2)),
		keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same'),
		keras.layers.ReLU(),
		keras.layers.MaxPool2D(pool_size=(1, 2)),
		keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same'),
		keras.layers.ReLU(),
		keras.layers.MaxPool2D(pool_size=(1, 2)),
		keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same'),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same'),
		keras.layers.ReLU(),
		keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same'),
		keras.layers.ReLU(),
		keras.layers.Flatten(),
		keras.layers.Dense(32),
		keras.layers.ReLU(),
		keras.layers.Dense(64),
		keras.layers.ReLU(),
		keras.layers.Dense(8),
	])


def map_grain_weight(raw_weight):
	return np.log2(raw_weight) - 10


def unmap_grain_weight(mapped_weight):
	return np.exp2(mapped_weight + 10)


def path_to_image_id(workdir, img_id):
	img_id_str = "{:05d}".format(int(img_id))
	return os.path.join(workdir, "images", img_id_str + ".tiff")

def read_image(workdir, img_id):
	return np.transpose(
		np.array(list(imgseq.Iterator(img.open(path_to_image_id(workdir, img_id)))), dtype=np.float32),
		axes=(1, 2, 0)
	) / 65535.0

def dataset_generator(workdir: str, split_dataframe: pd.DataFrame):
	for idx, row in split_dataframe.iterrows():
		image_id = row['ImageId']
		raw_grain_weight = float(row['GrainWeight'])
		mapped_grain_weight = map_grain_weight(raw_grain_weight)
		stomatal_conductance = float(row['Gsw'])
		phi_ps2 = float(row['PhiPS2'])
		fertilizer = float(row['Fertilizer'])
		prob_heerup = float(row['Heerup'])
		prob_kvium = float(row['Kvium'])
		prob_rembrandt = float(row['Rembrandt'])
		prob_sheriff = float(row['Sheriff'])

		yield \
			read_image(workdir, image_id), \
			np.array([
				mapped_grain_weight, 
				stomatal_conductance,
				phi_ps2, fertilizer, prob_heerup, prob_kvium, prob_rembrandt, prob_sheriff
			], dtype=np.float32)


def prepare_submission(workdir, out_fn, sample_submission: pd.DataFrame, model: keras.Model):
	output_data = sample_submission.copy(deep=True)
	for idx, row in output_data.iterrows():
		print(idx)
		input = np.array([read_image(workdir, row['ImageId'])])
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
	




def entry(workdir, in_mdl=None):
	
	if in_mdl is not None:
		mdl = keras.models.load_model(in_mdl)
		prepare_submission(workdir, "submission.csv", pd.read_csv(os.path.join(workdir, "sample_submission.csv")), mdl)
		return

	train_csv = pd.read_csv(os.path.join(workdir, "train.csv"))
	
	train_meta = train_csv.iloc[:1440]
	val_meta = train_csv.iloc[1440:]

	batch_size = 16

	print(train_meta)
	print(val_meta)

	dataset_train = tf.data.Dataset \
		.from_generator(
			lambda: dataset_generator(workdir, train_meta),
			output_signature=(
				tf.TensorSpec(shape=(48, 352, 204), dtype=tf.float32),
				tf.TensorSpec(shape=(8,), dtype=tf.float32)
			)
		) \
		.prefetch(2) \
		.batch(batch_size, drop_remainder=True)

	dataset_val = tf.data.Dataset \
		.from_generator(
			lambda: dataset_generator(workdir, val_meta),
			output_signature=(
				tf.TensorSpec(shape=(48, 352, 204), dtype=tf.float32),
				tf.TensorSpec(shape=(8,), dtype=tf.float32)
			)
		) \
		.prefetch(2) \
		.batch(batch_size, drop_remainder=True)

	model = dlcv_net()

	model.compile(
		loss=keras.losses.MeanSquaredError(),
		optimizer=keras.optimizers.Adam()
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

	model.fit(dataset_train, validation_data=dataset_val, epochs=600, callbacks=[savecb])



if __name__ == "__main__":
	print(sys.argv)
	if len(sys.argv) > 1:
		entry("/run/media/volt/36E62701E626C14B/dataset_hyperleaf/", "dlcvnet.keras")

	entry("/run/media/volt/36E62701E626C14B/dataset_hyperleaf/")
	#entry("/run/media/volt/d0p1_misc/dataset_hyperleaf")
