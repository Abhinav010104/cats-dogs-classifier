import tensorflow_datasets as tfds
(ds_train, ds_val, ds_test), info = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
    as_supervised=True,
    with_info=True,
)
print("Dataset loaded successfully!")