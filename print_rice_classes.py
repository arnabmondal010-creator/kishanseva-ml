from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator()
gen = datagen.flow_from_directory("rice_dataset", batch_size=1)

print(gen.class_indices)
