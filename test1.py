def build_my_cnn(dim, n_class):
    model = Sequential()

    model.add(BatchNormalization(input_shape=(dim, dim, 3)))

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalMaxPooling2D())

    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(n_class))
    model.add(Activation('softmax'))

    return model


model = build_my_cnn(224, 574)

model.compile(
    loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['categorical_accuracy'])

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    # preprocessing_function=get_random_eraser(p=0.2, v_l=0, v_h=255, pixel_level=True),
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
checkpointer = ModelCheckpoint(
    filepath=f'../models/cnn_{len(fc)}_fc.h5', verbose=0, save_best_only=True)
model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=48),
    steps_per_epoch=len(x_train) / batch_size,
    validation_data=val_datagen.flow(x_val, y_val, batch_size=batch_size),
    validation_steps=len(x_val) / batch_size,
    epochs=epochs,
    callbacks=[early_stopping, checkpointer, reduce_lr],
    max_queue_size=10,
    workers=4,
    use_multiprocessing=False)
