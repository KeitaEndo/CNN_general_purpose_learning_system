def fine(npz_path_train, npz_path_test):
    import keras
    from keras import optimizers
    from keras.models import load_model
    from keras.callbacks import EarlyStopping
    import matplotlib.pyplot as plt
    import numpy as np

    train_data = np.load(npz_path_train)
    x_train = train_data['x']
    y_train = train_data['y']

    test_data = np.load(npz_path_test)
    x_test = test_data['x']
    y_test = test_data['y']

    model_path = input('学習済みモデル(.h5)のパスを入力: ')
    model = load_model(model_path)

    epoch = input("学習回数を入力: ")
    if epoch == "":
        epoch = 50
    else:
        epoch = int(epoch)
    batch = input("バッチサイズを入力: ")
    if batch == "":
        batch = 64
    else:
        batch = int(batch)
    opt_select = input("オプティマイザを選択(1～6): ")
    es_yn = input("Early Stoppingを使用しますか(y/n): ")
    save_path = input("学習後のモデルを保存するパスを入力(.h5まで入力してください): ")


    names_conv1 = [l.name for l in model.layers if 'conv1' in l.name]
    last_conv1=names_conv1[-1]

    model.trainable = True
    set_trainable = False
    for layer in model.layers:
        if layer.name == last_conv1:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    if opt_select=="1" or opt_select=="":
        opt='adam'
    if opt_select=="2":
        opt='nadam'
    if opt_select=="3":
        opt='rmsprop'
    if opt_select=="4":
        opt='adamax'
    if opt_select=="5":
        opt='adagrad'
    if opt_select=="6":
        opt='sgd'

    if y_train.shape[1] == 2:
        model.summary()
        model.compile(optimizer = opt, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.summary()
        model.compile(optimizer = opt, loss='categorical_crossentropy', metrics=['accuracy'])

    if es_yn == 'y':
        es=EarlyStopping(monitor='val_loss',patience=4)
        hist=model.fit(x_train, y_train, epochs = epoch, batch_size=batch, verbose=1, validation_data=(x_test,y_test), callbacks=[es])
    if es_yn == 'n':
        hist=model.fit(x_train, y_train, epochs = epoch, batch_size=batch, verbose=1, validation_data=(x_test,y_test))

    eva = model.evaluate(x_test, y_test,batch_size=batch,verbose=1)
    print('正解率 = ',eva[1],'loss=',eva[0])

    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','test'],loc='upper left')
    plt.show()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train','test'],loc='upper left')
    plt.show()

    model.save(save_path)
    return save_path