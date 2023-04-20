def cnn_select(npz_path_train, npz_path_test):
    import numpy as np
    import tensorflow as tf
    import keras.backend as K
    from keras.utils import to_categorical
    #from tensorflow.keras.utils import to_categorical
    from keras.datasets import mnist
    from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D, AveragePooling2D, Dropout, Input, LeakyReLU
    from keras.models import Sequential, Model
    from keras.callbacks import EarlyStopping
    import matplotlib.pyplot as plt
    import argparse

    def first_conv(x, kernel, conv_st, feature_map, act, conv_cnt):
        x = Conv2D(feature_map[conv_cnt], (kernel, kernel), activation = act, strides = conv_st, padding = 'same', name="block1_conv1")(x)
        return x

    def conv(n, x, kernel, conv_st, feature_map, act, conv_cnt):
        for i in range(n):
            if conv_cnt == 0:
                x = Conv2D(feature_map[conv_cnt], (kernel, kernel), activation = act, strides = conv_st, padding = 'same', name="block{:}_conv{:}".format(conv_cnt+1,i+2))(x)
            else:
                x = Conv2D(feature_map[conv_cnt], (kernel, kernel), activation = act, strides = conv_st, padding = 'same', name="block{:}_conv{:}".format(conv_cnt+1,i+1))(x)            
        return x

    def max_pool(n, x, pooling, pool_cnt):
        for i in range(n):
            x = MaxPooling2D((pooling,pooling), strides=None, padding='same', name="block{:}_pool{:}".format(pool_cnt+1,i+1))(x)
        return x

    def ave_pool(n, x, pooling, pool_cnt):
        for i in range(n):
            x = AveragePooling2D((pooling,pooling), strides=None, padding='same', name="block{:}_pool{:}".format(pool_cnt+1,i+1))(x)
        return x

    pool = {
        "1" : max_pool,
        "2" : ave_pool
        }

    def dropout1(n, x, drop_para1, drop_cnt):
        for i in range(n):
            x = Dropout(drop_para1, name="dropout_{:}".format(drop_cnt+i+1))(x)
        return x

    def dense(n2, x, dense_para, act, dense_cnt):
        for i2 in range(dense_cnt[-2],dense_cnt[-1]):
            x = Dense(dense_para[i2], activation = act, name="dense_{:}".format(i2+1))(x)
        return x

    def dropout2(n2, x, drop_para2, drop_cnt):
        for i2 in range(n2):
            x = Dropout(drop_para2, name="dropout_{:}".format(drop_cnt+i2+1))(x)
        return x

    def make_cnn(cnn_list1, inputs, kernel, conv_st, feature_map, act, pooling, pool_method, drop_para1, cnn_list2, dense_para, drop_para2):
        conv_cnt = 0
        pool_cnt = 0
        drop_cnt = 0
        for n, i in enumerate(cnn_list1):
            if n == 0:
                x = first_conv(inputs, kernel, conv_st, feature_map, act, conv_cnt)
                if i - 1 > 0:
                    x = conv(i-1, x, kernel, conv_st, feature_map, act, conv_cnt)
                elif x.shape[1] == 1:
                    break
            if n % 3 == 0 and n > 0:
                conv_cnt += 1
                x = conv(i, x, kernel, conv_st, feature_map, act, conv_cnt)
                if x.shape[1] == 1:
                    break
            if n % 3 == 1:
                x = pool[pool_method](i, x, pooling, pool_cnt)
                pool_cnt += 1
                if x.shape[1] == 1:
                    break
            if n % 3 == 2:
                x = dropout1(i, x, drop_para1, drop_cnt)
                drop_cnt += i
                if x.shape[1] == 1:
                    break
        x = Flatten()(x)
        dense_cnt = []
        dense_cnt.append(0)
        for n2, i2 in enumerate(cnn_list2):
            if n2 % 2 == 0 or n2 == 0:
                dense_cnt.append(dense_cnt[-1] + i2)
                x = dense(i2, x, dense_para, act, dense_cnt)
                if x.shape[1] == 1:
                    break
            else:
                x = dropout2(i2, x, drop_para2, drop_cnt)
                drop_cnt += i2
                if x.shape[1] == 1:
                    break
        return x

    def inputs_param():
        print("Conv層とPooling層とDropout層を入力(CNN前半)")
        print("Ex)Conv層2つ→Pooling層1つ→Dropout層なし→Conv層1つ→Pooling層1つ→Dropout層1つのCNNを作る場合")
        print("2 1 0 1 1 1")
        print("入力してください")
        cnn_list1 = list(map(int, input().split()))
        kernel = input("畳み込みのカーネルサイズを入力: ")
        if kernel == "":
            kernel = 3
        else:
            kernel = int(kernel)
        conv_st = input("畳み込みのストライドを入力: ")
        if conv_st == "":
            conv_st = 1
        else:
            conv_st = int(conv_st)
        feature_map = []
        feature_map_num = 0
        feature_map_method = input("畳み込みの特徴マップ数の決め方(1:畳み込みとプーリングが繰り返されるごとに2をかける, 2:畳み込みとプーリングが繰り返されるごとに特徴マップ数を入力): ")
        if feature_map_method == "1":
            feature_map_in = input("最初の畳み込みの特徴マップ数を入力: ")
            if feature_map_in == "":
                feature_map.append(32)
            else:
                feature_map.append(int(feature_map_in))
            for n, i in enumerate(cnn_list1):
                if n % 3 == 0 or n == 0:
                    feature_map_num += 1
            for k in range(feature_map_num - 1):
                feature_map.append(feature_map[k] * 2)
        if feature_map_method == "2":
            for n, i in enumerate(cnn_list1):
                if n % 3 == 0 or n == 0:
                    feature_map_num += 1
                    feature_map.append(int(input("{:}ユニット目の畳み込みの特徴マップ数を入力: ".format(feature_map_num))))
        act_select = input("活性化関数を選択(1～6): ")
        pool_method = input("プーリング手法を選択(1,2): ")
        if pool_method == "":
            pool_method = "1"
        pooling = input("プーリングサイズを入力: ")
        if pooling == "":
            pooling = 2
        else:
            pooling = int(pooling)
        drop = []
        for m, l in enumerate(cnn_list1):
            if m % 3 == 2:
                if l > 0:
                    drop.append(l)
        if not drop == []:
            drop_para1 = input("ドロップアウト率を入力(ex：0.25): ")
            if drop_para1 == "":
                drop_para1 = 0.25
            else:
                drop_para1 = float(drop_para1)
        else:
            drop_para1 = ""
        print("全結合層とDropout層を入力(CNN後半)")
        print("Ex)全結合層2つ→Dropout層1つのCNNを作る場合")
        print("2 1")
        print("入力してください")
        cnn_list2 = list(map(int, input().split()))
        if not cnn_list2[0] == 0:
            dense_para = []
            dense_num = 0
            dense_para_method = input("全結合層のノード数の決め方(1:全結合の層数が増えるごとに2をかける, 2:全ての全結合層に対してノード数を入力): ")
            if dense_para_method == "1":
                dense_para_in = input("最初の全結合層のノード数を入力(ex：512): ")
                if dense_para_in == "":
                    dense_para.append(512)
                else:
                    dense_para.append(int(dense_para_in))
                for n2, i2 in enumerate(cnn_list2):
                    if n2 % 2 == 0 or n2 == 0:
                        dense_num += i2
                for j in range(dense_num - 1):
                    dense_para.append(dense_para[j] * 2)
            if dense_para_method == "2":
                for n2, i2 in enumerate(cnn_list2):
                    if n2 % 2 == 0 or n2 == 0:
                        dense_num += i2
                for j in range(dense_num):
                    dense_para.append(int(input("{:}層目の全結合層のノード数を入力: ".format(j+1))))
            if not cnn_list2[1] == 0:
                drop_para2 = input("ドロップアウト率を入力(ex：0.5): ")
                if drop_para2 == "":
                    drop_para2 = 0.5
                else:
                    drop_para2 = float(drop_para2)
            else:
                drop_para2 = ""
        else:
            dense_para = ""
            drop_para2 = ""
        epoch = input("学習回数を入力: ")
        if epoch == "":
            epoch = 50
        else:
            epoch = int(epoch)
        batch = input("バッチサイズを入力: ")
        if batch == "":
            batch = 32
        else:
            batch = int(batch)
        opt_select = input("オプティマイザを選択(1～6): ")
        es_yn = input("Early Stoppingを使用しますか(y/n): ")
        save_path = input("学習後のモデルを保存するパスを入力(.h5まで入力してください): ")
        return cnn_list1, epoch, batch, kernel, conv_st, feature_map, act_select, pool_method, pooling, drop_para1, cnn_list2, dense_para, dense_num, drop_para2, opt_select, es_yn, save_path

    cnn_list1, epoch, batch, kernel, conv_st, feature_map, act_select, pool_method, pooling, drop_para1, cnn_list2, dense_para, dense_num, drop_para2, opt_select, es_yn, save_path = inputs_param()

    train_data = np.load(npz_path_train)
    x_train = train_data['x']
    y_train = train_data['y']

    test_data = np.load(npz_path_test)
    x_test = test_data['x']
    y_test = test_data['y']

    inputs = Input(shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]))

    if act_select=="1" or act_select=="":
        act='relu'
    if act_select=="2":
        act='sigmoid'
    if act_select=="3":
        act='tanh'
    if act_select=="4":
        act=LeakyReLU()
    if act_select=="5":
        act='elu'
    if act_select=="6":
        act='selu'

    #CNNのモデル作成
    try:
        x = make_cnn(cnn_list1, inputs, kernel, conv_st, feature_map, act, pooling, pool_method, drop_para1, cnn_list2, dense_para, drop_para2)
    except BaseException as e:
        print('入力ミスがあります．再入力してください．')
        cnn_list1, epoch, batch, kernel, conv_st, feature_map, act_select, pool_method, pooling, drop_para1, cnn_list2, dense_para, dense_num, drop_para2, opt_select, es_yn, save_path = inputs_param()
        x = make_cnn(cnn_list1, inputs, kernel, conv_st, feature_map, act, pooling, pool_method, drop_para1, cnn_list2, dense_para, drop_para2)

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
        output = Dense(y_train.shape[1], activation='sigmoid', name="dense_{:}".format(dense_num+1))(x)
        model = Model(inputs=inputs, outputs=output)
        model.summary()
        model.compile(optimizer = opt, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        output = Dense(y_train.shape[1], activation='softmax', name="dense_{:}".format(dense_num+1))(x)
        model = Model(inputs=inputs, outputs=output)
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