def aug_select():
    import os,glob,random
    import cv2
    from PIL import Image
    import numpy as np
    import keras
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    import Image_Augmentation

    Image_Aug = {
        '1' : Image_Augmentation.padding,
        '2' : Image_Augmentation.resize,
        '3' : Image_Augmentation.center_crop,
        '4' : Image_Augmentation.random_crop,
        '5' : Image_Augmentation.scale_augmentation,
        '6' : Image_Augmentation.aspect_ratio_augmentation,
        '7' : Image_Augmentation.rotation_1,
        '8' : Image_Augmentation.rotation_2,
        '9' : Image_Augmentation.vertical_flip,
        '10' : Image_Augmentation.horizontal_flip,
        '11' : Image_Augmentation.zoom_in_only,
        '12' : Image_Augmentation.zoom_in_out,
        '13' : Image_Augmentation.vertical_shift,
        '14' : Image_Augmentation.horizontal_shift,
        '15' : Image_Augmentation.threshold_normal,
        '16' : Image_Augmentation.threshold_otsu,
        '17' : Image_Augmentation.adaptive_threshold_mean,
        '18' : Image_Augmentation.adaptive_threshold_gaussian,
        '19' : Image_Augmentation.blur,
        '20' : Image_Augmentation.gaussian_blur,
        '21' : Image_Augmentation.median_blur,
        '22' : Image_Augmentation.bilateral_filter,
        '23' : Image_Augmentation.gaussian_noise,
        '24' : Image_Augmentation.impulse_noise,
        '25' : Image_Augmentation.brightness_contrast,
        '26' : Image_Augmentation.gamma_correction,
        '27' : Image_Augmentation.pca_color_augmentation,
        '28' : Image_Augmentation.channel_shift,
        '29' : Image_Augmentation.affine,
        '30' : Image_Augmentation.syaei,
        '31' : Image_Augmentation.shear,
        '32' : Image_Augmentation.PCA_Whitening,
        '33' : Image_Augmentation.ZCA_Whitening,
        '34' : Image_Augmentation.standardization,
        '35' : Image_Augmentation.normalization_max_min,
        '36' : Image_Augmentation.normalization_255,
        '37' : Image_Augmentation.mean_subtraction,
        '38' : Image_Augmentation.per_pixel_mean_subtraction
        }

    def countFiles(target_dir_name):
        cnt = 0
        # 指定したパス内の全てのファイルとディレクトリを要素とするリストを返す
        target_dir_files = os.listdir(target_dir_name)
        for file in target_dir_files:
            new_target_dir_name = target_dir_name + "\\" +file
            #ディレクトリか非ディレクトリで条件分岐
            if os.path.isdir(new_target_dir_name):
                #ディレクトリの場合、中に入って探索する
                cnt += countFiles(new_target_dir_name)
            else:
                #非ディレクトリの場合、数え上げを行う
                cnt += 1
        return cnt

    input_data = []
    class_name = []
    x=[]
    y=[]
    count = 0
    all_count = 0

    while True:
        path_name = input('入力画像のパスを入力してください(終了: exit): ')
        if path_name == 'exit':
            break
        input_data.append(path_name)
        class_name.append(os.path.basename(path_name))
    print(class_name)

    for file_tag in range(len(input_data)):
        file_num = countFiles(input_data[file_tag])
        all_count += file_num

    size_method = input('画像サイズ変更方法選択(1～6): ')
    photo_size = int(input('変更サイズ: '))
    if size_method == "4":
        resize_size = int(input('大きめにリサイズする際の短辺の長さ: '))
        cnt2 = int(input('何枚ずつ生成しますか：'))
    if size_method == "5" or size_method == "6":
        resize_size_upperlimit = int(input('大きめにリサイズする際の短辺の長さの上限: '))
        resize_size_lowerlimit = int(input('大きめにリサイズする際の短辺の長さの下限: '))
        cnt2 = int(input('何枚ずつ生成しますか：'))

    for file_path in input_data:
        files=[]
        files.extend(glob.glob(file_path + "/*.jpg"))
        files.extend(glob.glob(file_path + "/*.jpeg"))
        files.extend(glob.glob(file_path + "/*.png"))
        random.shuffle(files)
        num=0
        for f in files:
            if num >=all_count : break
            num += 1
            if size_method == "1" or size_method == "2" or size_method == "3":
                img=Image_Aug[size_method](f,photo_size)
                x.append(img)
                y.append(count)
            if size_method == "4":
                img=Image_Aug[size_method](f,photo_size,resize_size,cnt2)
                x.extend(img)
                for j in range (cnt2):
                    y.append(count)
            if size_method == "5" or size_method == "6":
                img=Image_Aug[size_method](f,photo_size,resize_size_upperlimit,resize_size_lowerlimit,cnt2)
                x.extend(img)
                for j in range (cnt2):
                    y.append(count)
        count += 1

    x = np.array(x)
    y = np.array(y)

    print(x.shape)

    nb_classes=len(input_data)
    y=to_categorical(y, num_classes=nb_classes, dtype='int32')

    train_ratio = int(input('学習データの割合(%): '))
    train_ratio = float(train_ratio/100)

    if train_ratio == 0:
        x_train = np.array([])
        y_train = np.array([])
        x_test = x
        y_test = y
    if 0 < train_ratio < 1:
        x_train,x_test,y_train,y_test=train_test_split(
            x,y,train_size=train_ratio,stratify=y)
    if train_ratio == 1:
        x_train = x
        y_train = y
        x_test = np.array([])
        y_test = np.array([])

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    x_out1 = []
    y_out1 = []

    while True:
        x_new = []
        y_new = []
        method = []
        cnt3 = 0
        while True:
            method_1 = input('前処理・水増し方法の選択(7～31)(終了: exit): ')
            if method_1 == 'exit':
                break
            if cnt3 == 0:
                x_new,y_new = Image_Aug[method_1](x_train,y_train)
            if method_1 == '7' or method_1 == '8':
                method.append(0)
                if cnt3 >= 1 and method.count(0) <= 1:
                    x_new,y_new = Image_Aug[method_1](x_new,y_new)
                if cnt3 >= 1 and method.count(0) > 1:
                    print('同じ種類の処理であるため実行しませんでした．')
            if method_1 == '9':
                method.append(1)
                if cnt3 >= 1 and method.count(1) <= 1:
                    x_new,y_new = Image_Aug[method_1](x_new,y_new)
                if cnt3 >= 1 and method.count(1) > 1:
                    print('同じ種類の処理であるため実行しませんでした．')
            if method_1 == '10':
                method.append(2)
                if cnt3 >= 1 and method.count(2) <= 1:
                    x_new,y_new = Image_Aug[method_1](x_new,y_new)
                if cnt3 >= 1 and method.count(2) > 1:
                    print('同じ種類の処理であるため実行しませんでした．')
            if method_1 == '11' or method_1 == '12':
                method.append(3)
                if cnt3 >= 1 and method.count(3) <= 1:
                    x_new,y_new = Image_Aug[method_1](x_new,y_new)
                if cnt3 >= 1 and method.count(3) > 1:
                    print('同じ種類の処理であるため実行しませんでした．')
            if method_1 == '13':
                method.append(4)
                if cnt3 >= 1 and method.count(4) <= 1:
                    x_new,y_new = Image_Aug[method_1](x_new,y_new)
                if cnt3 >= 1 and method.count(4) > 1:
                    print('同じ種類の処理であるため実行しませんでした．')
            if method_1 == '14':
                method.append(5)
                if cnt3 >= 1 and method.count(5) <= 1:
                    x_new,y_new = Image_Aug[method_1](x_new,y_new)
                if cnt3 >= 1 and method.count(5) > 1:
                    print('同じ種類の処理であるため実行しませんでした．')
            if method_1 == '15' or method_1 == '16' or method_1 == '17' or method_1 == '18':
                method.append(6)
                if cnt3 >= 1 and method.count(6) <= 1:
                    x_new,y_new = Image_Aug[method_1](x_new,y_new)
                if cnt3 >= 1 and method.count(6) > 1:
                    print('同じ種類の処理であるため実行しませんでした．')
            if method_1 == '19' or method_1 == '20' or method_1 == '21' or method_1 == '22':
                method.append(7)
                if cnt3 >= 1 and method.count(7) <= 1:
                    x_new,y_new = Image_Aug[method_1](x_new,y_new)
                if cnt3 >= 1 and method.count(7) > 1:
                    print('同じ種類の処理であるため実行しませんでした．')
            if method_1 == '23' or method_1 == '24':
                method.append(8)
                if cnt3 >= 1 and method.count(8) <= 1:
                    x_new,y_new = Image_Aug[method_1](x_new,y_new)
                if cnt3 >= 1 and method.count(8) > 1:
                    print('同じ種類の処理であるため実行しませんでした．')
            if method_1 == '25' or method_1 == '26' or method_1 == '27':
                method.append(9)
                if cnt3 >= 1 and method.count(9) <= 1:
                    x_new,y_new = Image_Aug[method_1](x_new,y_new)
                if cnt3 >= 1 and method.count(9) > 1:
                    print('同じ種類の処理であるため実行しませんでした．')
            if method_1 == '28':
                method.append(10)
                if cnt3 >= 1 and method.count(10) <= 1:
                    x_new,y_new = Image_Aug[method_1](x_new,y_new)
                if cnt3 >= 1 and method.count(10) > 1:
                    print('同じ種類の処理であるため実行しませんでした．')
            if method_1 == '29' or method_1 == '30' or method_1 == '31':
                method.append(11)
                if cnt3 >= 1 and method.count(11) <= 1:
                    x_new,y_new = Image_Aug[method_1](x_new,y_new)
                if cnt3 >= 1 and method.count(11) > 1:
                    print('同じ種類の処理であるため実行しませんでした．')
            cnt3 += 1
        if method_1 == 'exit' and cnt3 == 0:
            break
        x_out1.extend(x_new)
        y_out1.extend(y_new)
        continue_1 = input('別の前処理・水増しの組み合わせでも画像を生成しますか(y/n): ')
        if continue_1 == 'y':
            continue
        if continue_1 == 'n':
            x_train = np.array(x_out1)
            y_train = np.array(y_out1)
            break

    print(x_train.shape)

    mabiku = input('水増し後の画像が何枚以上の場合間引くか: ')
    if mabiku == "":
        mabiku = 10000
    else:
        mabiku = int(mabiku)

    if x_train.shape[0] > mabiku:
        x_train_random = []
        y_train_random = []
        par = round(mabiku/nb_classes)
        for class_cnt in range (nb_classes):
            x_train_parclass = []
            y_train_parclass = []
            for l,yt in enumerate(y_train):
                yt = yt.tolist()
                yt = yt.index(1)
                if yt == class_cnt:
                    y_train_parclass.append(l)
            for p,yt2 in enumerate (y_train_parclass):
                x_train_parclass.append(x_train[yt2])
            x_train_parclass_random = random.sample(x_train_parclass,par)
            x_train_random.extend(x_train_parclass_random)
            y_train_parclass_random = np.array([class_cnt] * par)
            y_train_parclass_random = to_categorical(y_train_parclass_random, num_classes=nb_classes, dtype='int32')
            y_train_parclass_random = y_train_parclass_random.tolist()
            y_train_random.extend(y_train_parclass_random)
        x_train = np.array(x_train_random)
        y_train = np.array(y_train_random)


    png_path = input('pngでデータセットを出力するパス: ')
    os.makedirs(png_path + "\\train", exist_ok=True)
    for class_cnt_train in range (nb_classes):
        os.makedirs(png_path + "\\train\class{:}_{:}".format(class_cnt_train,class_name[class_cnt_train]), exist_ok=True)
        train_count = 0
        for l_train,yt_train in enumerate(y_train):
            yt_train = yt_train.tolist()
            yt_train = yt_train.index(1)
            if yt_train == class_cnt_train:
                x_train_parclass_save = x_train[l_train]
                x_train_parclass_save = Image.fromarray(x_train_parclass_save)
                x_train_parclass_save.save(png_path + "\\train\class{:}_{:}\{:}.png".format(class_cnt_train,class_name[class_cnt_train],train_count))
                train_count += 1

    os.makedirs(png_path + "\\test", exist_ok=True)
    for class_cnt_test in range (nb_classes):
        os.makedirs(png_path + "\\test\class{:}_{:}".format(class_cnt_test,class_name[class_cnt_test]), exist_ok=True)
        test_count = 0
        for l_test,yt_test in enumerate(y_test):
            yt_test = yt_test.tolist()
            yt_test = yt_test.index(1)
            if yt_test == class_cnt_test:
                x_test_parclass_save = x_test[l_test]
                x_test_parclass_save = Image.fromarray(x_test_parclass_save)
                x_test_parclass_save.save(png_path + "\\test\class{:}_{:}\{:}.png".format(class_cnt_test,class_name[class_cnt_test],test_count))
                test_count += 1


    method_2 = input('全体にかける前処理選択(32～38)(終了: exit): ')
    if not method_2 == 'exit':
        x_out2,y_out2 = Image_Aug[method_2](x_train,y_train)
        x_out3,y_out3 = Image_Aug[method_2](x_test,y_test)
        x_train = np.array(x_out2)
        y_train = np.array(y_out2)
        x_test = np.array(x_out3)
        y_test = np.array(y_out3)

    npz_path_train = input('npzで学習画像のデータセットを出力するパス(.npzまで入力してください): ')
    np.savez(npz_path_train,x=x_train,y=y_train)
    print("保存しました: " + npz_path_train,len(x_train))

    npz_path_test = input('npzでテスト画像のデータセットを出力するパス(.npzまで入力してください): ')
    np.savez(npz_path_test,x=x_test,y=y_test)
    print("保存しました: " + npz_path_test,len(x_test))
    
    class_name_path = input('クラス名を保存するパス(.txtまで入力してください): ')
    str_ = '\n'.join(class_name)
    with open(class_name_path, 'wt') as f:
        f.write(str_)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    return npz_path_train, npz_path_test, class_name