import Image_Aug_choice
import cnn_learning
import predict
import fine_tuning

npz_path_train = ""
npz_path_test = ""
save_path = ""
class_name = []

while True:
    run_select = input('機械学習汎用システムの実行項目を選択(1～4)(終了: exit): ')
    if run_select == "1":
        npz_path_train, npz_path_test, class_name = Image_Aug_choice.aug_select()

    if run_select == "2":
        if npz_path_train == "":
            npz_path_train = input("学習画像のデータセットを入力(npzファイル): ")
        if npz_path_test == "":
            npz_path_test = input("テスト画像のデータセットを入力(npzファイル): ")
        save_path = cnn_learning.cnn_select(npz_path_train, npz_path_test)

    if run_select == "3":
        if npz_path_train == "":
            npz_path_train = input("追加で学習させる学習画像のデータセットを入力(npzファイル): ")
        if npz_path_test == "":
            npz_path_test = input("追加で学習させるテスト画像のデータセットを入力(npzファイル): ")
        save_path = fine_tuning.fine(npz_path_train, npz_path_test)

    if run_select == "4":
        if save_path == "":
            save_path = input('学習済みモデル(.h5)のパスを入力: ')
        if class_name == []:
            class_name_path = input('クラス名を保存したパスを入力(.txtまで入力してください): ')
            with open(class_name_path, 'r') as f:
                class_name = [line.strip() for line in f]
        if npz_path_test == "":
            npz_path_test = input('推論する画像のパスを入力(.npzまで入力してください): ')
        predict.pred(save_path, class_name, npz_path_test)

    if run_select == 'exit':
        break