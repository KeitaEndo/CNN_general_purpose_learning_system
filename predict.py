def pred(save_path, class_name, path_name):
    import sys, os, glob
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import keras
    from keras.models import load_model

    def check_photo(path,labels_num):
        x=np.asarray(path)
        im_rows=x.shape[0]
        im_cols=x.shape[1]
        im_color=x.shape[2]
        in_shape=(im_rows,im_cols,im_color)
        nb_classes=labels_num
        x=x.reshape(-1,im_rows,im_cols,im_color)

        pre=model.predict([x])[0]
        pre2=pre
        idx=pre.argmax()
        per=int(pre[idx]*100)
        x=list(range(nb_classes+1))
        del x[0]

        plt.tight_layout()
        plt.rcParams["font.size"] = 18
        plt.imshow(path)
        plt.show()
        print(pre2)
        return idx,per

    def check_photo_str(path,labels_num,class_name):
        idx,per=check_photo(path,labels_num)
        print("これは、",class_name[idx])
        print("可能性は、",per,"%")
        return

    plt.axis("off")

    model = load_model(save_path)

    pre_data = np.load(path_name)
    files = pre_data['x']

    for img in files:
        check_photo_str(img,len(class_name),class_name)
    return