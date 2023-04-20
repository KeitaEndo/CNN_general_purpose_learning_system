import numpy as np
import cv2
import math
from keras.preprocessing import image


# パディングして正方形にリサイズ
def padding(input_path,resize_size):
    img_array = np.fromfile(input_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    tmp = img[:, :]
    height, width = img.shape[:2]
    if(height > width):
        size = height
        limit = width
    else:
        size = width
        limit = height
    start = int((size - limit) / 2)
    fin = int((size + limit) / 2)
    new_img = cv2.resize(np.zeros((1, 1, 3), np.uint8), (size, size))
    if(size == height):
        new_img[:, start:fin] = tmp
    else:
        new_img[start:fin, :] = tmp
    resize_img = cv2.resize(new_img, dsize=(resize_size, resize_size))
    return resize_img


# リサイズのみ
def resize(input_path,resize_size):
    img_array = np.fromfile(input_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    resize_img = cv2.resize(img, dsize=(resize_size, resize_size))
    return resize_img


# Center Crop
def center_crop(input_path,resize_size):
    img_array = np.fromfile(input_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if h <= w:
        width = round(w * (resize_size / h))
        dst = cv2.resize(img, dsize=(width, resize_size))
        #切り取る部分の左上の頂点を決める
        j = int((width-resize_size)/2)
        #頂点から指定のサイズでスライスし、画像を書き出す
        resize_img = dst[0:resize_size, j:j+resize_size]
    else:
        height = round(h * (resize_size / w))
        dst = cv2.resize(img, dsize=(resize_size, height))
        #切り取る部分の左上の頂点を決める
        i = int((height-resize_size)/2)
        #頂点から指定のサイズでスライスし、画像を書き出す
        resize_img = dst[i:i+resize_size, 0:resize_size]
    return resize_img


# Random Crop
def random_crop(input_path,crop_size,resize_size,count):
    output_img = []
    img_array = np.fromfile(input_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if h <= w:
        width = round(w * (resize_size / h))
        dst = cv2.resize(img, dsize=(width, resize_size))
        left = int((width-resize_size)/2)
        center_img = dst[0:resize_size, left:left+resize_size]
        for j in range (count):
            top_random = np.random.randint(0, resize_size - crop_size)
            left_random = np.random.randint(0, resize_size - crop_size)
            bottom = top_random + crop_size
            right = left_random + crop_size
            resize_img = center_img[top_random:bottom, left_random:right]
            output_img.append(resize_img)
    else:
        height = round(h * (resize_size / w))
        dst = cv2.resize(img, dsize=(resize_size, height))
        top = int((height-resize_size)/2)
        center_img = dst[top:top+resize_size, 0:resize_size]
        for j in range (count):
            top_random = np.random.randint(0, resize_size - crop_size)
            left_random = np.random.randint(0, resize_size - crop_size)
            bottom = top_random + crop_size
            right = left_random + crop_size
            resize_img = center_img[top_random:bottom, left_random:right]
            output_img.append(resize_img)
    return output_img


# Scale Augmentation
def scale_augmentation(input_path,crop_size,resize_size_upperlimit,resize_size_lowerlimit,count):
    output_img = []
    img_array = np.fromfile(input_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if h <= w:
        resize_size = np.random.randint(resize_size_lowerlimit,resize_size_upperlimit)
        width = round(w * (resize_size / h))
        dst = cv2.resize(img, dsize=(width, resize_size))
        for j in range (count):
            top = np.random.randint(0, resize_size - crop_size)
            left = np.random.randint(0, width - crop_size)
            bottom = top + crop_size
            right = left + crop_size
            resize_img = dst[top:bottom, left:right, :]
            output_img.append(resize_img)
    else:
        resize_size = np.random.randint(resize_size_lowerlimit,resize_size_upperlimit)
        height = round(h * (resize_size / w))
        dst = cv2.resize(img, dsize=(resize_size, height))
        for j in range (count):
            top = np.random.randint(0, height - crop_size)
            left = np.random.randint(0, resize_size - crop_size)
            bottom = top + crop_size
            right = left + crop_size
            resize_img = dst[top:bottom, left:right, :]
            output_img.append(resize_img)
    return output_img


# Aspect Ratio Augmentation
def aspect_ratio_augmentation(input_path,crop_size,resize_size_upperlimit,resize_size_lowerlimit,count):
    output_img = []
    img_array = np.fromfile(input_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if h <= w:
        resize_size = np.random.randint(resize_size_lowerlimit,resize_size_upperlimit)
        aspect = np.random.uniform(1,4/3)
        width = round(resize_size * aspect)
        dst = cv2.resize(img, dsize=(width, resize_size))
        for j in range (count):
            top = np.random.randint(0, resize_size - crop_size)
            left = np.random.randint(0, width - crop_size)
            bottom = top + crop_size
            right = left + crop_size
            resize_img = dst[top:bottom, left:right, :]
            output_img.append(resize_img)
    else:
        resize_size = np.random.randint(resize_size_lowerlimit,resize_size_upperlimit)
        aspect = np.random.uniform(1,4/3)
        height = round(resize_size * aspect)
        dst = cv2.resize(img, dsize=(resize_size, height))
        for j in range (count):
            top = np.random.randint(0, height - crop_size)
            left = np.random.randint(0, resize_size - crop_size)
            bottom = top + crop_size
            right = left + crop_size
            resize_img = dst[top:bottom, left:right, :]
            output_img.append(resize_img)
    return output_img


#　はみ出して回転（パディング）
def rotation_1(input_x,input_y,angle=60):
    angle2 = input('回転角度：')
    if not angle2 == "":
        angle = int(angle2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        #高さを定義
        height = xi.shape[0]
        #幅を定義
        width = xi.shape[1]
        #回転の中心を指定
        center = (int(width/2), int(height/2))
        for j in range(math.ceil(360/angle)):
            # 回転を実行
            angle3 = j*angle
            mtx=cv2.getRotationMatrix2D(center,angle3,1)
            img_rot=cv2.warpAffine(xi,mtx,(height,width))
            output_xi.append(img_rot)
            output_yi.append(yi)
    return output_xi,output_yi


#　はみ出さずに回転（パディング）
def rotation_2(input_x,input_y,angle=60):
    angle2 = input('回転角度：')
    if not angle2 == "":
        angle = int(angle2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        #高さを定義
        h = xi.shape[0]      
        #幅を定義
        w = xi.shape[1]
        for j in range(math.ceil(360/angle)):
            # 回転後のサイズ
            angle3 = j*angle
            radian = np.radians(angle3)
            sine = np.abs(np.sin(radian))
            cosine = np.abs(np.cos(radian))
            tri_mat = np.array([[cosine, sine],[sine, cosine]], np.float32)
            old_size = np.array([w,h], np.float32)
            new_size = np.ravel(np.dot(tri_mat, old_size.reshape(-1,1)))
            # 回転アフィン
            affine = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle3, 1.0)
            # 平行移動
            affine[:2,2] += (new_size-old_size)/2.0
            # リサイズ
            affine[:2,:] *= (old_size / new_size).reshape(-1,1)
            img_rot = cv2.warpAffine(xi, affine, (w, h))
            output_xi.append(img_rot)
            output_yi.append(yi)
    return output_xi,output_yi


# 上下反転
def vertical_flip(input_x,input_y):
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        img = cv2.flip(xi,0)
        output_xi.append(xi)
        output_yi.append(yi)
        output_xi.append(img)
        output_yi.append(yi)
    return output_xi,output_yi


# 左右反転
def horizontal_flip(input_x,input_y):
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        img = cv2.flip(xi,1)
        output_xi.append(xi)
        output_yi.append(yi)
        output_xi.append(img)
        output_yi.append(yi)
    return output_xi,output_yi


# 拡大のみ（アスペクト比固定）
def zoom_in_only(input_x,input_y,zoom_range_lowerlimit=100,zoom_range_upperlimit=150,count=3):
    zoom_range_lowerlimit2 = input('拡大倍率下限(%)：')
    if not zoom_range_lowerlimit2 == "":
        zoom_range_lowerlimit = int(zoom_range_lowerlimit2)
    zoom_range_lowerlimit = zoom_range_lowerlimit/100
    zoom_range_upperlimit2 = input('拡大倍率上限(%)：')
    if not zoom_range_upperlimit2 == "":
        zoom_range_upperlimit = int(zoom_range_upperlimit2)
    zoom_range_upperlimit = zoom_range_upperlimit/100
    count2 = input('何枚ずつ生成しますか：')
    if not count2 == "":
        count = int(count2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        #高さを定義
        h = xi.shape[0]                    
        #幅を定義
        w = xi.shape[1]
        for j in range (count):
            zoom_range = np.random.uniform(zoom_range_lowerlimit, zoom_range_upperlimit)
            img_zoom = cv2.resize(xi, dsize=None, fx=zoom_range, fy=zoom_range)
            resize_img = img_zoom[int((h*zoom_range-h)/2):int(((h*zoom_range-h)/2)+h), int((w*zoom_range-w)/2):int(((w*zoom_range-w)/2)+w)]
            output_xi.append(resize_img)
            output_yi.append(yi)
    return output_xi,output_yi


# 拡大縮小（縮小時近辺画素パディング）
def zoom_in_out(input_x,input_y,zoom_range_lowerlimit=50,zoom_range_upperlimit=150,count=5):
    zoom_range_lowerlimit2 = input('拡大縮小倍率下限(%)：')
    if not zoom_range_lowerlimit2 == "":
        zoom_range_lowerlimit = int(zoom_range_lowerlimit2)
    zoom_range_lowerlimit = zoom_range_lowerlimit/100
    zoom_range_upperlimit2 = input('拡大縮小倍率上限(%)：')
    if not zoom_range_upperlimit2 == "":
        zoom_range_upperlimit = int(zoom_range_upperlimit2)
    zoom_range_upperlimit = zoom_range_upperlimit/100
    count2 = input('何枚ずつ生成しますか：')
    if not count2 == "":
        count = int(count2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        xi = np.array(xi)
        datagen = image.ImageDataGenerator(zoom_range=[zoom_range_lowerlimit,zoom_range_upperlimit])
        x = xi[np.newaxis]
        gen = datagen.flow(x, batch_size=1)
        for j in range(count):
            batches = next(gen)
            resize_img = batches[0].astype(np.uint8)
            output_xi.append(resize_img)
            output_yi.append(yi)
    return output_xi,output_yi


# 垂直シフト
def vertical_shift(input_x,input_y,zure=20,up=40,down=40):
    zure2 = input('何ピクセルずつずらしますか：')
    if not zure2 == "":
        zure = int(zure2)
    up2 = input('上方向にどこまでずらしますか：')
    if not up2 == "":
        up = int(up2)
    down2 = input('下方向にどこまでずらしますか：')
    if not down2 == "":
        down = int(down2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        h, w = xi.shape[:2]
        if h/2 <= up:
            up=h/2
        if h/2 <= down:
            down=h/2
        for i in range(int((down/zure)+1)):
            src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
            dest = src.copy()
            dest[:,1] += i*zure # シフトするピクセル値
            affine = cv2.getAffineTransform(src, dest)
            image_shift_down = cv2.warpAffine(xi, affine, (w, h))
            output_xi.append(image_shift_down)
            output_yi.append(yi)
        for i in range(1,int((up/zure)+1)):
            src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
            dest2 = src.copy()    
            dest2[:,1] -= i*zure # シフトするピクセル値
            affine2 = cv2.getAffineTransform(src, dest2)
            image_shift_up = cv2.warpAffine(xi, affine2, (w, h))
            output_xi.append(image_shift_up)
            output_yi.append(yi)
    return output_xi,output_yi


# 水平シフト
def horizontal_shift(input_x,input_y,zure=20,left=40,right=40):
    zure2 = input('何ピクセルずつずらしますか：')
    if not zure2 == "":
        zure = int(zure2)
    left2 = input('左方向にどこまでずらしますか：')
    if not left2 == "":
        left = int(left2)
    right2 = input('右方向にどこまでずらしますか：')
    if not right2 == "":
        right = int(right2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        h, w = xi.shape[:2]
        if w/2 <= left:
            left=w/2
        if w/2 <= right:
            right=w/2
        for i in range(int((right/zure)+1)):
            src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
            dest = src.copy()
            dest[:,0] += i*zure # シフトするピクセル値
            affine = cv2.getAffineTransform(src, dest)
            image_shift_right = cv2.warpAffine(xi, affine, (w, h))
            output_xi.append(image_shift_right)
            output_yi.append(yi)
        for i in range(1,int((left/zure)+1)):
            src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
            dest2 = src.copy()    
            dest2[:,0] -= i*zure # シフトするピクセル値
            affine2 = cv2.getAffineTransform(src, dest2)
            image_shift_left = cv2.warpAffine(xi, affine2, (w, h))
            output_xi.append(image_shift_left)
            output_yi.append(yi)
    return output_xi,output_yi


# 二値化（ノーマル）
def threshold_normal(input_x,input_y,threshold_range=127):
    threshold_range2 = input('二値化の閾値：')
    if not threshold_range2 == "":
        threshold_range = int(threshold_range2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        ret1,xi[:,:,0] = cv2.threshold(xi[:,:,0], threshold_range, 255, cv2.THRESH_BINARY)
        ret2,xi[:,:,1] = cv2.threshold(xi[:,:,1], threshold_range, 255, cv2.THRESH_BINARY)
        ret3,xi[:,:,2] = cv2.threshold(xi[:,:,2], threshold_range, 255, cv2.THRESH_BINARY)
        output_xi.append(xi)
        output_yi.append(yi)
    return output_xi,output_yi


# 二値化（大津の二値化）
def threshold_otsu(input_x,input_y):
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        xi = xi.astype(np.uint8)
        ret1,xi[:,:,0] = cv2.threshold(xi[:,:,0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret2,xi[:,:,1] = cv2.threshold(xi[:,:,1], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret3,xi[:,:,2] = cv2.threshold(xi[:,:,2], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        output_xi.append(xi)
        output_yi.append(yi)
    return output_xi,output_yi


# 二値化（adaptive threshold_mean）
def adaptive_threshold_mean(input_x,input_y,block_size=9,c_value=0):
    block_size2 = input('近傍ピクセルサイズ(奇数)：')
    if not block_size2 == "":
        block_size = int(block_size2)
    c_value2 = input('平均または加重平均から引く値：')
    if not c_value2 == "":
        c_value = int(c_value2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        xi = xi.astype(np.uint8)
        xi[:,:,0] = cv2.adaptiveThreshold(xi[:,:,0],255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,block_size,c_value)
        xi[:,:,1] = cv2.adaptiveThreshold(xi[:,:,1],255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,block_size,c_value)
        xi[:,:,2] = cv2.adaptiveThreshold(xi[:,:,2],255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,block_size,c_value)
        output_xi.append(xi)
        output_yi.append(yi)
    return output_xi,output_yi


# 二値化（adaptive threshold_gaussian）
def adaptive_threshold_gaussian(input_x,input_y,block_size=9,c_value=0):
    block_size2 = input('近傍ピクセルサイズ(奇数)：')
    if not block_size2 == "":
        block_size = int(block_size2)
    c_value2 = input('平均または加重平均から引く値：')
    if not c_value2 == "":
        c_value = int(c_value2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        xi = xi.astype(np.uint8)
        xi[:,:,0] = cv2.adaptiveThreshold(xi[:,:,0],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,c_value)
        xi[:,:,1] = cv2.adaptiveThreshold(xi[:,:,1],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,c_value)
        xi[:,:,2] = cv2.adaptiveThreshold(xi[:,:,2],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,c_value)
        output_xi.append(xi)
        output_yi.append(yi)
    return output_xi,output_yi


# ぼかし（平均値フィルタ）
def blur(input_x,input_y,blur_strength=5):
    blur_strength2 = input('ぼかしの強度(フィルタサイズ)(奇数)：')
    if not blur_strength2 == "":
        blur_strength = int(blur_strength2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        img_blur = cv2.blur(xi,(blur_strength,blur_strength))
        output_xi.append(img_blur)
        output_yi.append(yi)
    return output_xi,output_yi


# ぼかし（ガウシアンフィルタ）
def gaussian_blur(input_x,input_y,blur_strength=5):   
    blur_strength2 = input('ぼかしの強度(フィルタサイズ)(奇数)：')
    if not blur_strength2 == "":
        blur_strength = int(blur_strength2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        img_blur = cv2.GaussianBlur(xi,(blur_strength,blur_strength),0)
        output_xi.append(img_blur)
        output_yi.append(yi)
    return output_xi,output_yi


# ぼかし（中央値フィルタ）
def median_blur(input_x,input_y,blur_strength=5):
    blur_strength2 = input('ぼかしの強度(フィルタサイズ)(奇数)：')
    if not blur_strength2 == "":
        blur_strength = int(blur_strength2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        img_blur = cv2.medianBlur(xi,blur_strength)
        output_xi.append(img_blur)
        output_yi.append(yi)
    return output_xi,output_yi


# ぼかし（バイラテラルフィルタ）
def bilateral_filter(input_x,input_y,diameter=9,sigma_color=75,sigma_space=75):
    diameter2 = input('周辺の何ピクセルを見るか(9程度)：')
    if not diameter2 == "":
        diameter = int(diameter2)
    sigma_color2 = input('色の分散値(75～100程度)：')
    if not sigma_color2 == "":
        sigma_color = int(sigma_color2)
    sigma_space2 = input('距離の分散値(75～100程度)：')
    if not sigma_space2 == "":
        sigma_space = int(sigma_space2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        img_blur = cv2.bilateralFilter(xi,diameter,sigma_color,sigma_space)
        output_xi.append(img_blur)
        output_yi.append(yi)
    return output_xi,output_yi


# ガウシアンノイズ付与
def gaussian_noise(input_x,input_y,sigma=15):
    sigma2 = input('ノイズの量(15程度)：')
    if not sigma2 == "":
        sigma = int(sigma2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        row,col,ch= xi.shape
        mean = 0
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        gauss = gauss.astype(np.uint8)
        noise_img = xi + gauss
        output_xi.append(noise_img)
        output_yi.append(yi)
    return output_xi,output_yi


# インパルスノイズ付与（ゴマ塩ノイズ）
def impulse_noise(input_x,input_y,amount=0.5):
    amount2 = input('ノイズの量(%)：')
    if not amount2 == "":
        amount = float(amount2)
    amount = amount/100
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        row,col,ch = xi.shape
        s_vs_p = 0.5
        noise_img = xi.copy()
        num_salt = np.ceil(amount * xi.size * s_vs_p)
        coords = [np.random.randint(0, j-1 , int(num_salt)) for j in xi.shape]
        noise_img[coords[:-1]] = (255,255,255)
        num_pepper = np.ceil(amount * xi.size * (1. - s_vs_p))
        coords = [np.random.randint(0, j-1 , int(num_pepper)) for j in xi.shape]
        noise_img[coords[:-1]] = (0,0,0)
        output_xi.append(noise_img)
        output_yi.append(yi)
    return output_xi,output_yi


# 積和演算による輝度，コントラストの調整
def brightness_contrast(input_x,input_y,alpha_lowerlimit=0.6,alpha_upperlimit=1.8,beta_lowerlimit=-50,beta_upperlimit=50,count=5):
    alpha_lowerlimit2 = input('コントラストの調整下限(0～3程度)：')
    if not alpha_lowerlimit2 == "":
        alpha_lowerlimit = float(alpha_lowerlimit2)
    alpha_upperlimit2 = input('コントラストの調整上限(0～3程度)：')
    if not alpha_upperlimit2 == "":
        alpha_upperlimit = float(alpha_upperlimit2)
    beta_lowerlimit2 = input('明るさの調整下限(-100～100程度)：')
    if not beta_lowerlimit2 == "":
        beta_lowerlimit = int(beta_lowerlimit2)
    beta_upperlimit2 = input('明るさの調整上限(-100～100程度)：')
    if not beta_upperlimit2 == "":
        beta_upperlimit = int(beta_upperlimit2)
    count2 = input('何枚ずつ生成しますか：')
    if not count2 == "":
        count = int(count2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        for j in range (count):
            alpha =np.random.uniform(alpha_lowerlimit, alpha_upperlimit)
            beta = np.random.randint(beta_lowerlimit, beta_upperlimit)
            dst = alpha * xi + beta
            img_out = np.clip(dst, 0, 255).astype(np.uint8)
            output_xi.append(img_out)
            output_yi.append(yi)
    return output_xi,output_yi


# ガンマ補正による輝度の調整
def gamma_correction(input_x,input_y,gamma_lowerlimit=0.4,gamma_upperlimit=2.4,count=3):
    gamma_lowerlimit2 = input('ガンマ下限：')
    if not gamma_lowerlimit2 == "":
        gamma_lowerlimit = float(gamma_lowerlimit2)
    gamma_upperlimit2 = input('ガンマ上限：')
    if not gamma_upperlimit2 == "":
        gamma_upperlimit = float(gamma_upperlimit2)
    count2 = input('何枚ずつ生成しますか：')
    if not count2 == "":
        count = int(count2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        for j in range (count):
            xi = xi.astype(np.uint8)
            gamma = np.random.uniform(gamma_lowerlimit, gamma_upperlimit)
            gamma =round(gamma,1)
            table = (np.arange(256) / 255) ** gamma * 255
            table = np.clip(table, 0, 255).astype(np.uint8)
            img_out = cv2.LUT(xi, table)
            output_xi.append(img_out)
            output_yi.append(yi)
    return output_xi,output_yi


# PCA Color Augmentation（輝度の調整自動）
def pca_color_augmentation(input_x,input_y,count=5):
    count2 = input('何枚ずつ生成しますか：')
    if not count2 == "":
        count = int(count2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        for j in range (count):
            assert xi.ndim == 3 and xi.shape[2] == 3
            img = xi.reshape(-1, 3).astype(np.float32)
            # 分散を計算
            ch_var = np.var(img, axis=0)
            # 分散の合計が3になるようにスケーリング
            scaling_factor = np.sqrt(3.0 / sum(ch_var))
            # 平均で引いてスケーリング
            img = (img - np.mean(img, axis=0)) * scaling_factor
            cov = np.cov(img, rowvar=False)
            lambd_eigen_value, p_eigen_vector = np.linalg.eig(cov)
            rand = np.random.randn(3) * 0.1
            delta = np.dot(p_eigen_vector, rand*lambd_eigen_value)
            delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]
            img_out = np.clip(xi + delta, 0, 255).astype(np.uint8)
            output_xi.append(img_out)
            output_yi.append(yi)
    return output_xi,output_yi


# 画像の色のチャネルをシフト
def channel_shift(input_x,input_y,shift_range=360,count=5):
    shift_range2 = input('色相をランダムでシフトする上限：')
    if not shift_range2 == "":
        shift_range = int(shift_range2)
    count2 = input('何枚ずつ生成しますか：')
    if not count2 == "":
        count = int(count2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        img_hsv = cv2.cvtColor(xi,cv2.COLOR_RGB2HSV) # 色空間をRGBからHSVに変換
        for j in range(count):
            h_deg = np.random.randint(0, shift_range)
            img_hsv[:,:,(0)] = img_hsv[:,:,(0)]+h_deg # 色相の計算
            img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB) # 色空間をHSVからRGBに変換
            output_xi.append(img_bgr)
            output_yi.append(yi)
    return output_xi,output_yi


# アフィン変換
def affine(input_x,input_y,x_range=0.25,y_range=0.25):
    x_range2 = input('元のx座標からどれだけ動かすか：')
    if not x_range2 == "":
        x_range = float(x_range2)
    y_range2 = input('元のy座標からどれだけ動かすか：')
    if not y_range2 == "":
        y_range = float(y_range2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        h, w = xi.shape[:2]
        x = w * x_range
        y = h * y_range
        x2 = w - x
        y2 = h - y
        pts1 = np.float32([[0,0],[w,0],[w,h]])
        pts2 = np.float32([[x,y],[w,0],[x2,y2]])
        M = cv2.getAffineTransform(pts1,pts2)
        img_affine = cv2.warpAffine(xi,M,(w,h))
        output_xi.append(img_affine)
        output_yi.append(yi)
    return output_xi,output_yi


# 射影変換
def syaei(input_x,input_y,syaei_range_left=0.2,syaei_range_right=0.8):
    syaei_range_left2 = input('台形の上底左側：')
    if not syaei_range_left2 == "":
        syaei_range_left = float(syaei_range_left2)
    syaei_range_right2 = input('台形の上底右側：')
    if not syaei_range_right2 == "":
        syaei_range_right = float(syaei_range_right2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        h, w = xi.shape[:2]
        left = w * syaei_range_left
        right = w * syaei_range_right
        pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        pts2 = np.float32([[left,0],[right,0],[0,h],[w,h]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        img_syaei = cv2.warpPerspective(xi,M,(w,h))
        output_xi.append(img_syaei)
        output_yi.append(yi)
    return output_xi,output_yi


# せん断（シアー）
def shear(input_x,input_y,sh_range=40,count=3):
    sh_range2 = input('せん断する際の角度の範囲：')
    if not sh_range2 == "":
        sh_range = int(sh_range2)
    count2 = input('何枚ずつ生成しますか：')
    if not count2 == "":
        count = int(count2)
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        xi = np.array(xi)
        datagen = image.ImageDataGenerator(shear_range=sh_range)
        x = xi[np.newaxis]
        gen = datagen.flow(x, batch_size=1)
        for j in range(count):
            batches = next(gen)
            img_shear = batches[0].astype(np.uint8)
            output_xi.append(img_shear)
            output_yi.append(yi)
    return output_xi,output_yi


# PCA白色化
def PCA_Whitening(input_x,input_y):
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        #最小値の計算
        min_n = xi.min(axis = None, keepdims = True)
        #最大値の計算
        max_n = xi.max(axis = None, keepdims = True)
        #正規化の計算
        xi = (xi - min_n) / (max_n - min_n)
        xi2 = xi.reshape(xi.shape[0],-1)
        mean = np.mean(xi2,axis=0)
        xi_ = xi2 - mean
        cov_mat = np.dot(xi_.T, xi_) / xi_.shape[0]
        try:
            A, L, _ = np.linalg.svd(cov_mat)
        except:
            AA = np.dot(cov_mat.T, cov_mat)
            A2, L2, _  = np.linalg.svd(AA)
            L = np.sqrt(L2)
            A = np.dot(cov_mat, _.T); A = np.dot(A, np.linalg.inv(np.diag(L)))
        PCA_mat = np.dot(np.diag(1. / (np.sqrt(L) + 1E-6)), A.T)
        xi_ = np.dot(xi_, PCA_mat)
        img_out = xi_.reshape(xi.shape)
        mean2 = np.mean(img_out, axis = None, keepdims = True)
        std2 = np.std(img_out, axis = None, keepdims = True,ddof = 0)
        img_out = (img_out - mean2) / std2
        img_out = img_out.astype(np.float32)
        output_xi.append(img_out)
        output_yi.append(yi)
    return output_xi,output_yi


# ZCA白色化
def ZCA_Whitening(input_x,input_y):
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        xi2 = xi.reshape(xi.shape[0],-1)
        mean = np.mean(xi2,axis=0)
        xi_ = xi2 - mean
        cov_mat = np.dot(xi_.T, xi_) / xi_.shape[0]
        try:
            A, L, _ = np.linalg.svd(cov_mat)
        except:
            AA = np.dot(cov_mat.T, cov_mat)
            A2, L2, _  = np.linalg.svd(AA)
            L = np.sqrt(L2)
            A = np.dot(cov_mat, _.T); A = np.dot(A, np.linalg.inv(np.diag(L)))
        ZCA_mat = np.dot(A, np.dot(np.diag(1. / (np.sqrt(L) + 1E-6)), A.T))
        xi_ = np.dot(xi_, ZCA_mat)
        img_out = xi_.reshape(xi.shape)
        img_out = img_out.astype(np.float32)
        output_xi.append(img_out)
        output_yi.append(yi)
    return output_xi,output_yi


# 標準化
def standardization(input_x,input_y):
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        mean = np.mean(xi, axis = None, keepdims = True)
        std = np.std(xi, axis = None, keepdims = True,ddof = 0)
        img_out = (xi - mean) / std
        img_out = img_out.astype(np.float32)
        output_xi.append(img_out)
        output_yi.append(yi)
    return output_xi,output_yi


# 正規化 最大値最小値
def normalization_max_min(input_x,input_y):
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        #最小値の計算
        min_n = xi.min(axis = None, keepdims = True)
        #最大値の計算
        max_n = xi.max(axis = None, keepdims = True)
        #正規化の計算
        img_out = (xi - min_n) / (max_n - min_n)
        img_out = img_out.astype(np.float32)
        output_xi.append(img_out)
        output_yi.append(yi)
    return output_xi,output_yi


# 正規化 255で割る
def normalization_255(input_x,input_y):
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        img_out = xi.astype('float32')/255
        output_xi.append(img_out)
        output_yi.append(yi)
    return output_xi,output_yi


# mean subtraction
def mean_subtraction(input_x,input_y):
    output_xi = []
    output_yi = []
    count = 0
    mean_r_sum = 0
    mean_g_sum = 0
    mean_b_sum = 0
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        r,g,b = cv2.split(xi)
        mean_r = np.mean(r)
        mean_r_sum += mean_r
        mean_g = np.mean(g)
        mean_g_sum += mean_g
        mean_b = np.mean(b)
        mean_b_sum += mean_b
        count += 1
    mean_r_all = mean_r_sum / count
    mean_g_all = mean_g_sum / count
    mean_b_all = mean_b_sum / count
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        r,g,b = cv2.split(xi)
        img_out_r = cv2.absdiff(r,mean_r_all)
        img_out_g = cv2.absdiff(g,mean_g_all)
        img_out_b = cv2.absdiff(b,mean_b_all)
        img_out = cv2.merge((img_out_r, img_out_g, img_out_b))
        output_xi.append(img_out)
        output_yi.append(yi)
    return output_xi,output_yi


# per pixel mean subtraction
def per_pixel_mean_subtraction(input_x,input_y):
    output_xi = []
    output_yi = []
    for i, xi in enumerate(input_x):
        yi = input_y[i]
        xi = xi.astype(np.uint8)
        r,g,b = cv2.split(xi)
        mean_r = np.mean(r)
        img_out_r = cv2.absdiff(r,mean_r)
        mean_g = np.mean(g)
        img_out_g = cv2.absdiff(g,mean_g)
        mean_b = np.mean(b)
        img_out_b = cv2.absdiff(b,mean_b)
        img_out = cv2.merge((img_out_r, img_out_g, img_out_b))
        output_xi.append(img_out)
        output_yi.append(yi)
    return output_xi,output_yi

