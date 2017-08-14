import openslide
import numpy as np
from pylab import *

level = 6

# 读取包含有肿瘤区域的大图（全切片病理图像）
origin_images_path = "/atlas/home/zwpeng/paper_rebuild/camelyon/train/tumor/origin_images/Tumor_005.tif"
origin_slide = openslide.open_slide(origin_images_path)

# 读取该肿瘤区域的标注图
annotation_images_path = "/atlas/home/zwpeng/paper_rebuild/camelyon/train/tumor/annotation_images/Tumor_005_Mask.tif"
mask_slide = openslide.open_slide(annotation_images_path)

origin_size = origin_slide.dimensions
origin_widths = origin_size[0]
origin_heights = origin_size[1]

object_size = origin_slide.level_dimensions[level]
object_widths = int(object_size[0])
object_heights = object_size[1]

# 方法三：通过获取每一块区域的像素值R G B各自的平均数，然后相减，设置一个阈值，将噪点（墨迹）和有效区　分开

rgb_list_y = list()
rgb_list_x = list()
rgb_var_x = []
rgb_var_y = []
rgb_var_xi = []
rgb_var_yi = []

# 寻找有效区域的y值、高度
for k in range(100):
    slide = origin_slide.read_region((0, k*origin_heights//100), level, (object_widths, object_heights//50)) 
    slide_arr = array(slide.convert("RGB"))
    arrR = np.mean(slide_arr[:,:,:1])
    arrG = np.mean(slide_arr[:,:,1:2])
    arrB = np.mean(slide_arr[:,:,2:3])
    rgb_list_y.append((arrR,arrG,arrB))
for i,rgbVar in enumerate(rgb_list_y):
    rgb_var_y.append(np.var(rgbVar))
    if np.var(rgbVar)>=1:
        rgb_var_yi.append(i)

print(rgb_var_yi)
effective_y = min(rgb_var_yi)*origin_heights//100        #有效区域的左上顶点y坐标找到了
effective_heights = (max(rgb_var_yi)-min(rgb_var_yi))*origin_heights//100 + origin_heights//50  #有效区域的高度也出来了
print("有效区域的ｙ值是：%d" %effective_y, "有效区域的高度是：%d" %effective_heights)

# 寻找有效区域的x值、宽度
for j in range(100):
    slide = origin_slide.read_region((j*origin_widths//100, effective_y), level, 
                                      (object_widths//50, effective_heights//62))     # 循环顺序读取50宽的区域
#     slide = origin_slide.read_region((j*origin_widths//100, 0), level, 
#                                       (object_widths//50, object_heights))     # 循环顺序读取50宽的区域
    
    slide_arr = array(slide.convert("RGB"))
    arrR = np.mean(slide_arr[:,:,:1])
    arrG = np.mean(slide_arr[:,:,1:2])
    arrB = np.mean(slide_arr[:,:,2:3])
    rgb_list_x.append((arrR,arrG,arrB))
for i,rgbVar in enumerate(rgb_list_x):
    rgb_var_x.append(np.var(rgbVar))
    if np.var(rgbVar)>=2:
        rgb_var_xi.append(i)

print(rgb_var_xi)
effective_x = min(rgb_var_xi)*origin_widths//100        # 有效区域的左上顶点y坐标找到了
effective_widths = (max(rgb_var_xi) - min(rgb_var_xi))*origin_widths//100 + origin_widths//50  # 有效区域的宽度也出来了
print("有效区域的ｘ值是：%d" %effective_x, "有效区域的宽度是：%d" %effective_widths)
# plt.plot(range(100), rgb_var_y[:100], label='rgb_var_curve')
# plt.plot(range(100), rgb_var_x[:100], label='rgb_var_curve')
# plt.legend()
# plt.show()

# 有效区域（感兴趣区域）
effective_area = (effective_x, effective_y)
effective_area_size = (effective_widths, effective_heights)

mask_level = 7

# level0　的尺寸
mask_size = mask_slide.dimensions
mask_widths = mask_size[0]
mask_heights = mask_size[1]
# level7 的尺寸
mask_level_size = mask_slide.level_dimensions[mask_level]
mask_level_widths = mask_level_size[0]
mask_level_heights = mask_level_size[1]

mask_level_slide = mask_slide.read_region((0, 0), mask_level, (mask_level_widths, mask_level_heights))
mask_level_slide_gray = mask_level_slide.convert("L")
mask_level_slide_arr = array(mask_level_slide_gray)

mask_y, mask_x = nonzero(mask_level_slide_arr)  # 因为mask是黑白图，只需直接获得非零像素的坐标
# mask_x, mask_y

# 有效区域（感兴趣区域）level7上的
tumor_leftup_x = (min(mask_x)-1) * int(mask_slide.level_downsamples[7])
tumor_leftup_y = (min(mask_y)-1) * int(mask_slide.level_downsamples[7])
tumor_rightdown_x = (max(mask_x)+1) * int(mask_slide.level_downsamples[7])
tumor_rightdown_y = (max(mask_y)+1) * int(mask_slide.level_downsamples[7])

from PIL.Image import Image
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

widths = 299
heights = 299

def data_generator(widths=299,heights=299):
    while True:
        random_num = np.random.random(1)
        print(random_num)

        if random_num > 0.5:
            # 定义随机坐标,一定要取到一张含有tumor的图片
            random_x = np.random.randint(tumor_leftup_x, tumor_rightdown_x - widths)    # mask上的tumor有效区的起点和终点
            random_y = np.random.randint(tumor_leftup_y, tumor_rightdown_y - heights)
            print("取tumor随机点坐标是：%d,%d"%(random_x,random_y))
            random_img_mask = mask_slide.read_region((random_x,random_y),0,(widths,heights))
            random_img_mask_arr = array(random_img_mask.convert("L"))
            random__img_y, random_img_x = nonzero(random_img_mask_arr)
            while len(random_img_x)==0:
                random_x = np.random.randint(tumor_leftup_x, tumor_rightdown_x - widths)
                random_y = np.random.randint(tumor_leftup_y, tumor_rightdown_y - heights)
                print("取tumor随机点坐标是：%d,%d"%(random_x,random_y))
                random_img_mask = mask_slide.read_region((random_x,random_y),0,(widths,heights))
                random_img_mask_arr = array(random_img_mask.convert("L"))
                random__img_y, random_img_x = nonzero(random_img_mask_arr)

            #*********************上面这个 while 循环结束后，就产生了一个合格的坐标点*********************#
            random_img = origin_slide.read_region((random_x,random_y),0,(widths,heights))
    #         plt.imshow(random_img)
    #         plt.show()
            #***接下来就给他贴标签，并处理成训练所需的数据结构***#
            random_img_arr = array(random_img.convert("RGB"))
            x = np.expand_dims(random_img_arr, axis=0)/255.
            y = to_categorical(0,2)    
        else:
            # 定义随机坐标，一定要取到一张不含有tumor的normal图片
            random_x = np.random.randint(effective_x,effective_x+effective_widths-widths)   # 大图上,nomal有效区的起点和终点
            random_y = np.random.randint(effective_y,effective_y+effective_heights-heights)
            print("取normal随机点坐标是：%d,%d"%(random_x,random_y))
            random_img_mask = mask_slide.read_region((random_x,random_y),0,(widths,heights))
            random_img_mask_arr = array(random_img_mask.convert("L"))
            random__img_y, random_img_x = nonzero(random_img_mask_arr)
            while len(random_img_x) != 0:
                random_x = np.random.randint(effective_x,effective_x+effective_widths-widths)
                random_y = np.random.randint(effective_y,effective_y+effective_heights-heights)
                print("取normal随机点坐标是：%d,%d" %(random_x,random_y))
                random_img_mask = mask_slide.read_region((random_x,random_y),0,(widths,heights))
                random_img_mask_arr = array(random_img_mask.convert("L"))
                random__img_y, random_img_x = nonzero(random_img_mask_arr)

            #*********************上面这个 while 循环结束后，就产生了一个合格的坐标点*********************#
            random_img = origin_slide.read_region((random_x,random_y),0,(widths,heights))
    #         plt.imshow(random_img)
    #         plt.show()
            #***接下来就给他贴标签，并处理成训练所需的数据结构***#
            random_img_arr = array(random_img.convert("RGB"))
            x = np.expand_dims(random_img_arr, axis=0)/255.
            y = to_categorical(1,2) 
        yield (x,y)

