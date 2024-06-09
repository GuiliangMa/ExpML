import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


# 读取图片的函数
def read_image(file_path):
    return mpimg.imread(file_path)


# 展示图片的函数
def display_images_in_grid(image_paths, grid_size=(3,2)):
    num_images = len(image_paths)
    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 8))

    for i, image_path in enumerate(image_paths):
        row = i // grid_size[1]
        col = i % grid_size[1]
        image = read_image(image_path)
        axs[row, col].imshow(image)
        axs[row, col].axis('off')  # 不显示坐标轴
        axs[row, col].set_title(os.path.basename(image_path), fontsize=8)

    # 隐藏未使用的子图
    for j in range(i + 1, grid_size[0] * grid_size[1]):
        fig.delaxes(axs.flatten()[j])

    plt.tight_layout()
    plt.savefig('../pic/CART/5tree.png')
    plt.show()


# 本地图片的路径列表（请替换为你自己的图片路径）
image_paths = [
    '../pic/CART/5CART0.png',
    '../pic/CART/5CART1.png',
    '../pic/CART/5CART2.png',
    '../pic/CART/5CART3.png',
    '../pic/CART/5CART4.png',
]

# 展示图片
display_images_in_grid(image_paths)
