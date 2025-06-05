import numpy as np
import gzip
import shutil

def create_pairs(images, labels):
    pairs = []
    pair_labels = []
    num_classes = 10  # MNIST 包含 10 个数字类别 (0-9)

    # 按数字分组
    digit_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    for idx, img1 in enumerate(images):
        label1 = labels[idx]

        # 创建正样本（标签为 1）：两张图片属于同一类别
        same_class_idx = np.random.choice(digit_indices[label1])
        img2 = images[same_class_idx]

        # 确保 img1 和 img2 形状为 (1, 28, 28)，并堆叠形成 (2, 28, 28)
        pair = np.stack([img1, img2], axis=0)  # stack along the channel dimension
        pairs.append(pair)
        pair_labels.append(1)

        # 创建负样本（标签为 0）：两张图片属于不同类别
        diff_class = (label1 + np.random.randint(1, num_classes)) % num_classes
        diff_class_idx = np.random.choice(digit_indices[diff_class])
        img2 = images[diff_class_idx]

        # 确保 img1 和 img2 形状为 (1, 28, 28)，并堆叠形成 (2, 28, 28)
        pair = np.stack([img1, img2], axis=0)  # stack along the channel dimension
        pairs.append(pair)
        pair_labels.append(0)

    # 转换为 numpy 数组并确保标签为浮动数类型
    return np.array(pairs), np.array(pair_labels, dtype=np.float32)

# 解压 .gz 文件
def decompress_file(gz_file, out_file):
    with gzip.open(gz_file, 'rb') as f_in:
        with open(out_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# 加载 MNIST 图像数据
def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

# 加载 MNIST 标签数据
def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels
