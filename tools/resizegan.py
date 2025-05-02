import os
from PIL import Image

# 输入和输出路径
input_folder = '/a800data1/wangshijie/Domain-Adaptive-Faster-RCNN-PyTorch-master/datasets/COCO_DIOR/ganimages/dior_image/train'  # 图片所在的文件夹
output_folder = '/a800data1/wangshijie/Domain-Adaptive-Faster-RCNN-PyTorch-master/datasets/COCO_DIOR/ganimages/dior_image_resize/train'  # 调整大小后的图片保存路径

# 目标图片大小
target_size = (800, 800)  # 宽度和高度

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 获取所有图片文件
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# 遍历所有图片并调整大小
for image_file in image_files:
    try:
        # 读取图片
        img_path = os.path.join(input_folder, image_file)
        img = Image.open(img_path)
        
        # 调整图片大小
        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)  # 使用抗锯齿
        
        # 保存调整大小后的图片
        output_path = os.path.join(output_folder, image_file)
        resized_img.save(output_path)
        
        print(f"已处理并保存: {image_file}")
    except Exception as e:
        print(f"处理图片 {image_file} 时出错: {e}")

print("所有图片处理完成！")