# import nibabel as nib
# import numpy as np
# import os
# file_path='/home/lvyao/git/dataset/feta_2.2_mial/sub-002/anat/sub-002_rec-mial_T2w_norm_reg.nii.gz'
# result_folder='/home/lvyao/git/dataset/test'
# nifti_img = nib.load(file_path)
# img=nifti_img.get_fdata()
# original_affine = nifti_img.affine
# img=np.flip(img, axis=(0, 1))

# # 设置新的 affine 矩阵 (没有翻转)
# new_affine = np.copy(original_affine)
# new_affine[0, 0] = abs(new_affine[0, 0])  # Ensure positive scaling on X axis
# new_affine[1, 1] = abs(new_affine[1, 1])  # Ensure positive scaling on Y axis
# new_affine[2, 2] = abs(new_affine[2, 2]) 
# nifti_img = nib.Nifti1Image(img, affine=new_affine)
# nib.save(nifti_img,os.path.join(result_folder,'feta'))
import os
import nibabel as nib
import numpy as np
import shutil

def process_image(input_path, output_path):
    """处理单个图像：翻转并修改 affine 矩阵"""
    
    # 读取图像
    img = nib.load(input_path)
    img_data = img.get_fdata()
    original_affine = img.affine

    # 反转 X 和 Y 轴
    img_data_flipped =img_data.transpose(2,1,0)

    # 设置新的 affine 矩阵 (没有翻转)
    new_affine = np.copy(original_affine)
    new_affine[0, 0] = abs(new_affine[0, 0])  # Ensure positive scaling on X axis
    new_affine[1, 1] = abs(new_affine[1, 1])  # Ensure positive scaling on Y axis
    new_affine[2, 2] = abs(new_affine[2, 2])  # Ensure positive scaling on Z axis

    # 使用新 affine 保存图像
    new_img = nib.Nifti1Image(img_data_flipped, new_affine)
    nib.save(new_img, output_path)
    print(f"Processed and saved: {output_path}")

def process_folder(input_folder, output_folder):
    """遍历文件夹，处理所有的 .nii.gz 文件并保持目录结构"""
    
    # 遍历输入文件夹
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.nii.gz') or file.endswith('.nii'):
                input_path = os.path.join(root, file)
                
                # 计算输出文件夹中的相对路径
                rel_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, rel_path)
                
                # 如果输出目录不存在，创建它
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # 计算输出文件路径
                output_path = os.path.join(output_dir, file)
                
                # 处理图像
                process_image(input_path, output_path)

if __name__ == "__main__":
    # 输入和输出文件夹路径
    input_folder = '/home/lvyao/local/DAPS_EXP/sim_exp/dpcp_large_1216'  # 替换为实际的输入文件夹路径
    output_folder = '/home/lvyao/local/DAPS_EXP/sim_exp/dpcp_large_1216_reffined'  # 替换为实际的输出文件夹路径

    
    # 处理文件夹中的所有文件
    process_folder(input_folder, output_folder)
    print("Processing complete.")
