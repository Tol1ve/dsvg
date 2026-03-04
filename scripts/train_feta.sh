conda activate meddpm

# 定义工作目录
cd /home/lvyao/git/med-ddpm

# 检查是否传入了 CUDA_VISIBLE_DEVICES 参数
if [ -z "$1" ]; then
    echo "No GPU specified. Defaulting to GPU 0."
    export CUDA_VISIBLE_DEVICES=0  # 默认使用GPU 0
else
    export CUDA_VISIBLE_DEVICES=$1  # 使用传入的 GPU
fi
# 运行Python脚本
python3 /home/lvyao/git/med-ddpm/train.py \
    --with_condition \
    --inputfolder '' \
    --crldatasetfolder /home/lvyao/git/dataset/CRL_reaffined \
    --chndatasetfolder /home/lvyao/git/dataset/CHN-fetal-brain-atlas \
    --kcldatasetfolder '' \
    --timesteps 1000 \
    --batchsize 1 \
    --save_and_sample_every 25000 \
    --num_channels 64 \
    --num_res_blocks 1 \
    --evalwhiletrain \
    --result "/home/lvyao/git/med-ddpm/results/results_5_CRLandCHN/1222" \
    -nz \
    --downsample \
    --padded \
    --use_cfg \
#   -r /home/lvyao/git/med-ddpm/results/results_4_Timestep1000/1203/model-2.pt


# resized版本
# python3 /home/lvyao/git/med-ddpm/train.py \
#     --with_condition \
#     --inputfolder /home/lvyao/git/dataset/feta_2.2_reaffined \
#     --crldatasetfolder /home/lvyao/git/dataset/CRL_reaffined \
#     --chndatasetfolder /home/lvyao/git/dataset/CHN-fetal-brain-atlas \
#     --kcldatasetfolder /home/lvyao/git/dataset/kcl \
#     --timesteps 250 \
#     --batchsize 1 \
#     --save_and_sample_every 25000 \
#     --num_channels 64 \
#     --num_res_blocks 1 \
#     --evalwhiletrain \
#     --result "/home/lvyao/git/med-ddpm/results/results_2_resized/1205" \
#     -nz \
#     --resized \
#     -r /home/lvyao/git/med-ddpm/results/model_128.pt
    #--use_cfg \