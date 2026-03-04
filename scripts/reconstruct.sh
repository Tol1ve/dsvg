#!/bin/bash

# 定义日志文件路径（可根据需要修改）
LOG_FILE="/home/lvyao/git/med-ddpm/scripts/scripts.log"

# 清空旧日志（如果需要保留历史，可注释这行）
> $LOG_FILE

# # 激活conda环境（确保conda命令可用，如果是bash需要先初始化conda）
# echo "========================================" | tee -a $LOG_FILE
# echo "开始激活conda环境: diffusermonai" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# source ~/.bashrc  # 初始化conda（根据你的shell调整，zsh则用~/.zshrc）
# conda activate diffusermonai 2>&1 | tee -a $LOG_FILE

# #for private

# # -------------------------- ours 模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 ours 模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/private_config/private_init_daps_config.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/private_exp/configs/private_init_daps.yaml 2>&1 | tee -a $LOG_FILE

# # -------------------------- fide 模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 fide 模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/private_config/private_fide_config.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/private_exp/configs/private_diff_fide.yaml 2>&1 | tee -a $LOG_FILE

# # -------------------------- dpas 模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 dpas 模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/private_config/private_daps_config.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/private_exp/configs/private_diff_daps.yaml 2>&1 | tee -a $LOG_FILE

# # -------------------------- splatting 模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 splatting 模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/svr/DMpipe.py --config /home/lvyao/git/med-ddpm/config/private_config/private_splatting_config.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/private_exp/configs/private_compare_splatting.yaml 2>&1 | tee -a $LOG_FILE

# # -------------------------- svort 模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 svort 模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/private_config/private_svort_config.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/private_exp/configs/private_compare_svort.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- svrtk 模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 svrtk 模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/utils/baseline.py --config /home/lvyao/git/med-ddpm/config/private_config/private_svrtk_config.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/private_exp/configs/private_compare_svrtk.yaml 2>&1 | tee -a $LOG_FILE


### for public
# -------------------------- ours 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 ours 模块命令" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/public_config/public_init_daps.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_init_daps.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- fide 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 fide 模块命令" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
#python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/public_config/public_fide_config.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_diff_fide.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- dpas 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 dpas 模块命令" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/public_config/public_daps_config.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_diff_daps.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- splatting 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 splatting 模块命令" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/svr/DMpipe.py --config /home/lvyao/git/med-ddpm/config/public_config/public_splatting_config.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_compare_splatting.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- svort 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 svort 模块命令" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/public_config/public_svort_config.yaml  2>&1 | tee -a $LOG_FILE
#python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_compare_svort.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- svrtk 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 svrtk 模块命令" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/utils/baseline.py --config /home/lvyao/git/med-ddpm/config/public_config/public_svrtk_config.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_compare_svrtk.yaml 2>&1 | tee -a $LOG_FILE


# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 dmplug 模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/public_config/public_dmplug_config.yaml
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_dmplug.yaml

echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 dmplug 模块命令" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/private_config/private_dmplug_config.yaml
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/private_exp/configs/private_dmplug.yaml
# 执行完成提示
echo -e "\n========================================" | tee -a $LOG_FILE
echo "所有命令执行完成！日志已保存到: $LOG_FILE" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE