#!/bin/bash

# 定义日志文件路径（可根据需要修改）
LOG_FILE="/home/lvyao/git/med-ddpm/scripts/scripts_ablation.log"

# 清空旧日志（如果需要保留历史，可注释这行）
> $LOG_FILE

# 激活conda环境（确保conda命令可用，如果是bash需要先初始化conda）
echo "========================================" | tee -a $LOG_FILE
echo "开始激活conda环境: diffusermonai" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
source ~/.bashrc  # 初始化conda（根据你的shell调整，zsh则用~/.zshrc）
conda activate diffusermonai 2>&1 | tee -a $LOG_FILE

#for private

# # --------------------------psf 模块 --------------------------
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/ablation_config/ablation_no_psf.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/ablation_exp/configs/motion_no_psf.yaml 2>&1 | tee -a $LOG_FILE

# # --------------------------exclusion 模块 --------------------------
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/ablation_config/ablation_no_exclusion.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py  --config /home/lvyao/local/config_and_evaluate/ablation_exp/configs/motion_no_exclusion.yaml 2>&1 | tee -a $LOG_FILE

# --------------------------regular 模块 --------------------------
python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/ablation_config/ablation_no_regular.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py  --config /home/lvyao/local/config_and_evaluate/ablation_exp/configs/motion_no_regular.yaml 2>&1 | tee -a $LOG_FILE


# # --------------------------tweedie 模块 --------------------------
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/ablation_config/ablation_tweedie.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py  --config /home/lvyao/local/config_and_evaluate/ablation_exp/configs/motion_tweedie.yaml 2>&1 | tee -a $LOG_FILE
# 执行完成提示
echo -e "\n========================================" | tee -a $LOG_FILE
echo "所有命令执行完成！日志已保存到: $LOG_FILE" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE