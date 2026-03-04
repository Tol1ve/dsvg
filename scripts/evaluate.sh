#!/bin/bash

# 定义日志文件路径（可根据需要修改）
LOG_FILE="/home/lvyao/git/med-ddpm/scripts/evaluate.log"

# 清空旧日志（如果需要保留历史，可注释这行）
> $LOG_FILE

### for public

# -------------------------- svrtk 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 svrtk模块命令" | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_compare_svrtk.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- splatting模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 splatting模块命令" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_compare_splatting.yaml 2>&1 | tee -a $LOG_FILE
# -------------------------- svort 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 svort模块命令" | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_compare_svort.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- no_restart 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 no_restart模块命令" | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_diff_no_restart.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- fide 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 fide模块命令" | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_diff_fide.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- daps 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 daps模块命令" | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_diff_daps.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- ours 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 ours模块命令" | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_init_daps.yaml 2>&1 | tee -a $LOG_FILE

### for private

# -------------------------- svrtk 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 svrtk模块命令" | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/private_exp/configs/private_compare_svrtk.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- splatting模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 splatting模块命令" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/private_exp/configs/private_compare_splatting.yaml 2>&1 | tee -a $LOG_FILE
# -------------------------- svort 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 svort模块命令" | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/private_exp/configs/private_compare_svort.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- no_restart 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 no_restart模块命令" | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/private_exp/configs/private_diff_no_restart.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- fide 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 fide模块命令" | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/private_exp/configs/private_diff_fide.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- daps 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 daps模块命令" | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/private_exp/configs/private_diff_daps.yaml 2>&1 | tee -a $LOG_FILE

# -------------------------- ours 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 ours模块命令" | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/private_exp/configs/private_init_daps.yaml 2>&1 | tee -a $LOG_FILE
# 执行完成提示
echo -e "\n========================================" | tee -a $LOG_FILE
echo "所有命令执行完成！日志已保存到: $LOG_FILE" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE


