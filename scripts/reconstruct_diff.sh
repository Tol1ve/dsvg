#!/bin/bash

# 定义日志文件路径（可根据需要修改）
LOG_FILE="/home/lvyao/git/med-ddpm/scripts/scripts_diff.log"

# 清空旧日志（如果需要保留历史，可注释这行）
> $LOG_FILE

### for public
# -------------------------- restart模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 restart模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/public_config/public_init_daps.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_init_daps.yaml 2>&1 | tee -a $LOG_FILE
# #-------------------------- no_restart 模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 no_restart模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/public_config/public_no_restart_config.yaml 2>&1 | tee -a $LOG_FILE

# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/public_exp/configs/public_diff_no_restart.yaml 2>&1 | tee -a $LOG_FILE


# #for private
# -------------------------- restart 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 restart模块命令" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/private_config/private_init_daps_config.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/private_exp/configs/private_init_daps.yaml
# -------------------------- no_restart 模块 --------------------------
echo -e "\n========================================" | tee -a $LOG_FILE
echo "开始执行 no_restart模块命令" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/private_config/private_no_restart_config.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config  /home/lvyao/local/config_and_evaluate/private_exp/configs/private_diff_no_restart.yaml




# 执行完成提示
echo -e "\n========================================" | tee -a $LOG_FILE
echo "所有命令执行完成！日志已保存到: $LOG_FILE" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE