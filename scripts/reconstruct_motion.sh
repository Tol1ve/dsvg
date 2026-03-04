#!/bin/bash

# 定义日志文件路径（可根据需要修改）
LOG_FILE="/home/lvyao/git/med-ddpm/scripts/scripts_motion.log"

# 清空旧日志（如果需要保留历史，可注释这行）
> $LOG_FILE

# 激活conda环境（确保conda命令可用，如果是bash需要先初始化conda）
echo "========================================" | tee -a $LOG_FILE
echo "开始激活conda环境: diffusermonai" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
source ~/.bashrc  # 初始化conda（根据你的shell调整，zsh则用~/.zshrc）
conda activate diffusermonai 2>&1 | tee -a $LOG_FILE








python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E1.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E2.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E3.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E4.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E5.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E6.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E7.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E8.yaml 2>&1 | tee -a $LOG_FILE

python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E1.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E2.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E3.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E4.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E5.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E6.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E7.yaml 2>&1 | tee -a $LOG_FILE
python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E8.yaml 2>&1 | tee -a $LOG_FILE 
#for private



# # -------------------------- ours E1 模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 ours E1模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_init_daps_E1.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E1.yaml 2>&1 | tee -a $LOG_FILE
# # -------------------------- splatting E1模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 splatting E1模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/svr/DMpipe.py--config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_splatting_E1.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E1.yaml 2>&1 | tee -a $LOG_FILE

# # -------------------------- ours E2 模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 ours E2模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_init_daps_E2.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E2.yaml 2>&1 | tee -a $LOG_FILE


# # -------------------------- splatting E2模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 splatting E2模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/svr/DMpipe.py--config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_splatting_E2.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E2.yaml 2>&1 | tee -a $LOG_FILE


# # -------------------------- ours E3 模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 ours E3模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_init_daps_E3.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E3.yaml 2>&1 | tee -a $LOG_FILE
# # -------------------------- splatting E3模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 splatting E3模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/svr/DMpipe.py--config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_splatting_E3.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E3.yaml 2>&1 | tee -a $LOG_FILE


# # -------------------------- ours E4 模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 ours E4模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_init_daps_E4.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E4.yaml 2>&1 | tee -a $LOG_FILE
# # -------------------------- splatting E4模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 splatting E4模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/svr/DMpipe.py--config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_splatting_E4.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E4.yaml 2>&1 | tee -a $LOG_FILE

# # -------------------------- ours E5 模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 ours E5模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_init_daps_E5.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E5.yaml 2>&1 | tee -a $LOG_FILE
# # -------------------------- splatting E5模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 splatting E5模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/svr/DMpipe.py--config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_splatting_E5.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E5.yaml 2>&1 | tee -a $LOG_FILE

# # -------------------------- ours E6 模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 ours E6模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_init_daps_E6.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E6.yaml 2>&1 | tee -a $LOG_FILE
# # -------------------------- splatting E6模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 splatting E6模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/svr/DMpipe.py--config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_splatting_E6.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E6.yaml 2>&1 | tee -a $LOG_FILE


# # -------------------------- ours E7 模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 ours E7模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_init_daps_E7.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E7.yaml 2>&1 | tee -a $LOG_FILE
# # -------------------------- splatting E7模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 splatting E7模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/svr/DMpipe.py--config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_splatting_E7.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E7.yaml 2>&1 | tee -a $LOG_FILE

# # -------------------------- ours E8 模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 ours E8模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/med-ddpm/DMpipe.py --config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_init_daps_E8.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_init_daps_E8.yaml 2>&1 | tee -a $LOG_FILE
# # -------------------------- splatting E8模块 --------------------------
# echo -e "\n========================================" | tee -a $LOG_FILE
# echo "开始执行 splatting E8模块命令" | tee -a $LOG_FILE
# echo "========================================" | tee -a $LOG_FILE
# python /home/lvyao/git/svr/DMpipe.py--config /home/lvyao/git/med-ddpm/config/diff_motion_config/motion_splatting_E8.yaml 2>&1 | tee -a $LOG_FILE
# python /home/lvyao/local/evaluate_folder2.py --config /home/lvyao/local/config_and_evaluate/motion_exp/configs/motion_splatting_E8.yaml 2>&1 | tee -a $LOG_FILE