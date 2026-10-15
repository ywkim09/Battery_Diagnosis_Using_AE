#!/bin/bash
#SBATCH -A uppmax2026-1-226
#SBATCH -p pelle            # 혹은 이 줄을 아예 삭제해도 pelle가 기본값으로 적용됩니다.
#SBATCH -c 4                # Pelle 문서 권장: 멀티코어 지정 시 -c 사용
#SBATCH -t 00:30:00
#SBATCH -J my_task
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err


# 로그 디렉토리가 없을 경우 생성
mkdir -p logs

# 1. 필요한 모듈 로드
module load Miniconda3/25.7.0-2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./mini_env
# 2. 작업 제출 디렉토리로 이동
cd $SLURM_SUBMIT_DIR

# 3. 작업 실행
echo "Job started on $(hostname) at $(date)"

# 실행할 프로그램/스크립트 작성
python main.py

echo "Job completed at $(date)"