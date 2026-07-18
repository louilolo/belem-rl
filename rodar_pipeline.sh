#!/usr/bin/env bash
# Roda o pipeline completo (Optuna -> modelos finais) pros dois algoritmos (PPO e DQN).
#
# Fase 1: os dois Optuna (pipeline_optuna_independentes.py) rodam AO MESMO TEMPO.
# Só depois que os dois terminarem:
# Fase 2: os dois treinos finais (treinar_modelos_finais.py) rodam AO MESMO TEMPO.
#
# Roda 2 pipelines completos em paralelo -- espera usar bastante CPU/RAM (cada um sobe
# sua própria simulação SUMO). Uso: ./rodar_pipeline.sh (a partir de qualquer diretório).
set -uo pipefail

DIR_RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR_RAIZ"

# Garante SUMO_HOME/PROJ_DATA/PROJ_LIB mesmo se o script rodar fora de um shell com o
# conda já ativado (ex: cron, execução não-interativa) -- esses vêm dos hooks em
# miniconda3/envs/belem/etc/conda/activate.d/env_vars.sh, então precisamos ativar o env.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate belem

# Usa o python do env pelo caminho absoluto (via $CONDA_PREFIX), não o `python` do PATH --
# nesta máquina outro shell hook (esp-idf) reordena o PATH depois do conda activate e faz
# `python` apontar pro venv errado (sem sumo_rl/stable_baselines3/optuna instalados).
PYTHON="$CONDA_PREFIX/bin/python"

# Sem isso, com stdout indo pra um arquivo (não um terminal), o Python usa buffer cheio em
# vez de por linha -- os prints ficam "presos" e só aparecem no log quando o buffer enche
# ou o processo termina (a saída bruta do SUMO não sofre disso, por isso ela sempre aparece
# na hora enquanto os prints do próprio script parecem sumir ou vir fora de ordem).
export PYTHONUNBUFFERED=1

mkdir -p outputs/logs

echo "=========================================="
echo " FASE 1: Optuna (PPO + DQN em paralelo)"
echo "=========================================="
"$PYTHON" train/ppo/pipeline_optuna_independentes.py > outputs/logs/optuna_ppo.log 2>&1 &
PID_PPO=$!
"$PYTHON" train/dqn/pipeline_optuna_independentes.py > outputs/logs/optuna_dqn.log 2>&1 &
PID_DQN=$!

wait "$PID_PPO"; STATUS_PPO=$?
wait "$PID_DQN"; STATUS_DQN=$?

if [ "$STATUS_PPO" -ne 0 ] || [ "$STATUS_DQN" -ne 0 ]; then
    echo "Fase 1 (Optuna) falhou -- ppo=$STATUS_PPO dqn=$STATUS_DQN."
    echo "Veja outputs/logs/optuna_ppo.log e outputs/logs/optuna_dqn.log"
    exit 1
fi
echo "Fase 1 concluída. Logs em outputs/logs/optuna_{ppo,dqn}.log"

echo "=========================================="
echo " FASE 2: Modelos finais (PPO + DQN em paralelo)"
echo "=========================================="
"$PYTHON" train/ppo/treinar_modelos_finais.py > outputs/logs/final_ppo.log 2>&1 &
PID_PPO=$!
"$PYTHON" train/dqn/treinar_modelos_finais.py > outputs/logs/final_dqn.log 2>&1 &
PID_DQN=$!

wait "$PID_PPO"; STATUS_PPO=$?
wait "$PID_DQN"; STATUS_DQN=$?

if [ "$STATUS_PPO" -ne 0 ] || [ "$STATUS_DQN" -ne 0 ]; then
    echo "Fase 2 (modelos finais) falhou -- ppo=$STATUS_PPO dqn=$STATUS_DQN."
    echo "Veja outputs/logs/final_ppo.log e outputs/logs/final_dqn.log"
    exit 1
fi

echo "=========================================="
echo " PIPELINE COMPLETO"
echo "=========================================="
echo "Resultados:    outputs/modelos_finais_ppo.csv, outputs/modelos_finais_dqn.csv"
echo "Modelos:       outputs/modelos/{ppo,dqn}/{com_optuna,sem_optuna}/{reward}/"
echo "Tensorboard:   outputs/tensorboard/{ppo,dqn}/"
echo "Logs:          outputs/logs/"
