# RL para controle de semáforos - Belém

Projeto final IC 2026.2 - UFPA

## Setup
conda env create -f environment.yml
conda activate belem

## Contrato de arquivos (não mudar os nomes)
- cenarios/almirante/belem.net.xml   rede
- cenarios/almirante/rotas.rou.xml   demanda
- outputs/                           tudo que é gerado (baselines, modelos, csv, db)

## Estrutura de pastas
- cenarios/almirante/   rede e demanda (contrato acima)
- baselines/            controle de tempo fixo e Max Pressure (rodar_maxpressure.py, analisar_baseline.py)
- reward/               funções de recompensa (rewards.py), compartilhadas por PPO e DQN
- train/ppo/            treino e otimização com PPO (otimizar_optuna.py, treinar_ppo.py,
                        treinar_agentes_independentes.py, pipeline_optuna_independentes.py,
                        comparar_recompensas.py, ambiente_rl.py)
- train/dqn/            treino e otimização com DQN, mesma estrutura do train/ppo/
                        (treinar_agentes_independentes.py, pipeline_optuna_independentes.py)
- utils/                scripts auxiliares (listar_tls.py)
- outputs/              tudo que é gerado (csv, db, modelos, resultados)

Os scripts sempre rodam a partir da raiz do projeto (ex: `python train/ppo/otimizar_optuna.py`),
já que os caminhos pra cenarios/ e outputs/ são relativos à raiz, não à pasta do script.

## Divisão
- A: rede/OSM
- B: demanda + Max Pressure
- C: agentes RL
- D: Optuna + avaliação
