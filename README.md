# RL para controle de semáforos - Belém

Projeto final IC 2026.2 - UFPA

## Instalação

### 1. Instalar o conda (se ainda não tiver)
Precisa de algum conda (Miniconda é o mais leve). No Linux:
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Segue o instalador (aceita os padrões), fecha e abre o terminal de novo pro `conda` funcionar.
Windows/Mac: baixa o instalador em https://docs.conda.io/en/latest/miniconda.html.

### 2. Criar e ativar o ambiente
```bash
conda env create -f environment.yml
conda activate belem
```

### 3. Configurar o SUMO_HOME
O `sumo_rl` exige a variável `SUMO_HOME` apontando pra pasta do SUMO -- mas como o SUMO aqui
vem via pip (`eclipse-sumo`), ele fica dentro do próprio ambiente conda, não em `/usr/share/sumo`
como numa instalação via apt. Sem isso, qualquer script dá `ImportError: Please declare the
environment variable 'SUMO_HOME'`. Configura pra ativar sozinho toda vez que entrar no ambiente:
```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" << 'EOF'
export SUMO_HOME="$CONDA_PREFIX/lib/python3.11/site-packages/sumo"
export PROJ_DATA="$CONDA_PREFIX/lib/python3.11/site-packages/sumo/data/proj"
export PROJ_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/sumo/data/proj"
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh" << 'EOF'
unset SUMO_HOME
unset PROJ_DATA
unset PROJ_LIB
EOF
```
(`PROJ_DATA`/`PROJ_LIB` evitam uns avisos irrelevantes de `pj_obj_create: Cannot find proj.db`
-- o SUMO usa uma versão antiga da lib PROJ que só lê a variável `PROJ_LIB`.)

Depois, sai e reativa o ambiente pra pegar as variáveis novas:
```bash
conda deactivate
conda activate belem
```

### 4. Testar
```bash
python -c "from sumo_rl import SumoEnvironment; print('tudo certo')"
```

## Contrato de arquivos (não mudar os nomes)
- cenarios/almirante/belem.net.xml   rede
- cenarios/almirante/rotas.rou.xml   demanda
- outputs/                           tudo que é gerado (baselines, modelos, csv, db)

## Estrutura de pastas
- cenarios/almirante/   rede e demanda (contrato acima)
- baselines/            controle de tempo fixo e Max Pressure (rodar_maxpressure.py, analisar_baseline.py)
- reward/               funções de recompensa (rewards.py), compartilhadas por PPO e DQN
- train/config.yaml     hiperparâmetros, espaço de busca do Optuna e orçamento de treino --
                        editar aqui em vez de mexer direto nos scripts (train/config.py carrega)
- train/callbacks.py    callback compartilhado que loga métricas de sistema no tensorboard
- train/ppo/            treino e otimização com PPO (otimizar_optuna.py, treinar_ppo.py,
                        treinar_agentes_independentes.py, pipeline_optuna_independentes.py,
                        treinar_modelos_finais.py, comparar_recompensas.py, ambiente_rl.py)
- train/dqn/            treino e otimização com DQN, mesma estrutura do train/ppo/
                        (treinar_agentes_independentes.py, pipeline_optuna_independentes.py,
                        treinar_modelos_finais.py)
- utils/                scripts auxiliares (listar_tls.py, visualizar_simulacao.py -- abre o SUMO-GUI)
- outputs/              tudo que é gerado (csv, db, modelos, tensorboard, resultados)
- rodar_pipeline.sh     roda os Optuna e os treinos finais de PPO e DQN em paralelo

Os scripts sempre rodam a partir da raiz do projeto (ex: `python train/ppo/otimizar_optuna.py`),
já que os caminhos pra cenarios/ e outputs/ são relativos à raiz, não à pasta do script.

## Divisão
- A: rede/OSM
- B: demanda + Max Pressure
- C: agentes RL
- D: Optuna + avaliação
