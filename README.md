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

## Como rodar

Sempre com o ambiente ativado (`conda activate belem`), a partir da raiz do projeto.

### 1. (Opcional) Baselines
Pra ter um comparativo antes de treinar os agentes de RL:
```bash
python baselines/rodar_maxpressure.py     # roda a simulação com controle Max Pressure
python baselines/analisar_baseline.py     # lê o tripinfo e resume espera/atraso médios
python utils/listar_tls.py                # lista os IDs dos semáforos da rede
```

### 2. Configurar o treino
Antes de treinar, dá uma olhada em `train/config.yaml`: 
- hiperparâmetros fixos, espaço de busca do Optuna (`search_params`/`espaco_busca`)
- Quantos trials (`n_trials`)
- Orçamento de treino (`n_rodadas`/`timesteps_por_turno`/`n_eval_episodios`, separado pra fase do Optuna e pra fase dos modelos finais)
- Paralelizar o Optuna entre as recompensas (`paralelizar_optuna: true/false`).

### 3. Rodar o treino -- PPO e DQN são independentes, roda cada um separadamente

Cada algoritmo tem 2 fases, nessa ordem:
1. **Optuna** (`pipeline_optuna_independentes.py`) -- busca de hiperparâmetros, roda 1 study
   por função de recompensa e salva os resultados em `outputs/{ppo,dqn}_{recompensa}_optuna.csv`.
2. **Modelos finais** (`treinar_modelos_finais.py`) -- treina e salva os 6 modelos finais
   (3 recompensas x com/sem Optuna) usando os hiperparâmetros achados na fase 1. **Precisa
   rodar a fase 1 primeiro** (lê o `.db` do Optuna); sem isso dá erro `RuntimeError` avisando
   pra rodar a fase 1 antes.

**PPO:**
```bash
python train/ppo/pipeline_optuna_independentes.py   # fase 1: Optuna
python train/ppo/treinar_modelos_finais.py          # fase 2: 6 modelos finais
```

**DQN:**
```bash
python train/dqn/pipeline_optuna_independentes.py   # fase 1: Optuna
python train/dqn/treinar_modelos_finais.py          # fase 2: 6 modelos finais
```

Pode rodar só um dos dois algoritmos, ou os dois -- nesse caso, cada comando acima é
independente, então dá pra rodar em terminais separados (ou um de cada vez, na ordem que
quiser). Cada fase pode demorar bastante (horas), então considere rodar com `nohup` pra não
depender do terminal ficar aberto:
```bash
mkdir -p outputs/logs
nohup python train/ppo/pipeline_optuna_independentes.py > outputs/logs/optuna_ppo.log 2>&1 &
```

### 4. Ver os resultados
```bash
tensorboard --logdir outputs/tensorboard
```
Resumo em CSV (1 linha por configuração -- recompensa x com/sem Optuna):
`outputs/modelos_finais_ppo.csv` e `outputs/modelos_finais_dqn.csv`.

### 5. Ver a simulação rodando (SUMO-GUI)
```bash
python utils/visualizar_simulacao.py --modo baseline
python utils/visualizar_simulacao.py --modo modelo --algoritmo ppo --tag com_optuna --recompensa diff_waiting_time
```
(`--tag` é `com_optuna` ou `sem_optuna`; `--algoritmo` é `ppo` ou `dqn`; precisa já ter os
`.zip` correspondentes em `outputs/modelos/`, ou seja, já ter rodado o passo 3 pra essa
combinação.)

### 6. (Opção) Rodar tudo de uma vez com rodar_pipeline.sh
Em vez de rodar os 4 comandos do passo 3 na mão (2 por algoritmo), o `rodar_pipeline.sh` roda
a fase 1 (Optuna) dos dois algoritmos em paralelo, espera os dois terminarem, e só depois roda
a fase 2 (modelos finais) dos dois também em paralelo:
```bash
./rodar_pipeline.sh
# ou, pra sobreviver ao terminal fechar:
nohup ./rodar_pipeline.sh &
```
Isso usa bem mais CPU/RAM ao mesmo tempo (2 pipelines completos rodando juntos) -- se sua
máquina tiver poucos núcleos, prefira rodar PPO e DQN separados (passo 3). Os logs de cada
processo ficam em `outputs/logs/`.

