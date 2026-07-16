# RL para controle de semáforos - Belém

Projeto final IC 2026.2 - UFPA

## Setup
conda env create -f environment.yml
conda activate belem

## Contrato de arquivos (não mudar os nomes)
- scenario/corredor.net.xml  rede
- scenario/rotas.rou.xml     demanda

## Divisão
- A: rede/OSM
- B: demanda + Max Pressure
- C: agentes RL
- D: Optuna + avaliação
