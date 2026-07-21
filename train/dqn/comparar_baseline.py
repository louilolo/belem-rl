"""Roda os 6 modelos finais (DQN) até TODOS os veículos completarem a viagem (igual ao
baseline em baselines/rodar_maxpressure.py), gravando tripinfo, e calcula waitingTime/
timeLoss médios por viagem completa -- a MESMA métrica usada pra analisar o baseline
(baselines/analisar_baseline.py). Isso permite comparação direta com o baseline, ao
contrário do system_mean_waiting_time usado no resto do pipeline pra comparar recompensas
entre si (que é uma média instantânea sobre os veículos presentes a cada passo, não por
viagem completa -- ver conversa sobre por que não dava pra comparar os dois direto).
"""
import os
import sys
import csv
import xml.etree.ElementTree as ET

import pandas as pd
from stable_baselines3 import DQN

sys.path.append(os.path.dirname(__file__))
from treinar_modelos_finais import (  # noqa: E402
    RECOMPENSAS, construir_reward_fn, carregar_melhores_hiperparametros, hiperparams_sem_optuna,
)
from treinar_agentes_independentes import criar_sumo_compartilhado  # noqa: E402

# margem folgada -- o baseline (max-pressure) terminou as 4236 viagens em ~4043s
NUM_SECONDS = 7200


def carregar_modelos_salvos(sumo_env, pasta):
    modelos = {}
    for ts_id in sumo_env.ts_ids:
        modelos[ts_id] = DQN.load(os.path.join(pasta, f'{ts_id}.zip'))
    return modelos


def ler_tripinfo(caminho):
    root = ET.parse(caminho).getroot()
    dados = [{
        'waitingTime': float(t.get('waitingTime')),
        'timeLoss': float(t.get('timeLoss')),
    } for t in root.findall('tripinfo')]
    return pd.DataFrame(dados)


def rodar_e_medir(nome_recompensa, com_optuna):
    tag = 'com_optuna' if com_optuna else 'sem_optuna'
    print(f"\n--- [{nome_recompensa} | {tag}] rodando até todos os veículos chegarem ---")
    if com_optuna:
        _, min_green, peso = carregar_melhores_hiperparametros(nome_recompensa)
    else:
        _, min_green, peso = hiperparams_sem_optuna()

    reward_fn = construir_reward_fn(nome_recompensa, peso)
    tripinfo_path = os.path.abspath(f'outputs/tripinfo_dqn_{tag}_{nome_recompensa}.xml')
    sumo_env = criar_sumo_compartilhado(
        out_csv_name=None,
        reward_fn=reward_fn,
        min_green=min_green,
        num_seconds=NUM_SECONDS,
        additional_sumo_cmd=f'--tripinfo-output {tripinfo_path} --tripinfo-output.write-unfinished',
    )
    obs_dict = sumo_env.reset()

    pasta = f'outputs/modelos/dqn/{tag}/{nome_recompensa}'
    modelos = carregar_modelos_salvos(sumo_env, pasta)

    dones = {"__all__": False}
    while not dones["__all__"]:
        acoes = {ts_id: int(modelos[ts_id].predict(obs_dict[ts_id], deterministic=True)[0]) for ts_id in sumo_env.ts_ids}
        obs_dict, _, dones, _ = sumo_env.step(acoes)
    sumo_env.close()

    df = ler_tripinfo(tripinfo_path)
    medias = df[['waitingTime', 'timeLoss']].mean()
    print(f"[{nome_recompensa} | {tag}] {len(df)} viagens -- "
          f"waitingTime médio: {medias['waitingTime']:.2f}s, timeLoss médio: {medias['timeLoss']:.2f}s")
    return {
        'reward_function': nome_recompensa, 'com_optuna': com_optuna, 'n_viagens': len(df),
        'waitingTime_medio': medias['waitingTime'], 'timeLoss_medio': medias['timeLoss'],
    }


if __name__ == "__main__":
    linhas = []
    for nome_recompensa in RECOMPENSAS:
        for com_optuna in (True, False):
            linhas.append(rodar_e_medir(nome_recompensa, com_optuna))

    caminho = 'outputs/comparacao_baseline_dqn.csv'
    with open(caminho, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['reward_function', 'com_optuna', 'n_viagens', 'waitingTime_medio', 'timeLoss_medio'])
        writer.writeheader()
        writer.writerows(linhas)
    print(f"\n{caminho} salvo.")
