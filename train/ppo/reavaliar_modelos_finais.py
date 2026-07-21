"""Re-roda só a avaliação dos 6 modelos finais (PPO) usando os .zip já treinados, sem
retreinar -- usado depois da correção do bug em avaliar_agentes_independentes (não
resetava entre episódios de avaliação, ver treinar_agentes_independentes.py).
Sobrescreve outputs/modelos_finais_ppo.csv com as métricas corrigidas.
"""
import os
import sys
import csv

from stable_baselines3 import PPO

sys.path.append(os.path.dirname(__file__))
from treinar_modelos_finais import (  # noqa: E402
    RECOMPENSAS, N_EVAL_EPISODIOS, CSV_COLUNAS,
    construir_reward_fn, carregar_melhores_hiperparametros, hiperparams_sem_optuna,
)
from treinar_agentes_independentes import criar_sumo_compartilhado, avaliar_agentes_independentes  # noqa: E402


def carregar_modelos_salvos(sumo_env, pasta):
    modelos = {}
    for ts_id in sumo_env.ts_ids:
        modelos[ts_id] = PPO.load(os.path.join(pasta, f'{ts_id}.zip'))
    return modelos


def reavaliar(nome_recompensa, com_optuna):
    tag = 'com_optuna' if com_optuna else 'sem_optuna'
    if com_optuna:
        _, min_green, peso = carregar_melhores_hiperparametros(nome_recompensa)
    else:
        _, min_green, peso = hiperparams_sem_optuna()

    reward_fn = construir_reward_fn(nome_recompensa, peso)
    sumo_env = criar_sumo_compartilhado(
        out_csv_name=f'outputs/reavaliacao_ppo_{tag}_{nome_recompensa}_eval',
        reward_fn=reward_fn,
        min_green=min_green,
    )
    sumo_env.reset()  # só pra popular ts_ids antes de carregar os modelos

    pasta = f'outputs/modelos/ppo/{tag}/{nome_recompensa}'
    modelos = carregar_modelos_salvos(sumo_env, pasta)

    metricas = avaliar_agentes_independentes(sumo_env, modelos, N_EVAL_EPISODIOS)
    sumo_env.close()

    print(f"[{nome_recompensa} | {tag}] métricas corrigidas: {metricas}")
    return metricas


if __name__ == "__main__":
    linhas = []
    for nome_recompensa in RECOMPENSAS:
        for com_optuna in (True, False):
            metricas = reavaliar(nome_recompensa, com_optuna)
            linhas.append({'reward_function': nome_recompensa, 'com_optuna': com_optuna, **metricas})

    caminho = 'outputs/modelos_finais_ppo.csv'
    with open(caminho, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUNAS)
        writer.writeheader()
        writer.writerows(linhas)
    print(f"\n{caminho} atualizado com as métricas corrigidas.")
