"""Fase 2 do pipeline PPO: treina e salva os 6 modelos finais (3 funções de recompensa x
{com Optuna, sem Optuna}), depois de rodar pipeline_optuna_independentes.py com as 3
recompensas em RECOMPENSAS (cada uma com seu próprio study do Optuna).

- "Com Optuna": começa dos hiperparâmetros fixos em train/config.yaml (ppo.hyperparams) e
  sobrescreve com os vencedores do PRÓPRIO study daquela função de recompensa -- só os que
  estavam em ppo.optuna.search_params (os demais ficam no valor fixo).
- "Sem Optuna": usa ppo.hyperparams direto, sem nenhuma tunagem.

Cada uma das 6 configurações salva os 9 modelos (1 por semáforo) em
outputs/modelos/ppo/{com_optuna|sem_optuna}/{reward_function}/{ts_id}.zip, e a avaliação final
do sistema em outputs/modelos_finais_ppo.csv.
"""
import os
import sys
import csv

import optuna
import multiprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'reward'))
from rewards import velocity_time, velocity_time_delta  # noqa: E402
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import carregar_config  # noqa: E402
from treinar_agentes_independentes import (  # noqa: E402
    criar_sumo_compartilhado,
    criar_modelos,
    treinar_round_robin,
    avaliar_agentes_independentes,
)

os.makedirs('outputs', exist_ok=True)

CONFIG = carregar_config()
CONFIG_PPO = CONFIG['ppo']
RECOMPENSAS = CONFIG['recompensas']
N_RODADAS = CONFIG_PPO['train_params']['modelos_finais']['n_rodadas']
TIMESTEPS_POR_TURNO = CONFIG_PPO['train_params']['modelos_finais']['timesteps_por_turno']
N_EVAL_EPISODIOS = CONFIG_PPO['train_params']['modelos_finais']['n_eval_episodios']
PARALELIZAR_OPTUNA = CONFIG.get('paralelizar_optuna', False)

CSV_COLUNAS = ['reward_function', 'com_optuna', 'system_mean_waiting_time', 'system_mean_speed', 'system_total_stopped']


def construir_reward_fn(nome, peso):
    if nome == 'diff_waiting_time':
        return 'diff-waiting-time'
    if nome == 'velocity_time':
        return velocity_time(peso)
    if nome == 'velocity_time_delta':
        return velocity_time_delta(peso)
    raise ValueError(f"Recompensa desconhecida: {nome}")


def hiperparams_sem_optuna():
    """Hiperparâmetros fixos direto de train/config.yaml (ppo.hyperparams), sem nenhuma
    tunagem -- min_green e peso_recompensa também vêm de lá."""
    hiperparams = dict(CONFIG_PPO['hyperparams'])
    min_green = hiperparams.pop('min_green')
    peso = hiperparams.pop('peso_recompensa')
    return hiperparams, min_green, peso


def carregar_melhores_hiperparametros(nome_recompensa):
    """Começa dos hiperparâmetros fixos (ppo.hyperparams) e sobrescreve com os vencedores
    do PRÓPRIO study daquela função de recompensa (1 study por recompensa, criado por
    pipeline_optuna_independentes.py) -- só os que de fato entraram na busca (search_params)."""
    study_name = f'independentes_{nome_recompensa}'
    storage = f'sqlite:///outputs/optuna_independentes_{nome_recompensa}.db'
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        melhores = dict(study.best_params)
    except (ValueError, KeyError):
        raise RuntimeError(
            f"Não achei um trial completo do study '{study_name}' em {storage}. "
            f"Rode primeiro pipeline_optuna_independentes.py com '{nome_recompensa}' em RECOMPENSAS."
        )

    hiperparams = dict(CONFIG_PPO['hyperparams'])
    hiperparams.update(melhores)
    min_green = hiperparams.pop('min_green')
    peso = hiperparams.pop('peso_recompensa')
    return hiperparams, min_green, peso


def salvar_modelos(modelos, pasta):
    os.makedirs(pasta, exist_ok=True)
    for ts_id, modelo in modelos.items():
        modelo.save(os.path.join(pasta, f'{ts_id}.zip'))


def salvar_linha_csv(caminho, linha):
    existe = os.path.exists(caminho)
    with open(caminho, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUNAS)
        if not existe:
            writer.writeheader()
        writer.writerow(linha)


def treinar_e_salvar(nome_recompensa, hiperparams, min_green, peso, com_optuna, csv_path):
    tag = 'com_optuna' if com_optuna else 'sem_optuna'
    tag_tb = 'optuna' if com_optuna else 'base'
    print(f"\n--- [{nome_recompensa} | {tag}] ---")

    reward_fn = construir_reward_fn(nome_recompensa, peso)
    sumo_env = criar_sumo_compartilhado(
        out_csv_name=f'outputs/final_ppo_{tag}_{nome_recompensa}_treino',
        reward_fn=reward_fn,
        min_green=min_green,
    )
    sumo_env.reset()  # só pra popular ts_ids/spaces antes de criar os modelos
    ts_ids = list(sumo_env.ts_ids)

    tensorboard_dir = f'outputs/tensorboard/ppo/{tag_tb}_{nome_recompensa}'
    modelos = criar_modelos(sumo_env, hiperparams, tensorboard_dir=tensorboard_dir)
    treinar_round_robin(modelos, ts_ids, N_RODADAS, TIMESTEPS_POR_TURNO)

    sumo_env.out_csv_name = f'outputs/final_ppo_{tag}_{nome_recompensa}_eval'
    metricas = avaliar_agentes_independentes(sumo_env, modelos, N_EVAL_EPISODIOS)
    sumo_env.close()

    salvar_modelos(modelos, f'outputs/modelos/ppo/{tag}/{nome_recompensa}')

    linha = {'reward_function': nome_recompensa, 'com_optuna': com_optuna, **metricas}
    salvar_linha_csv(csv_path, linha)

    print(f"[{nome_recompensa} | {tag}] concluído: {metricas}")

def rodar_para_recompensa(recompensa:str, optuna:bool, csv_path:str):
    if optuna:
        hiperparams, min_green, peso = carregar_melhores_hiperparametros(recompensa)
    else:
        hiperparams, min_green, peso = hiperparams_sem_optuna()
    print(f"[{recompensa}] hiperparâmetros {'optuna' if optuna else 'base'}: {hiperparams}, "
            f"min_green={min_green}, peso={peso}")
    treinar_e_salvar(recompensa, hiperparams, min_green, peso, com_optuna=optuna, csv_path=csv_path) 
    print("\n=========================================")

if __name__ == "__main__":
    csv_path = 'outputs/modelos_finais_ppo.csv'

    if PARALELIZAR_OPTUNA:
        tarefas = []
        for nome_recompensa in RECOMPENSAS:
            tarefas.append((nome_recompensa, True, csv_path))   # Com Optuna
            tarefas.append((nome_recompensa, False, csv_path))  # Sem Optuna
        with multiprocessing.Pool() as pool:
            pool.starmap(rodar_para_recompensa, tarefas)
    else:
        for nome_recompensa in RECOMPENSAS:
            hiperparams_optuna, min_green_optuna, peso_optuna = carregar_melhores_hiperparametros(nome_recompensa)
            print(f"[{nome_recompensa}] hiperparâmetros do Optuna: {hiperparams_optuna}, "
                f"min_green={min_green_optuna}, peso={peso_optuna}")
            treinar_e_salvar(nome_recompensa, hiperparams_optuna, min_green_optuna, peso_optuna, com_optuna=True, csv_path=csv_path)

            hiperparams_base, min_green_base, peso_base = hiperparams_sem_optuna()
            treinar_e_salvar(nome_recompensa, hiperparams_base, min_green_base, peso_base, com_optuna=False, csv_path=csv_path)

    print("\n=========================================")
    print(" 6 MODELOS FINAIS (PPO) TREINADOS E SALVOS")
    print(" outputs/modelos/ppo/{com_optuna,sem_optuna}/{reward_function}/")
    print(" outputs/modelos_finais_ppo.csv")
    print("=========================================")
