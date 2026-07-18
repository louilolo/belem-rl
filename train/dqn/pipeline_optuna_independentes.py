"""Pipeline Optuna para o treino de agentes independentes (1 DQN por semáforo, round-robin).
Mesma estrutura de train/ppo/pipeline_optuna_independentes.py, trocando o algoritmo e o
espaço de hiperparâmetros (DQN é off-policy: buffer de replay, exploração epsilon-greedy,
target network -- não tem n_steps/ent_coef como o PPO).

Roda um Optuna study separado pra cada função de recompensa (diff-waiting-time nativa,
velocity_time, velocity_time_delta), treina os 9 agentes em round-robin por trial, avalia o
sistema (métricas neutras, sem detalhar por agente) e salva uma linha em
outputs/dqn_{reward_function}_optuna.csv. Otimiza pra MINIMIZAR system_mean_waiting_time.
"""
import os
import sys
import csv
import warnings

import optuna

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'reward'))
from rewards import velocity_time, velocity_time_delta  # noqa: E402

from treinar_agentes_independentes import (  # noqa: E402
    criar_sumo_compartilhado,
    criar_modelos,
    treinar_round_robin,
    avaliar_agentes_independentes,
)

warnings.filterwarnings("ignore", message=".*render_mode.*")
os.makedirs('outputs', exist_ok=True)

# Orçamento de treino por trial (fixo -- só os hiperparâmetros variam entre trials).
N_RODADAS = 3
TIMESTEPS_POR_TURNO = 1200
N_EVAL_EPISODIOS = 2
N_TRIALS = 10  # por função de recompensa
BUFFER_SIZE = 50_000  # fixo -- não crítico o suficiente pra entrar na busca do Optuna

RECOMPENSAS = ['diff_waiting_time', 'velocity_time', 'velocity_time_delta']

CSV_COLUNAS = [
    'trial', 'reward_function',
    'learning_rate', 'gamma', 'batch_size', 'train_freq', 'target_update_interval',
    'exploration_fraction', 'exploration_final_eps', 'min_green', 'peso_recompensa',
    'system_mean_waiting_time', 'system_mean_speed', 'system_total_stopped',
]


def construir_reward_fn(nome, peso):
    """diff-waiting-time é uma string (built-in do sumo_rl); as outras duas são fábricas
    que recebem 1 peso e usam (1-peso) internamente pro outro termo."""
    if nome == 'diff_waiting_time':
        return 'diff-waiting-time'
    if nome == 'velocity_time':
        return velocity_time(peso)
    if nome == 'velocity_time_delta':
        return velocity_time_delta(peso)
    raise ValueError(f"Recompensa desconhecida: {nome}")


def salvar_linha_csv(caminho, linha):
    existe = os.path.exists(caminho)
    with open(caminho, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUNAS)
        if not existe:
            writer.writeheader()
        writer.writerow(linha)


def objective(trial, nome_recompensa, caminho_csv):
    print(f"\n--- [{nome_recompensa}] Trial {trial.number} ---")

    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    train_freq = trial.suggest_categorical('train_freq', [1, 4, 8])
    target_update_interval = trial.suggest_int('target_update_interval', 100, 1000)
    exploration_fraction = trial.suggest_float('exploration_fraction', 0.05, 0.3)
    exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.01, 0.2)
    min_green = trial.suggest_int('min_green', 10, 20)
    # diff-waiting-time não tem peso (não é a nossa fábrica com 1 único parâmetro).
    peso = trial.suggest_float('peso_recompensa', 0.0, 1.0) if nome_recompensa != 'diff_waiting_time' else None

    reward_fn = construir_reward_fn(nome_recompensa, peso)
    dqn_kwargs = dict(
        learning_rate=lr, gamma=gamma, batch_size=batch_size, train_freq=train_freq,
        target_update_interval=target_update_interval, exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps, buffer_size=BUFFER_SIZE, learning_starts=100,
    )

    sumo_env = criar_sumo_compartilhado(
        out_csv_name=f'outputs/dqn_{nome_recompensa}_optuna_trial{trial.number}_treino',
        reward_fn=reward_fn,
        min_green=min_green,
    )
    sumo_env.reset()  # só pra popular ts_ids/spaces antes de criar os modelos
    ts_ids = list(sumo_env.ts_ids)

    modelos = criar_modelos(sumo_env, dqn_kwargs)
    treinar_round_robin(modelos, ts_ids, N_RODADAS, TIMESTEPS_POR_TURNO)

    sumo_env.out_csv_name = f'outputs/dqn_{nome_recompensa}_optuna_trial{trial.number}_eval'
    metricas = avaliar_agentes_independentes(sumo_env, modelos, N_EVAL_EPISODIOS)
    sumo_env.close()

    linha = {
        'trial': trial.number,
        'reward_function': nome_recompensa,
        'learning_rate': lr, 'gamma': gamma, 'batch_size': batch_size, 'train_freq': train_freq,
        'target_update_interval': target_update_interval, 'exploration_fraction': exploration_fraction,
        'exploration_final_eps': exploration_final_eps, 'min_green': min_green, 'peso_recompensa': peso,
        **metricas,
    }
    salvar_linha_csv(caminho_csv, linha)

    print(f"[{nome_recompensa}] Trial {trial.number} concluído: {metricas}")
    return metricas['system_mean_waiting_time']


if __name__ == "__main__":
    for nome_recompensa in RECOMPENSAS:
        print("\n=========================================")
        print(f" OPTUNA (DQN) -- {nome_recompensa}")
        print("=========================================")

        caminho_csv = f'outputs/dqn_{nome_recompensa}_optuna.csv'
        study = optuna.create_study(
            direction='minimize',  # minimiza system_mean_waiting_time
            storage=f'sqlite:///outputs/optuna_independentes_dqn_{nome_recompensa}.db',
            study_name=f'independentes_dqn_{nome_recompensa}',
            load_if_exists=True,
        )
        study.optimize(lambda trial: objective(trial, nome_recompensa, caminho_csv), n_trials=N_TRIALS)

        print(f"\nMelhores parâmetros para {nome_recompensa}: {study.best_params}")
        print(f"Menor system_mean_waiting_time: {study.best_value}")

    print("\n=========================================")
    print(" PIPELINE DQN CONCLUÍDO")
    print(" CSVs gerados: " + ", ".join(f"outputs/dqn_{r}_optuna.csv" for r in RECOMPENSAS))
    print("=========================================")
