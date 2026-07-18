"""Abre o SUMO-GUI pra assistir a simulação rodando -- com os agentes treinados ou no
baseline de tempo fixo (sem agente nenhum).

Uso:
  python utils/visualizar_simulacao.py --modo baseline
  python utils/visualizar_simulacao.py --modo modelo --algoritmo ppo --tag com_optuna --recompensa diff_waiting_time

--tag com_optuna  = modelos treinados com os hiperparâmetros vencedores do Optuna
--tag sem_optuna  = modelos treinados com hiperparâmetros fixos (sem Optuna)
(ver train/{ppo,dqn}/treinar_modelos_finais.py -- é de lá que vêm os .zip em outputs/modelos/)
"""
import os
import argparse

from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO, DQN


def criar_env_gui(min_green, fixed_ts):
    return SumoEnvironment(
        net_file='cenarios/almirante/belem.net.xml',
        route_file='cenarios/almirante/rotas.rou.xml',
        out_csv_name=None,  # só pra assistir, não precisa logar nada
        use_gui=True,
        num_seconds=3600,
        delta_time=5,
        yellow_time=4,
        min_green=min_green,
        max_green=60,
        single_agent=False,
        reward_fn='diff-waiting-time',  # irrelevante aqui: não afeta as ações, só o valor de recompensa reportado
        fixed_ts=fixed_ts,
        sumo_warnings=False,
    )


def carregar_modelos(sumo_env, algoritmo, pasta):
    if not os.path.isdir(pasta):
        raise RuntimeError(f"Não achei modelos treinados em {pasta}. Rode treinar_modelos_finais.py primeiro.")
    Classe = PPO if algoritmo == 'ppo' else DQN
    modelos = {}
    for ts_id in sumo_env.ts_ids:
        caminho = os.path.join(pasta, f'{ts_id}.zip')
        modelos[ts_id] = Classe.load(caminho)
    return modelos


def rodar_baseline(min_green):
    print("Rodando baseline (tempo fixo, sem agente)...")
    sumo_env = criar_env_gui(min_green, fixed_ts=True)
    sumo_env.reset()
    dones = {"__all__": False}
    while not dones["__all__"]:
        _, _, dones, _ = sumo_env.step({})  # sem ações -- fixed_ts já ignora, mas deixa explícito
    sumo_env.close()


def rodar_modelo(algoritmo, tag, recompensa, min_green):
    pasta = f'outputs/modelos/{algoritmo}/{tag}/{recompensa}'
    print(f"Rodando com os agentes treinados: {algoritmo}/{tag}/{recompensa}")

    sumo_env = criar_env_gui(min_green, fixed_ts=False)
    modelos = carregar_modelos(sumo_env, algoritmo, pasta)  # ts_ids já existe, não precisa de reset() antes

    obs_dict = sumo_env.reset()
    dones = {"__all__": False}
    while not dones["__all__"]:
        acoes = {ts_id: int(modelos[ts_id].predict(obs_dict[ts_id], deterministic=True)[0]) for ts_id in sumo_env.ts_ids}
        obs_dict, _, dones, _ = sumo_env.step(acoes)
    sumo_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--modo', choices=['baseline', 'modelo'], required=True)
    parser.add_argument('--algoritmo', choices=['ppo', 'dqn'], default='ppo')
    parser.add_argument('--tag', choices=['com_optuna', 'sem_optuna'], default='sem_optuna')
    parser.add_argument('--recompensa', choices=['diff_waiting_time', 'velocity_time', 'velocity_time_delta'], default='diff_waiting_time')
    parser.add_argument('--min_green', type=int, default=10, help="use o mesmo min_green usado no treino desse modelo, se souber")
    args = parser.parse_args()

    if args.modo == 'baseline':
        rodar_baseline(args.min_green)
    else:
        rodar_modelo(args.algoritmo, args.tag, args.recompensa, args.min_green)
