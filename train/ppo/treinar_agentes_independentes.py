"""Um agente PPO por semáforo (9 no total), cada um com seu próprio espaço de ação/observação
nativo -- sem padding, sem política compartilhada.

Como sumo_rl/SB3 não têm suporte nativo pra treinar N agentes de verdade em paralelo contra
UMA simulação compartilhada, o treino aqui é round-robin: em cada "turno", o agente da vez
treina normalmente via model.learn() (SB3 dirige a simulação inteira), enquanto os outros 8
agentes agem usando suas próprias políticas atuais (via .predict(), consultadas ao vivo a cada
passo). Depois de várias rodadas passando por todos os semáforos, cada um já viu bastante
experiência e reagiu ao comportamento (em evolução) dos vizinhos.

Isso resolve o problema do padding: cada semáforo aprende só sobre suas próprias fases reais,
sem a queda no-op para fase 0 quando um índice de ação não existe pra ele.
"""
import os
import sys
import warnings

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure as configurar_logger
from sumo_rl import SumoEnvironment

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'reward'))
from rewards import velocity_time_delta  # noqa: E402
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from callbacks import MetricasSimulacaoCallback  # noqa: E402

warnings.filterwarnings("ignore", message=".*render_mode.*")
os.makedirs('outputs', exist_ok=True)


def criar_sumo_compartilhado(out_csv_name=None, reward_fn=None, peso=0.95, min_green=10):
    return SumoEnvironment(
        net_file='cenarios/almirante/belem.net.xml',
        route_file='cenarios/almirante/rotas.rou.xml',
        out_csv_name=out_csv_name,
        use_gui=False,
        num_seconds=3600,
        delta_time=5,
        yellow_time=4,
        min_green=min_green,
        max_green=60,
        single_agent=False,
        reward_fn=reward_fn if reward_fn is not None else velocity_time_delta(peso),
        fixed_ts=False,
        sumo_warnings=False,
    )


class AgenteSemaforoEnv(gym.Env):
    """Visão de UM semáforo sobre a simulação compartilhada.

    As ações dos outros semáforos são decididas pelas políticas atuais deles (dict
    `modelos`, compartilhado e consultado ao vivo a cada passo -- então sempre reflete
    a versão mais recente de cada um, mesmo em treino).
    """

    def __init__(self, ts_id, sumo_env, modelos):
        super().__init__()
        self.ts_id = ts_id
        self.sumo_env = sumo_env
        self.modelos = modelos
        self.observation_space = sumo_env.observation_spaces(ts_id)
        self.action_space = sumo_env.action_spaces(ts_id)
        self._ultimas_obs = None

    def reset(self, seed=None, options=None):
        obs_dict = self.sumo_env.reset()
        self._ultimas_obs = obs_dict
        return obs_dict[self.ts_id], {}

    def step(self, action):
        acoes = {self.ts_id: int(action)}
        for outro_id in self.sumo_env.ts_ids:
            if outro_id == self.ts_id:
                continue
            obs_outro = self._ultimas_obs[outro_id]
            acao_outro, _ = self.modelos[outro_id].predict(obs_outro, deterministic=False)
            acoes[outro_id] = int(acao_outro)

        obs_dict, rewards_dict, dones_dict, info = self.sumo_env.step(acoes)
        self._ultimas_obs = obs_dict
        truncated = bool(dones_dict["__all__"])
        return obs_dict[self.ts_id], float(rewards_dict[self.ts_id]), False, truncated, info


def criar_modelos(sumo_env, ppo_kwargs, tensorboard_dir=None):
    """Cria 1 PPO por semáforo, cada um com seu próprio espaço de ação/observação nativo.

    Se tensorboard_dir for passado (ex: 'outputs/tensorboard/ppo/optuna_diff_waiting_time'),
    cada agente i grava seu próprio log em '{tensorboard_dir}_{i}'. set_logger() é chamado
    ANTES de qualquer .learn(), então SB3 nunca re-configura o logger sozinho (o que criaria
    um sufixo numérico extra) -- e como treinar_round_robin já usa reset_num_timesteps=False,
    o log fica contínuo através das rodadas, em vez de fragmentado.
    """
    modelos = {}
    for i, ts_id in enumerate(sumo_env.ts_ids):
        wrapper = AgenteSemaforoEnv(ts_id, sumo_env, modelos)
        vec_env = DummyVecEnv([lambda w=wrapper: w])
        modelo = PPO("MlpPolicy", vec_env, device="cpu", verbose=0, **ppo_kwargs)
        if tensorboard_dir is not None:
            modelo.set_logger(configurar_logger(f"{tensorboard_dir}_{i}", ["tensorboard"]))
        modelos[ts_id] = modelo
    return modelos


def treinar_round_robin(modelos, ts_ids, n_rodadas, timesteps_por_turno):
    for rodada in range(1, n_rodadas + 1):
        print(f"\n=== Rodada {rodada}/{n_rodadas} ===")
        for ts_id in ts_ids:
            print(f"  Treinando semáforo {ts_id} ({timesteps_por_turno} timesteps)...")
            modelos[ts_id].learn(
                total_timesteps=timesteps_por_turno, reset_num_timesteps=False,
                callback=MetricasSimulacaoCallback(),
            )


def avaliar_agentes_independentes(sumo_env, modelos, n_episodios=2):
    """Roda os 9 agentes juntos (ação determinística) e tira a média das métricas de sistema --
    as mesmas usadas pra comparar com o approach de política compartilhada."""
    valores = {'system_mean_waiting_time': [], 'system_mean_speed': [], 'system_total_stopped': []}
    obs_dict = sumo_env.reset()
    episodios_completos = 0
    while episodios_completos < n_episodios:
        acoes = {ts_id: int(modelos[ts_id].predict(obs_dict[ts_id], deterministic=True)[0]) for ts_id in sumo_env.ts_ids}
        obs_dict, rewards_dict, dones_dict, info = sumo_env.step(acoes)
        for k in valores:
            valores[k].append(info[k])
        if dones_dict["__all__"]:
            episodios_completos += 1
    return {k: float(np.mean(v)) for k, v in valores.items()}


if __name__ == "__main__":
    N_RODADAS = 3
    TIMESTEPS_POR_TURNO = 1200
    N_EVAL_EPISODIOS = 2
    PPO_KWARGS = dict(learning_rate=3e-4, gamma=0.99, n_steps=256, batch_size=64, ent_coef=0.01)

    sumo_env = criar_sumo_compartilhado(out_csv_name='outputs/independente_treino')
    sumo_env.reset()  # só pra popular ts_ids/spaces antes de criar os modelos
    ts_ids = list(sumo_env.ts_ids)
    print(f"Semáforos: {ts_ids}")

    modelos = criar_modelos(sumo_env, PPO_KWARGS)
    treinar_round_robin(modelos, ts_ids, N_RODADAS, TIMESTEPS_POR_TURNO)

    print("\n--- Avaliando os 9 agentes juntos ---")
    sumo_env.out_csv_name = 'outputs/independente_eval'
    metricas = avaliar_agentes_independentes(sumo_env, modelos, N_EVAL_EPISODIOS)
    sumo_env.close()

    print("\n=========================================")
    print(" AGENTES INDEPENDENTES POR SEMÁFORO")
    print(f" ({N_RODADAS} rodadas x {TIMESTEPS_POR_TURNO} timesteps/turno, {N_EVAL_EPISODIOS} episódios de avaliação)")
    print("=========================================")
    for k, v in metricas.items():
        print(f"{k}: {v:.3f}")
    print("=========================================")
