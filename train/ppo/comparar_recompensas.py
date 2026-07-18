"""Compara a recompensa custom (velocidade - espera, ponderada) com a 'diff-waiting-time'
do sumo_rl. Treina uma política PPO com cada uma (mesmos hiperparâmetros/seed) e avalia
as duas usando métricas de trânsito neutras (tempo médio de espera, velocidade média,
veículos parados) — não dá pra comparar o valor bruto da recompensa entre as duas, já
que cada uma tem escala/unidade diferente.
"""
import os
import sys
import warnings

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from otimizar_optuna import criar_env

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'reward'))
from rewards import velocity_time, velocity_time_delta  # noqa: E402

warnings.filterwarnings("ignore", message=".*render_mode.*")
os.makedirs('outputs', exist_ok=True)

TOTAL_TIMESTEPS = 100_000
N_EVAL_EPISODIOS = 2
SEED = 42
PPO_KWARGS = dict(learning_rate=3e-4, gamma=0.99, n_steps=1024, batch_size=64, ent_coef=0.01)

RECOMPENSAS = {
    'diff_waiting_time': 'diff-waiting-time',
    'velocity_time_delta': velocity_time_delta(peso=0.95),
}


def avaliar_metricas_sistema(model, env, n_episodios=N_EVAL_EPISODIOS):
    """Roda n_episodios e tira a média das métricas de sistema (iguais p/ qualquer reward_fn)."""
    valores = {'system_mean_waiting_time': [], 'system_mean_speed': [], 'system_total_stopped': []}
    obs = env.reset()
    episodios_completos = 0
    while episodios_completos < n_episodios:
        acoes, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(acoes)
        for k in valores:
            valores[k].append(infos[0][k])
        if dones[0]:
            episodios_completos += 1
    return {k: float(np.mean(v)) for k, v in valores.items()}


def treinar_e_avaliar(nome, reward_fn):
    print(f"\n--- Treinando com recompensa: {nome} ---")
    # supersuit's ConcatVecEnv não implementa .seed(), então semeamos os RNGs globais
    # direto em vez de passar seed= pro PPO (que tentaria chamar env.seed()).
    set_random_seed(SEED)
    env = criar_env(reward_fn=reward_fn, out_csv_name=f'outputs/comparacao_{nome}')
    model = PPO("MlpPolicy", env, device="cpu", verbose=0, **PPO_KWARGS)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    env.close()

    print(f"--- Avaliando: {nome} ---")
    eval_env = criar_env(reward_fn=reward_fn, out_csv_name=f'outputs/comparacao_{nome}_eval')
    metricas = avaliar_metricas_sistema(model, eval_env)
    eval_env.close()

    print(f"{nome}: {metricas}")
    return metricas


if __name__ == "__main__":
    resultados = {nome: treinar_e_avaliar(nome, reward_fn) for nome, reward_fn in RECOMPENSAS.items()}

    print("\n=========================================")
    print(" COMPARAÇÃO DE RECOMPENSAS")
    print(f" ({TOTAL_TIMESTEPS} timesteps de treino, {N_EVAL_EPISODIOS} episódios de avaliação)")
    print("=========================================")
    header = f"{'métrica':30s}" + "".join(f"{nome:>25s}" for nome in resultados)
    print(header)
    for metrica in ['system_mean_waiting_time', 'system_mean_speed', 'system_total_stopped']:
        linha = f"{metrica:30s}" + "".join(f"{resultados[nome][metrica]:>25.3f}" for nome in resultados)
        print(linha)
    print("=========================================")
    print("Menor system_mean_waiting_time é melhor. Maior system_mean_speed é melhor.")
