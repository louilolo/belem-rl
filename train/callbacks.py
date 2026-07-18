"""Callback compartilhado por PPO e DQN (independe do algoritmo -- só usa o info dict que
sumo_rl devolve em env.step(), disponível igual pra qualquer um). Loga as métricas de
sistema da simulação (velocidade média, tempo de espera, veículos parados) no mesmo
tensorboard do agente, lado a lado com as métricas de treino do próprio algoritmo
(loss, entropia, etc.).
"""
from stable_baselines3.common.callbacks import BaseCallback


class MetricasSimulacaoCallback(BaseCallback):
    """Grava simulacao/mean_waiting_time, simulacao/mean_speed e simulacao/total_stopped
    a cada passo de treino. Se o modelo não tiver um logger de tensorboard configurado
    (ex: durante os trials do Optuna, que não usam tensorboard_dir), os record()/dump()
    caem num logger vazio e não fazem nada -- sem custo real, sem precisar checar antes.
    """

    def _on_step(self):
        info = self.locals["infos"][0]  # DummyVecEnv com 1 env -> lista de 1 elemento
        self.logger.record("simulacao/mean_waiting_time", info["system_mean_waiting_time"])
        self.logger.record("simulacao/mean_speed", info["system_mean_speed"])
        self.logger.record("simulacao/total_stopped", info["system_total_stopped"])
        self.logger.dump(step=self.num_timesteps)
        return True
