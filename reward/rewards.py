"""Funções de recompensa custom, compartilhadas por todos os pipelines de treino
(PPO e DQN, política compartilhada ou agentes independentes). Cada uma é uma fábrica:
recebe 1 peso em [0,1] e devolve a função de recompensa de verdade (que o sumo_rl chama
com o TrafficSignal `ts` a cada passo). peso e (1-peso) somam 1 -- um único
hiperparâmetro controla o trade-off entre as duas escalas, em vez de dois pesos
independentes.
"""


def velocity_time(peso=0.95):
    """Recompensa = peso * speed_ratio - (1 - peso) * mean_waiting_time (valor absoluto).

    speed_ratio = velocidade média / limite de velocidade (já normalizado em [0,1] por get_average_speed).
    mean_waiting_time = tempo de espera acumulado médio entre as faixas de entrada do semáforo (segundos).
    """
    def recompensa(ts):
        speed_ratio = ts.get_average_speed()
        tempos_espera = ts.get_accumulated_waiting_time_per_lane()
        mean_waiting_time = sum(tempos_espera) / len(tempos_espera) if tempos_espera else 0.0
        return peso * speed_ratio - (1 - peso) * mean_waiting_time
    return recompensa


def velocity_time_delta(peso=0.95):
    """Recompensa = peso * speed_ratio + (1 - peso) * delta_espera.

    speed_ratio = velocidade média / limite de velocidade (já normalizado em [0,1] por get_average_speed).
    delta_espera = espera_media do passo anterior menos a atual (positivo = espera caiu = melhorou),
    no mesmo espírito da 'diff-waiting-time' nativa do sumo_rl (self.last_measure - ts_wait),
    em vez do valor absoluto de espera. Estado do "passo anterior" fica guardado no próprio objeto
    `ts`, que o sumo_rl recria do zero a cada reset() — então não precisa zerar isso manualmente.
    """
    def recompensa(ts):
        speed_ratio = ts.get_average_speed()
        tempos_espera = ts.get_accumulated_waiting_time_per_lane()
        mean_waiting_time = sum(tempos_espera) / len(tempos_espera) if tempos_espera else 0.0

        espera_anterior = getattr(ts, '_espera_media_anterior', 0.0)
        delta_espera = espera_anterior - mean_waiting_time
        ts._espera_media_anterior = mean_waiting_time

        return peso * speed_ratio + (1 - peso) * delta_espera
    return recompensa
