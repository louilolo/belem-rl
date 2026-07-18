import os
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# 1. Garantindo que as pastas existam para o código não dar erro
os.makedirs('outputs', exist_ok=True)
os.makedirs('modelos', exist_ok=True)
os.makedirs('tb', exist_ok=True)

# 2. O Ambiente (O tabuleiro do jogo)
env = SumoEnvironment(
    net_file='cenarios/almirante/belem.net.xml',
    route_file='cenarios/almirante/rotas.rou.xml',
    out_csv_name='outputs/ppo_belem',
    use_gui=False,
    num_seconds=3600,
    delta_time=5,
    yellow_time=3,
    min_green=10,
    max_green=60,
    single_agent=True,
    reward_fn='diff-waiting-time',
    fixed_ts=True,
)

# O Monitor anota a pontuação para a gente ver no gráfico depois
env = Monitor(env)

# 3. Criando o Cérebro da IA (PPO)
print("Criando o modelo PPO...")
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    tensorboard_log="tb/" # Aqui ele salva os gráficos
)

# 4. A hora do suor: Treinamento!
print("Iniciando o treinamento (100.000 passos)... Vai pegar um café! ☕")
model.learn(total_timesteps=100_000)

# 5. Salvando o cérebro treinado
model.save("modelos/ppo_belem")
print("Treinamento concluído e modelo salvo com sucesso!")