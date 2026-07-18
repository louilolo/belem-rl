import os
import sys
import warnings
import optuna
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor
from sumo_rl import parallel_env

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'reward'))
from rewards import velocity_time_delta  # noqa: E402 (recompensas ficam em reward/rewards.py)

# Aviso cosmético: supersuit espera um render_mode que o wrapper multiagente do
# sumo_rl 1.4.5 não propaga sozinho (nós já setamos manualmente em criar_env).
warnings.filterwarnings("ignore", message=".*render_mode.*")

# Garante que as pastas existem
os.makedirs('outputs', exist_ok=True)

def criar_env(trial=None, reward_fn=None, out_csv_name=None):
    # DICA DE OURO DA IMAGEM: Otimizando também as regras do trânsito!
    # O Optuna vai testar se o verde mínimo deve ser 10, 15 ou 20 segundos.
    verde_minimo = trial.suggest_int('min_green', 10, 20) if trial else 10

    if reward_fn is None:
        # peso e (1-peso) também entram na otimização.
        peso = trial.suggest_float('peso_recompensa', 0.0, 1.0) if trial else 0.95
        reward_fn = velocity_time_delta(peso)

    # parallel_env = versão multiagente (PettingZoo) do sumo_rl: um agente por semáforo,
    # todos rodando na mesma simulação SUMO compartilhada.
    env = parallel_env(
        net_file='cenarios/almirante/belem.net.xml',
        route_file='cenarios/almirante/rotas.rou.xml',
        out_csv_name=out_csv_name or (f'outputs/optuna_trial_{trial.number}' if trial else 'outputs/eval'),
        use_gui=False,
        num_seconds=3600,
        delta_time=5,
        yellow_time=4,
        min_green=verde_minimo, # Passando o parâmetro do Optuna
        max_green=60,
        reward_fn=reward_fn,
        fixed_ts=False, # Agora as ações dos agentes realmente controlam os semáforos
        sumo_warnings=False
    )
    # sumo_rl 1.4.5 não seta render_mode no wrapper PettingZoo, mas o supersuit
    # espera esse atributo (mudança de convenção entre as versões das libs).
    env.unwrapped.render_mode = None

    # Os 9 semáforos têm espaços de observação/ação diferentes (fases distintas).
    # pad_* uniformiza tudo para o maior espaço, permitindo 1 única política PPO
    # compartilhada entre os agentes (ações fora do espaço real do agente caem em no-op).
    env = ss.pad_action_space_v0(env)
    env = ss.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    return VecMonitor(env)

def objective(trial):
    print(f"\n--- Iniciando Trial {trial.number} ---")
    
    # 1. Sugerindo os hiperparâmetros do PPO (A "receita" do bolo)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 0.1, log=True)
    
    # 2. Cria o ambiente e o modelo para essa trial
    env = criar_env(trial)
    model = PPO("MlpPolicy", env, learning_rate=lr, gamma=gamma,
                n_steps=n_steps, batch_size=batch_size, ent_coef=ent_coef,
                verbose=0, device="cpu") # verbose=0 pra não poluir o terminal; cpu evita o probe de CUDA
                
    # 3. Treina com orçamento menor (ex: 30.000 passos para ser rápido)
    try:
        model.learn(total_timesteps=20000)
    finally:
        env.close() # FECHA O SUMO para liberar a porta e evitar erros!

    # 4. Avaliação: Testa o modelo treinado para dar uma nota pra essa Trial
    eval_env = criar_env() # Cria um mapa limpo pra avaliar
    try:
        # Roda 2 episódios (2 horas de simulação) para tirar a média da nota
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=2)
    finally:
        eval_env.close() # Fecha de novo pra garantir

    print(f"Trial {trial.number} finalizada com recompensa média de: {mean_reward}")
    return mean_reward

if __name__ == "__main__":
    print("Iniciando o Optuna! Ele vai salvar tudo no banco de dados SQLite.")
    
    # DICA DE OURO: Banco de dados SQLite
    # Isso permite parar, continuar depois, ou dividir o trabalho com a equipe!
    study = optuna.create_study(
        direction='maximize', 
        storage='sqlite:///outputs/optuna_belem.db',
        study_name='ppo_belem', 
        load_if_exists=True
    )
    
    # Pede pro Optuna fazer 10 tentativas de receitas (mude para 30 ou 50 quando for treinar pra valer)
    study.optimize(objective, n_trials=5)
    
    print("\n=========================================")
    print(" OTIMIZAÇÃO CONCLUÍDA! ")
    print("Os melhores parâmetros encontrados foram:")
    print(study.best_params)
    print(f"Com a pontuação máxima de: {study.best_value}")
    print("=========================================")