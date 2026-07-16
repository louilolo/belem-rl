import os
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from sumo_rl import SumoEnvironment

# Garante que as pastas existem
os.makedirs('almirante/outputs', exist_ok=True)

def criar_env(trial=None):
    # DICA DE OURO DA IMAGEM: Otimizando também as regras do trânsito!
    # O Optuna vai testar se o verde mínimo deve ser 10, 15 ou 20 segundos.
    verde_minimo = trial.suggest_int('min_green', 10, 20) if trial else 10

    env = SumoEnvironment(
        net_file='almirante/belem.net.xml',
        route_file='almirante/rotas.rou.xml',
        out_csv_name=f'almirante/outputs/optuna_trial_{trial.number}' if trial else 'almirante/outputs/eval',
        use_gui=False,
        num_seconds=3600,
        delta_time=5,
        yellow_time=3,
        min_green=verde_minimo, # Passando o parâmetro do Optuna
        max_green=60,
        single_agent=True,
        reward_fn='diff-waiting-time',
        fixed_ts=True # Essencial para não dar conflito com o TraCI!
    )
    return Monitor(env)

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
                verbose=0) # verbose=0 pra não poluir o terminal
                
    # 3. Treina com orçamento menor (ex: 30.000 passos para ser rápido)
    try:
        model.learn(total_timesteps=30_000)
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
        storage='sqlite:///almirante/optuna_belem.db', 
        study_name='ppo_belem', 
        load_if_exists=True
    )
    
    # Pede pro Optuna fazer 10 tentativas de receitas (mude para 30 ou 50 quando for treinar pra valer)
    study.optimize(objective, n_trials=10)
    
    print("\n=========================================")
    print(" OTIMIZAÇÃO CONCLUÍDA! ")
    print("Os melhores parâmetros encontrados foram:")
    print(study.best_params)
    print(f"Com a pontuação máxima de: {study.best_value}")
    print("=========================================")