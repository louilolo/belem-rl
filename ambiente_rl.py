from sumo_rl import SumoEnvironment

# Configurando o "Mundo" da nossa Inteligência Artificial
env = SumoEnvironment(
    net_file='sumo/almirante/belem.net.xml',
    route_file='sumo/almirante/rotas.rou.xml',
    out_csv_name='outputs/ppo_belem', # Vai criar uma pasta 'outputs' para salvar os relatórios
    use_gui=False,         # Deixe False para treinar rápido. Mude para True quando quiser "assistir" a IA jogando.
    num_seconds=3600,      # 1 hora simulada por partida (episódio)
    delta_time=5,          # A IA pensa e decide a cada 5 segundos
    yellow_time=3,         # Tempo de transição do amarelo
    min_green=10,          # O verde tem que durar pelo menos 10s para não bugar o trânsito
    max_green=60,          # O verde não pode passar de 1 minuto
    single_agent=True,     # Modo fácil: focar em 1 semáforo apenas
    reward_fn='diff-waiting-time', # A IA ganha pontos se diminuir o tempo de espera
)

print("Ambiente de Inteligência Artificial criado com sucesso!")
# (A IA ainda não está treinando, isso apenas preparou o tabuleiro do jogo)