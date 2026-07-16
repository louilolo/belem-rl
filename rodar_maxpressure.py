import traci

# O tempo mínimo (em segundos) que um sinal deve ficar verde antes de recalcular a pressão.
# Isso evita que o semáforo fique "piscando" loucamente a cada 1 segundo.
MIN_GREEN = 15 

def aplicar_max_pressure(tls_id):
    # 1. Pergunta ao SUMO quais faixas este semáforo controla
    links = traci.trafficlight.getControlledLinks(tls_id)

    # 2. Pega todas as combinações de luzes (fases) que o netconvert criou para este cruzamento
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    fases = logic.getPhases()

    melhor_fase_idx = traci.trafficlight.getPhase(tls_id)
    melhor_pressao = -999999

    # 3. Itera sobre todas as fases para encontrar as fases verdes (G ou g)
    for i, fase in enumerate(fases):
        state = fase.state
        
        # Ignora fases amarelas ou vermelhas puras
        if 'G' not in state and 'g' not in state:
            continue

        pressao_desta_fase = 0
        lanes_entrada_vistas = set()
        lanes_saida_vistas = set()

        # 4. Avalia letra por letra da fase atual (Ex: 'rrGGgrr')
        for char_idx, luz in enumerate(state):
            if luz == 'G' or luz == 'g':
                # Pega as faixas de entrada e saída que essa luz verde libera
                if char_idx < len(links) and links[char_idx]:
                    for conexao in links[char_idx]:
                        in_lane = conexao[0]
                        out_lane = conexao[1]

                        # Conta quantos carros querem entrar (Pressão Positiva)
                        if in_lane not in lanes_entrada_vistas:
                            pressao_desta_fase += traci.lane.getLastStepVehicleNumber(in_lane)
                            lanes_entrada_vistas.add(in_lane)

                        # Subtrai os carros que já estão travando a saída (Pressão Negativa)
                        if out_lane not in lanes_saida_vistas:
                            pressao_desta_fase -= traci.lane.getLastStepVehicleNumber(out_lane)
                            lanes_saida_vistas.add(out_lane)

        # 5. Verifica se esta fase ganhou o "campeonato" de pressão
        if pressao_desta_fase > melhor_pressao:
            melhor_pressao = pressao_desta_fase
            melhor_fase_idx = i

    # 6. Manda o SUMO mudar a luz para a fase campeã agora mesmo!
    traci.trafficlight.setPhase(tls_id, melhor_fase_idx)


def iniciar_simulacao():
    # O comando que o Python vai rodar por trás dos panos (sem janela gráfica para ser rápido)
    # Note que ele vai gerar um arquivo novo: baseline_maxpressure.xml
    sumo_cmd = [
        "sumo", 
        "-n", "sumo/almirante/belem.net.xml",
        "-r", "sumo/almirante/rotas.rou.xml",
        "--tripinfo-output", "sumo/almirante/baseline_maxpressure.xml",
        "--no-step-log"
    ]

    print("Iniciando a simulação do Max Pressure... Aguarde.")
    traci.start(sumo_cmd)
    
    # Pega a lista de todos os IDs de semáforos da sua rede de Belém
    tls_ids = traci.trafficlight.getIDList()

    passo_atual = 0
    # O loop principal da simulação: roda até o último carro chegar em casa
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        # Só toma decisões a cada 15 segundos (MIN_GREEN)
        if passo_atual % MIN_GREEN == 0:
            for tls_id in tls_ids:
                aplicar_max_pressure(tls_id)
        
        passo_atual += 1

    traci.close()
    print("Simulação concluída! Arquivo 'baseline_maxpressure.xml' gerado com sucesso.")

if __name__ == "__main__":
    iniciar_simulacao()