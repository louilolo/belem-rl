import pandas as pd
import xml.etree.ElementTree as ET

def ler_tripinfo(path):
    root = ET.parse(path).getroot()
    dados = [{
        'id': t.get('id'),
        'duration': float(t.get('duration')),
        'waitingTime': float(t.get('waitingTime')),
        'timeLoss': float(t.get('timeLoss')),
    } for t in root.findall('tripinfo')]
    return pd.DataFrame(dados)

df = ler_tripinfo('sumo/almirante/baseline_maxpressure.xml')

medias = df[['waitingTime', 'timeLoss']].mean()

print("=== Resultados do Baseline (Tempo Fixo) ===")
print(medias.to_string())
print("===========================================")

# 4. Salva o resultado formatado em um arquivo de texto
with open('resultados_baseline.txt', 'w', encoding='utf-8') as arquivo:
    arquivo.write("=== Resultados do Baseline (Tempo Fixo) ===\n")
    arquivo.write(f"Tempo médio de espera parado (waitingTime): {medias['waitingTime']:.2f} segundos\n")
    arquivo.write(f"Atraso médio na viagem (timeLoss): {medias['timeLoss']:.2f} segundos\n")
    arquivo.write("===========================================\n")

print("\nSucesso! Os resultados foram salvos no arquivo 'resultados_baseline.txt'.")