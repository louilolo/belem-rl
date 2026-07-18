import os
import sumolib

net = sumolib.net.readNet('cenarios/almirante/belem.net.xml')

os.makedirs('outputs', exist_ok=True)
with open('outputs/semaforos.txt', 'w') as arquivo:
    for tls in net.getTrafficLights():
        tls_id = tls.getID()

        print(tls_id)

        arquivo.write(f"{tls_id}\n")

print("\nSucesso! Todos os IDs foram salvos no arquivo 'outputs/semaforos.txt'.")