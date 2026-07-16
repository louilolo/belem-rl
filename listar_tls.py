import sumolib
net = sumolib.net.readNet('belem.net.xml')

with open('semaforos.txt', 'w') as arquivo:
    for tls in net.getTrafficLights():
        tls_id = tls.getID()

        print(tls_id)

        arquivo.write(f"{tls_id}\n")

print("\nSucesso! Todos os IDs foram salvos no arquivo 'semaforos.txt'.")