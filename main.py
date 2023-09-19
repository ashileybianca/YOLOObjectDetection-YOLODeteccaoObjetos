'''
* Codigo desenvolvido para estudo de Yolo - Detecção de Objetos
Como base:
- João Reis
https://www.youtube.com/watch?v=Sx_HioMUtiY
- Café e Computação
https://www.youtube.com/watch?v=pTbssYiEu1M&list=PLqSRiSjByYuKqUleBDbwjNt5GJLhoae70
- Repositorio do AlexeyAB (um dos desenvolvedores)
https://github.com/AlexeyAB/darknet

* Baixei:
- arquivos originais
- arquivos tiny (versao mais leve para rodar)
- coco.names com as classes catalogadas pelo yolo
'''

import cv2 # biblioteca para imagens/abre a cam
import numpy as np #matrizes (modo que a rede neural encontra os "pontos")

print(cv2.__version__) 
'''precisei atualizar pip e atualizar o cv2 -> pip3 install opencv-python==4.8.0.76
pois estava ocorrendo um erro de  cv2.error: Unknown C++ exception from OpenCV code'''

# Carregando o modelo YOLO pré-treinado.
''' Os argumentos são o arquivo de pesos (yolov4-tiny.weights) e
o arquivo de configuração da rede (yolov4-tiny.cfg).'''
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg") #o tiny é uma versao mais leve

# Carrega as classes (rótulos) do modelo
'''Lê o arquivo coco.names que contém os nomes das classes, armazena em classes cada uma das classes'''
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Inicializa a webcam
cap = cv2.VideoCapture(0)

'''Loop para capturar continuamente quadros da webcam e realizar a detecção de objetos.'''
while True:
    ret, frame = cap.read()

    # Detecte objetos no quadro
    '''Para cada quadro, pré-processa a imagem para torná-la adequada para o modelo YOLO'''
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    '''Explicando:
    o blob resulta em um "bloco" de imagem (um tensor) que está pronto
    para ser inserido no modelo YOLO para detecção de objetos.
    - frame = a captura
    - 0.00392 = fator de escala que é aplicado a todos os valores dos pixels
    - (416,416) = novos tamanhos da imagem, YOLO opera com imagens de tamanho fixo
    - (0, 0, 0) = valor de preenchimento que é adicionado à imagem para ajustá-la ao tamanho desejado
    Neste caso, nenhum preenchimento é adicionado (todos os valores são zero).
    - True: Isso indica que a imagem de entrada deve ser normalizada
    - crop=False: Isso indica que a imagem não deve ser cortada. Se crop fosse
    definido como True, a imagem seria cortada para se ajustar ao tamanho desejado (416x416),
    '''

    '''define a entrada da rede neural,
    o blob processado anteriormente'''
    net.setInput(blob)
    
    '''realiza a detecção
    forward passa a imagem de entrada e obtem as saidas'''
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Amarzenar as informaçoes das deteccoes
    class_ids = [] # IDs das classes dos objetos detectados
    confidences = [] #armazena as confianças associadas a cada detecção (%)
    boxes = [] #armazena as coordenadas das caixas delimitadoras dos objetos.

    for out in outs:
        '''itera pelas saídas do modelo YOLO'''

        for detection in out:
            '''Dentro de cada saída, o código itera pelas detecções individuais.'''

            '''O modelo YOLO fornece várias informações para
            cada detecção, incluindo as probabilidades (scores) de
            pertencer a cada classe. Essas probabilidades são armazenadas
            em scores, começando a partir do índice 5 do array detection.'''
            scores = detection[5:]

            '''A classe com a maior probabilidade é determinada usando np.argmax'''
            class_id = np.argmax(scores)

            '''confiança (probabilidade) da detecção é obtida a partir das probabilidades das classes.'''
            confidence = scores[class_id]


            if confidence > 0.5:

                #Calcula os dados para poder desenhar a caixa delimitadora
                #coloca eles em uma lista para "plotar" depois
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1]) 
                h = int(detection[3] * frame.shape[0]) 
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Aplicar não máxima supressão para evitar duplicatas
    #** eliminar detecções duplicadas ou sobrepostas
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i] #coordenadas
            label = str(classes[class_ids[i]]) # classe do objeto detectado
            confidence = confidences[i] #probalidadade, confianca
            color = (0, 255, 0)  # Cor da caixa (verde)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) #desenhando retangulo
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) #escrevendo classe

    # Mostrar as detecções na cam
    cv2.imshow("YOLO Webcam", frame)

    # 'q' (quit) - qpara sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a webcam e feche a janela
cap.release()
cv2.destroyAllWindows()
