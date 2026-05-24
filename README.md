# IA_25_26

Este repositório contém o projeto de suporte ao projeto prático de IA, para o ano letivo 25/26.

É um projeto de Visão Computacional baseado no modelo YOLO (You Only Look Once), versão 8.

Para facilitar o desenvolvimento, este projeto usa as seguintes bibliotecas principais:
- [Ultralytics](https://docs.ultralytics.com/quickstart/): A biblioteca Ultralytics é uma ferramenta open source focada em Visão Computacional e, especialmente, no treino e uso de modelos YOLO (You Only Look Once) para detecção de objetos, segmentação de instâncias e classificação de imagens. Esta biblioteca simplifica o uso da família de modelos YOLO (e.g. YOLOv5, YOLOv8, YOLOv11), acelerando significativamente o desenvolvimento de aplicações.
- [Roboflow](https://roboflow.com/): A biblioteca e plataforma cloud Roboflow facilita o desenvolvimento de projetos de Visão Computacional, especialmente para treino, gestão e implementação de modelos de detecção de objetos, segmentação e classificação de imagens. No caso deste trabalho prático, será útil para a criação de datasets etiquetados.


## Getting started

O principal ficherio deste repositório é o tutorial_yolo.ipynb.

Este notebook python exemplifica o processo completo de download de um dataset da plataforma Ultralytics, a sua utilização para fine-tuning de um modelo pré-treinado, e a utilização do modelo resultante para fazer previsões em imagens.

Para correr o tutorial, devem ser levados a cabo os seguintes passos:
- (Aconselhado) Criar um ambiente python para este projeto (pyenv ou conda)
- Instalar as bibliotecas necessárias se não estiverem já instaladas
- Criar uma conta, workspace e projeto na plataforma Roboflow, para etiquetagem de imagens
- Fazer upload das imagens fornecidas, etiquetar os objetos pretendidos, e criar uma primeira versão do dataset
- Na secção de treino do modelo do notebook, fazer as alterações necessárias relativamente à API key, nome do projeto e nome do workspace
- O dataset será descarregado para a pasta local do projeto. Pode ser necessário corrigir os paths na configuração do dataset (ficheiro data.yaml)
- Descarregar um modelo YOLO v8 pré treinado, para acelerar o processo de treino. Uma lista de modelos disponíveis publicamente pode ser encontrada [aqui](https://docs.ultralytics.com/models/yolov8/#performance-metrics). Note que quanto maior o número de parâmetros, mais capaz o modelo, mas mais demorados serão os processos de treino e inferência. Guardar localmente o ficheiro descarregado (modelo com extensão .pt)
- Por último, treinar o modelo, e no fim testá-lo em algumas imagens
