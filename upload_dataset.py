import os
import glob
from roboflow import Roboflow

# ==========================================
# 1. CREDENCIAIS DO ROBOFLOW
# ==========================================
API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
PROJECT = os.getenv("ROBOFLOW_PROJECT")
VERSION = int(os.getenv("ROBOFLOW_VERSION", 1))

# Inicializa o cliente do Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)

# Caminho para o dataset
dataset_path = "dataset_sku110k"
splits = ["train", "valid", "test"]

print("A iniciar o upload para o Roboflow...")

for split in splits:
    images_dir = os.path.join(dataset_path, split, "images")
    labels_dir = os.path.join(dataset_path, split, "labels")
    
    # Ignora caso a pasta não exista (ex: se não houver test)
    if not os.path.exists(images_dir):
        continue
        
    print(f"\n--- A processar a pasta: {split.upper()} ---")
    
    # Procura todas as imagens na pasta (suporta jpg, jpeg, png)
    for img_path in glob.glob(os.path.join(images_dir, "*.*")):
        if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        # Extrai o nome do ficheiro sem a extensão (ex: 'imagem1')
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Calcula onde o ficheiro de texto da label deve estar
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        try:
            # Se existir o ficheiro txt, envia a imagem JUNTAMENTE com a label
            if os.path.exists(label_path):
                print(f"A enviar {base_name} (COM LABEL)...")
                project.upload(
                    image_path=img_path,
                    annotation_path=label_path,
                    split=split,
                    num_retry_uploads=3
                )
            else:
                # Se não existir label, envia apenas a imagem vazia (ex: imagens de fundo)
                print(f"A enviar {base_name} (SEM LABEL)...")
                project.upload(
                    image_path=img_path,
                    split=split,
                    num_retry_uploads=3
                )
        except Exception as e:
            print(f"Erro ao enviar a imagem {base_name}: {e}")

print("\nUpload concluído com sucesso!")
