from roboflow import Roboflow

def download_dataset():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== Roboflow Dataset Downloader ===")
    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace = os.getenv("ROBOFLOW_WORKSPACE")
    project_name = os.getenv("ROBOFLOW_PROJECT")
    version = int(os.getenv("ROBOFLOW_VERSION", 1))
    if not api_key or not workspace or not project_name:
        print("Erro: Verifica se tens as variáveis ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE e ROBOFLOW_PROJECT no teu ficheiro .env")
        return
    
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        
        dataset = project.version(version).download("yolov8")
        
        import shutil
        import os
        
        pasta_origem = dataset.location
        pasta_destino = "submissao/dataset"
        
        if os.path.exists(pasta_destino):
            shutil.rmtree(pasta_destino)
            
        shutil.move(pasta_origem, pasta_destino)
        
        print(f"\nDownload concluído com sucesso!")
        print(f"O dataset foi movido e guardado em: {pasta_destino}")
        print("Podes agora usar a pasta submissao/dataset/ para o treino.")
    except Exception as e:
        print(f"\nOcorreu um erro ao fazer download: {e}")

if __name__ == "__main__":
    download_dataset()
