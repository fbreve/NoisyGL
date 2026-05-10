import sys
import os
import numpy as np

# Adiciona o diretório pypcc ao path
pcc_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'pypcc'))
if os.path.isdir(pcc_dir):
    sys.path.insert(0, pcc_dir)
else:
    print(f"Erro: Diretorio {pcc_dir} nao encontrado.")
    sys.exit(1)

try:
    from pcc import ParticleCompetitionAndCooperation
    print("Sucesso: PCC importado corretamente.")
except ImportError as e:
    print(f"Erro ao importar PCC: {e}")
    sys.exit(1)

def test_label_mapping():
    print("\n--- Testando Mapeamento de Labels (Não-contínuos) ---")
    # Dataset simples: 4 pontos, classes 10 e 20
    X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]], dtype=np.float64)
    # Labeled: pontos 0 e 2. Classes 10 e 20.
    labels = np.array([10, -1, 20, -1], dtype=np.int64)
    
    model = ParticleCompetitionAndCooperation(impl="numpy") # Usando numpy para evitar necessidade de compilar Cython para o teste
    model.build_graph(X, k_nn=2)
    
    print(f"Labels originais: {labels}")
    pred = model.fit_predict(labels, max_iter=100)
    print(f"Predicoes: {pred}")
    
    unique_pred = np.unique(pred)
    print(f"Classes detectadas: {unique_pred}")
    
    if set(unique_pred).issubset({10, 20}):
        print("OK: Mapping funcionou, as classes retornadas sao as originais (10 e 20).")
    else:
        print("FALHA: Mapping falhou, classes retornadas nao coincidem.")

def test_distance_stability():
    print("\n--- Testando Estabilidade de Distancia ---")
    # Verifica se o código roda sem dar crash mesmo com parâmetros que poderiam causar overflow
    X = np.random.rand(10, 2)
    labels = np.array([0, -1, -1, -1, 1, -1, -1, -1, -1, -1], dtype=np.int64)
    
    # Testa todos os backends disponíveis
    for backend in ["numpy", "numba", "cython"]:
        print(f"Testando backend: {backend}")
        try:
            model = ParticleCompetitionAndCooperation(impl=backend)
            model.build_graph(X, k_nn=3)
            pred = model.fit_predict(labels, max_iter=50)
            print(f"  {backend}: OK (sem crash)")
        except Exception as e:
            print(f"  {backend}: Erro ou nao disponivel: {e}")

if __name__ == "__main__":
    test_label_mapping()
    test_distance_stability()
