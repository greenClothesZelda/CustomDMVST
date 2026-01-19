import torch
import tqdm
from models.loader.GNVSTDataset import MyGraphDataset

class VectorizedRetriever:
    def __init__(self, train_dataset, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Initializing Retriever on {self.device}...")
        
        # 1. 데이터베이스 구축 (Flattening)
        # Train Set의 모든 그래프의 모든 노드를 하나의 큰 텐서로 합칩니다.
        # db_keys: (Total_Train_Nodes, T)
        # db_values: (Total_Train_Nodes, 1)
        db_keys = []
        db_values = []
        
        print("Building Database Tensor...")
        for data in train_dataset:
            # data.x shape: [Nodes, T, Features] -> [Nodes, T] (feature 0만 사용 시)
            db_keys.append(data.x[:, :, 0]) 
            db_values.append(data.y.unsqueeze(1)) # [Nodes, 1]
            
        self.db_keys = torch.cat(db_keys, dim=0).to(self.device)
        self.db_values = torch.cat(db_values, dim=0).to(self.device)
        
        # 2. 정규화 (Cosine Sim 계산 시 반복을 피하기 위해 미리 계산)
        # L2 Norm: ||k||
        self.db_norms = torch.norm(self.db_keys, dim=1, keepdim=True) + 1e-8
        
        print(f"Database Ready: {self.db_keys.shape[0]} sequences loaded.")

    def predict_batch(self, queries, top_k=50):
        """
        queries: (Batch_Size, T) - 테스트할 시계열 배치
        top_k: 상위 K개의 유사한 시계열만 사용하여 예측 (노이즈 제거 및 속도 향상)
        """
        queries = queries.to(self.device)
        
        # --- 1. Cosine Similarity (행렬 연산) ---
        # Sim(Q, K) = (Q @ K.T) / (||Q|| * ||K||)
        q_norms = torch.norm(queries, dim=1, keepdim=True) + 1e-8
        
        # (Batch, T) @ (T, Total_DB) -> (Batch, Total_DB)
        dot_product = queries @ self.db_keys.T
        similarity = dot_product / (q_norms @ self.db_norms.T)
        
        # --- 2. Top-K Retrieval (선택 사항) ---
        # 전체 평균보다는 유사도가 높은 상위 K개의 값만 참조하는 것이 IR에서 더 효과적입니다.
        # top_k가 None이면 전체 가중 평균 (기존 로직과 유사하지만 전역적)
        if top_k:
            values, indices = torch.topk(similarity, k=top_k, dim=1)
            # Softmax를 사용하여 유사도가 높은 것에 더 큰 가중치 부여
            weights = torch.softmax(values, dim=1) 
            
            # 해당하는 Y값 가져오기
            # indices: (Batch, K)
            retrieved_y = self.db_values[indices].squeeze(-1) # (Batch, K)
            
            # Weighted Sum
            pred = (weights * retrieved_y).sum(dim=1)
        else:
            # 기존 로직과 유사한 선형 가중치 (전체 DB 대상)
            # 주의: 음수 유사도 처리가 필요할 수 있어 Softmax 권장
            weights = torch.softmax(similarity, dim=1)
            pred = (weights @ self.db_values).squeeze(-1)
            
        return pred

if __name__ == "__main__":
    # 데이터셋 로드
    dataset = MyGraphDataset(root='./data', time_step=8)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size + val_size:] # Validation 제외하고 바로 Test

    # Retriever 초기화 (DB 구축)
    retriever = VectorizedRetriever(train_dataset)

    print(f'\nIn train set \n ================')
    total_error = 0.0
    total_count = 0
    
    # Test Loop
    # 그래프 단위가 아니라 배치 처리가 가능하도록 구조를 잡으면 더 빠르지만,
    # 여기서는 그래프 단위 루프는 유지하되 내부 연산을 최적화했습니다.
    with torch.no_grad():
        for data in tqdm.tqdm(test_dataset, desc="Fast Augmenting"):
            # 현재 그래프의 모든 노드를 한 번에 쿼리로 만듭니다.
            # queries shape: (Num_Nodes_In_Graph, T)
            queries = data.x[:, :, 0] 
            targets = data.y.to(retriever.device)
            
            # 배치 예측 (루프 없이 한 번에 계산)
            preds = retriever.predict_batch(queries, top_k=30)
            
            # 오차 계산
            error = torch.abs(preds - targets).sum().item()
            total_error += error
            total_count += data.num_nodes

    print(f'max demand in test set: {dataset.max_demand} num_nodes: {dataset[0].x.size(0)}')
    mae = total_error / total_count * dataset.max_demand * dataset[0].x.size(0) + dataset.dropped_point
    print(f'Test MAE after augmentation: {mae:.6f}')