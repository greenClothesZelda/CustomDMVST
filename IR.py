import torch
import tqdm
from models.loader.DemandDataset import MyGraphDataset

class NodeSpecificRetriever:
    def __init__(self, train_dataset, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Initializing Node-Specific Retriever on {self.device}...")
        
        # --- 1. 데이터 구조 변환 (Stacking) ---
        # 목표: (Num_Nodes, Num_Train_Samples, T) 형태로 만듭니다.
        # 이렇게 하면 'Num_Nodes'를 Batch Dimension으로 사용하여 병렬 처리가 가능합니다.
        
        x_list = []
        y_list = []
        
        # 데이터를 메모리에 로드
        print("Stacking Database Tensor by Node...")
        for data in train_dataset:
            # data.x: (Nodes, T, Feature) -> (Nodes, T)
            x_list.append(data.x[:, :, 0]) 
            # data.y: (Nodes,) -> (Nodes, 1)
            y_list.append(data.y.unsqueeze(1))
            
        # Stack result: (Num_Train_Samples, Nodes, T)
        # -> Permute to: (Nodes, Num_Train_Samples, T)
        self.db_keys = torch.stack(x_list).permute(1, 0, 2).to(self.device)
        self.db_values = torch.stack(y_list).permute(1, 0, 2).to(self.device)
        
        self.num_nodes = self.db_keys.shape[0]
        self.num_samples = self.db_keys.shape[1]

        print(f"Database stacked: Nodes={self.num_nodes}, Samples={self.num_samples}, Time_Steps={self.db_keys.shape[2]}")
        # --- 2. 정규화 미리 계산 ---
        # (Nodes, Num_Train_Samples, 1)
        self.db_norms = torch.norm(self.db_keys, dim=2, keepdim=True) + 1e-8
        
        print(f"DB Ready. Shape: [Nodes: {self.num_nodes}, History: {self.num_samples}]")

    def augment_graph(self, query_graph_x, top_k=30):
        """
        한 시점의 그래프 전체에 대해 Augmentation을 수행합니다.
        query_graph_x: (Nodes, T)
        """
        # (Nodes, T) -> (Nodes, 1, T) : bmm을 위한 차원 추가
        queries = query_graph_x.to(self.device).unsqueeze(1)
        
        # --- 1. Node-wise Cosine Similarity (Batch MatMul) ---
        # Q: (Nodes, 1, T)
        # K.T: (Nodes, T, Samples)  <-- db_keys를 transpose
        # Result: (Nodes, 1, Samples)
        # 각 노드는 자신의 배치(자신의 과거 데이터)하고만 연산됩니다.
        
        dot = torch.bmm(queries, self.db_keys.transpose(1, 2))
        
        q_norm = torch.norm(queries, dim=2, keepdim=True) + 1e-8 # (Nodes, 1, 1)
        
        # (Nodes, 1, Samples)
        similarity = dot / (q_norm * self.db_norms.transpose(1, 2))
        
        # --- 2. Top-K Retrieval ---
        # 유사도 차원(dim=2)에서 상위 K개 추출
        if top_k:
            values, indices = torch.topk(similarity, k=top_k, dim=2)
            weights = torch.softmax(values, dim=2) # (Nodes, 1, K)
            
            # 값 가져오기 (Gather)
            # self.db_values: (Nodes, Samples, 1)
            # indices: (Nodes, 1, K) -> (Nodes, K, 1)로 맞춰서 gather 해야 함
            indices_expanded = indices.permute(0, 2, 1) # (Nodes, K, 1)
            
            # 각 노드별로 해당하는 인덱스의 값을 가져옴
            retrieved_values = torch.gather(self.db_values, 1, indices_expanded) # (Nodes, K, 1)
            
            # Weighted Sum: (Nodes, 1, K) @ (Nodes, K, 1) -> (Nodes, 1, 1)
            augmented_val = torch.bmm(weights, retrieved_values).squeeze()
            
        else:
            weights = torch.softmax(similarity, dim=2) # (Nodes, 1, Samples)
            augmented_val = torch.bmm(weights, self.db_values).squeeze()
            
        return augmented_val # (Nodes, )

if __name__ == "__main__":
    dataset = MyGraphDataset(root='./data', time_step=8)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size + val_size:]

    # Node-Specific Retriever 초기화
    # 주의: Train Dataset 내의 모든 그래프는 노드 순서가 동일해야 합니다.
    retriever = NodeSpecificRetriever(train_dataset)

    print(f'In train set \n ================')
    total_error = 0.0
    total_count = 0
    
    # Validation Loop
    with torch.no_grad():
        for data in tqdm.tqdm(test_dataset, desc="Node-wise Augmenting"):
            # 현재 그래프의 입력 (Nodes, T)
            query_x = data.x[:, :, 0]
            target_y = data.y.to(retriever.device)
            
            # 노드별 개별 검색 및 예측 (Loop 없음, 병렬 처리)
            preds = retriever.augment_graph(query_x, top_k=20)
            
            # 에러 계산
            error = torch.abs(preds - target_y).sum().item()
            total_error += error
            total_count += data.num_nodes

    print(f'max demand in test set: {dataset.max_demand} num_nodes: {dataset[0].x.size(0)}')
    mae = total_error / total_count * dataset.max_demand * dataset[0].x.size(0) + dataset.dropped_point
    print(f'Test MAE after augmentation: {mae:.6f}')