from models.loader.GNVSTDataset import MyGraphDataset
from torch_geometric.loader import DataLoader

if __name__ == "__main__":
    dataset = MyGraphDataset(root='data', time_step=4)
    print(f"Number of graphs in the dataset: {len(dataset)}")
    print(f"First graph data: {dataset[0]}")

    #loader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for batch in loader:
        print(f"Batch data shape: {batch.x.shape}, Edge index shape: {batch.edge_index.shape}, Edge attr shape: {batch.edge_attr.shape},  y shape: {batch.y.shape}, weather shape: {batch.weather.shape}")
        print(f"Batch y shape: {batch.y.shape}")
        print(f'batch.num_graphs: {batch.num_graphs}')
        print(batch.x[:5, :])
        break  # 첫 번째 배치만 출력