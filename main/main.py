import pandas as pd
print("\nLoading PEMS04 data...")
speed_matrix = pd.read_csv('pems04_flow.csv',sep=',')
A = np.load('pems04_adj.npy')


print("\nPreparing train/test data...")
train_dataloader, valid_dataloader, test_dataloader, max_value = PrepareDataset(speed_matrix, BATCH_SIZE=48)

print("\nTraining STGmambomer model...")
STGmamba, STGmamba_loss = TrainSTG_Mamba(train_dataloader, valid_dataloader, A, K=3, num_epochs=25, mamba_features=307)
print("\nTesting STGmambomer model...")
results = TestSTG_Mamba(STGmamba, test_dataloader, max_value)
