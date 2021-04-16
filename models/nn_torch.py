import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from ax.service.managed_loop import optimize

# configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# global variables
input_size = 45
hidden_size1 = 128
hidden_size2 = 64
num_classes = 1
num_epochs=300
batch_size=64
lr = 0.001
num_folds = 5

# create custom dataset
class CustomDataset:
    def __init__(self, file_out):
        x = file_out.iloc[::,1:-1].values
        y = file_out.iloc[::,-1].values

        self.X_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

# read in data
train_raw_df = pd.read_csv('../data/df_train_standardized_45_columns.csv')
test_raw_df = pd.read_csv('../data/df_test_standardized_45_columns.csv')


# build network architecture
class NeuralNet(nn.Module):
    def __init__(self, parameterization):
        super(NeuralNet, self).__init__()
        self.relu = nn.ELU(parameterization.get('alpha', 1.0))
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(p=parameterization.get('dropout1', 0.2))
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(p=parameterization.get('dropout2', 0.2))
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

def net_train(net, train_loader, parameters, device):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=parameters.get('lr', 0.001))

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(parameters.get('step_size', 30)),
        gamma=parameters.get('gamma', 0.1),
    )

    for _ in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            outputs = net(inputs.float())
            loss = criterion(outputs, labels.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return net

def evaluate(model, loader):
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for i in loader:
            data,target = i
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = np.round(output)
            target = target.float()
            y_true.extend(target.tolist()) 
            y_pred.extend(pred.reshape(-1).tolist())
    return accuracy_score(y_true, y_pred)

def train_evaluate(parameterization):
    kfolds = KFold(n_splits=num_folds, shuffle=True) 
    res = []
    for fold, (train_ids, val_ids) in enumerate(kfolds.split(train_raw_df)):
        train_df = train_raw_df.iloc[train_ids]
        val_df = train_raw_df.iloc[val_ids]
        train_dataset= CustomDataset(train_df)
        val_dataset = CustomDataset(val_df)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=parameterization.get('batch_size', batch_size),
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=parameterization.get('batch_size', batch_size),
            shuffle=True
        )
        init_net = NeuralNet(parameterization)
        trained_net = net_train(net=init_net,
                            train_loader=train_loader,
                            parameters=parameterization,
                            device=device)

        res.append(evaluate(trained_net, val_loader))
    mean = sum(res) / len(res)
    print(mean)
    return mean

def main():
    # Bayesian Optimization
    parameters=[
        {'name': 'alpha', 'type': 'range', 'bounds': [0.01, 10.0], 'log_scale': True},
        {'name': 'batch_size', 'type': 'choice', 'values': [4,8,16,32]},
        {'name': 'dropout1', 'type': 'range', 'bounds': [0.1, 0.5]},
        {'name': 'dropout2', 'type': 'range', 'bounds': [0.1, 0.5]},
        {'name': 'lr', 'type': 'range', 'bounds': [1e-6, 0.5], 'log_scale': True},
        {'name': 'step_size', 'type': 'range', 'bounds': [10,50]},
        {'name': 'gamma', 'type': 'range', 'bounds': [1e-4, 0.5], 'log_scale': True},
    ]

    best_params, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=train_evaluate,
        objective_name='accuracy',
        total_trials=20,
    )

    # result on test set
    train_set = CustomDataset(train_raw_df)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=best_params.get('batch_size', batch_size),
        shuffle=True
    )
    test_set = CustomDataset(test_raw_df)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=best_params.get('batch_size', batch_size),
        shuffle=True
    )
    test_model = net_train(net=NeuralNet(best_params),
                        train_loader=train_loader,
                        parameters=best_params,
                        device=device)
    test_res = evaluate(test_model, test_loader)
    print(f"Test Accuracy: {test_res}")


    '''
    Save all relavent information to local
    '''
    torch.save(test_model, './nn.pth')

    with open('./log', 'w') as outfile:
        outfile.write('best params: ' + str(best_params) + '\n')
        outfile.write('values on validation set: ' + str(values) + '\n')
        outfile.write('result on test set: ' + str(test_res) + '\n')
        outfile.close()

if __name__ == '__main__':
    main()