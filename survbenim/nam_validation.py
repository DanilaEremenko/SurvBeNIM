import numpy as np
import torch
import matplotlib.pyplot as plt
from nam_esimators import MyFeatureNN, ReLULayer, SigmoidLayer, ExULayer

x = torch.tensor(np.random.random((100)), dtype=torch.float32)
y = torch.tensor(np.random.random((100)), dtype=torch.float32)
last_layers = [ExULayer, ReLULayer, SigmoidLayer]
fig, axes = plt.subplots(nrows=2, ncols=len(last_layers), figsize=(len(last_layers * 3), 6))
for row_i, last_layer in enumerate(last_layers):
    feature_nn = MyFeatureNN(
        layers=[ReLULayer, ReLULayer, last_layer],
        layers_args=[
            dict(in_features=1, out_features=64),
            dict(in_features=64, out_features=32),
            dict(in_features=32, out_features=1024)
        ]
    )

    optimizer = torch.optim.Adam(feature_nn.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()
    batch_step = 16
    loss_history = []
    best_loss = 1e5

    for i in range(500):
        for batch_start in range(0, len(x), batch_step):
            optimizer.zero_grad()
            loss = criterion(
                y[batch_start:batch_start + batch_step],
                feature_nn(x[batch_start:batch_start + batch_step])
            )
            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(feature_nn.state_dict(), 'best_val.pt')
            loss_history.append(loss_val)
            print(f'loss = {loss_val}')
            loss.backward()
            optimizer.step()

    feature_nn.load_state_dict(torch.load('best_val.pt'))

    axes[0, row_i].plot(loss_history)
    axes[0, row_i].set_title(f"{last_layer.__name__}\n loss history")

    x_sorted = torch.sort(x, dim=0)[0]
    axes[1, row_i].scatter(x.detach().numpy(), y.detach().numpy(), label='data')
    axes[1, row_i].plot(x_sorted.detach().numpy(), feature_nn(x_sorted).detach().numpy(), color='orange')
    axes[1, row_i].set_title(f"model with {last_layer.__name__}\nmse = {loss_val:.2f}")

plt.tight_layout()
plt.show()
