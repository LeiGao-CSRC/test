# random_mlp_tb_demo.py
# pip install torch tensorboard

import os
from datetime import datetime
import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
# import gdb
import logging

# ----- 1) 一个小型全连接网络 -----
class TinyMLP(nn.Module):
    def __init__(self, in_dim=100, hidden=64, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ----- 2) 工具函数：记录参数和梯度 -----
def log_params_and_grads(model: nn.Module, writer: SummaryWriter, global_step: int):
    for name, p in model.named_parameters():
        writer.add_histogram(f"params/{name}", p.data.cpu(), global_step)
        writer.add_scalar(f"params_stats/{name}_l2norm", p.data.norm(2).item(), global_step)
        writer.add_scalar(f"params_stats/{name}_absmax", p.data.abs().max().item(), global_step)

        if p.grad is not None:
            writer.add_histogram(f"grads/{name}", p.grad.cpu(), global_step)
            writer.add_scalar(f"grads_stats/{name}_l2norm", p.grad.data.norm(2).item(), global_step)
            writer.add_scalar(f"grads_stats/{name}_absmax", p.grad.data.abs().max().item(), global_step)

# ----- 3) 训练主循环（随机数据） -----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyMLP(in_dim=100, hidden=64, num_classes=10).to(device)
    for name, param in model.named_parameters():
        print(f"Name: {name}")
        print(f"Parameter: {param.shape}")
    criterion = nn.CrossEntropyLoss()
    lr = 0.1
    # gdb.set_trace()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # TensorBoard logdir
    logdir = os.path.join("runs", "random_mlp_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(logdir)
    print(f"TensorBoard logdir: {logdir}")

    steps = 200
    batch_size = 32
    for step in range(1, steps + 1):
        # 随机输入和标签
        x = torch.randn(batch_size, 100).to(device)
        y = torch.randint(0, 10, (batch_size,), device=device)

        # 前向、反向、更新
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # 记录损失 & 参数/梯度
        writer.add_scalar("train/loss", loss.item(), step)
        log_params_and_grads(model, writer, step)

        if step % 20 == 0:
            print(f"[Step {step}] loss = {loss.item():.4f}")

    writer.close()
    print("Done. Launch TensorBoard with:\n  tensorboard --logdir runs")

if __name__ == "__main__":
    main()
