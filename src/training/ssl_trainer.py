import torch
import logging
import time
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

# 设置后端以在没有GUI的服务器上生成图表
matplotlib.use('Agg')

from src.utils.visualization import plot_training_curves

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_ssl_model(model, train_loader, val_loader, optimizer, device, config, resume=False):
    """
    训练SSL模型

    Args:
        model: SSL模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        device: 训练设备
        config: 配置参数
        resume: 是否从上次中断处继续训练

    Returns:
        model: 训练后的模型
    """
    logger.info("Starting SSL model training")

    epochs = config['ssl_epochs']
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = checkpoint_dir / 'best_ssl_model.pt'
    last_model_path = checkpoint_dir / 'last_ssl_model.pt'
    train_state_path = checkpoint_dir / 'ssl_train_state.json'

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_epoch = 0

    # 断点续训
    if resume and last_model_path.exists() and train_state_path.exists():
        logger.info(f"Resuming training from {last_model_path}")

        checkpoint = torch.load(last_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        with open(train_state_path, 'r') as f:
            train_state = json.load(f)

        start_epoch = train_state['epoch']
        train_losses = train_state['train_losses']
        val_losses = train_state['val_losses']
        best_val_loss = train_state['best_val_loss']

        logger.info(f"Resuming training from epoch {start_epoch}/{epochs}")
    else:
        logger.info("Starting training from scratch")

    model.train()
    start_time = time.time()

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0

            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
                for batch in pbar:
                    views = batch['views']
                    view1, view2 = views[0].to(device), views[1].to(device)

                    _, proj1 = model(view1)
                    _, proj2 = model(view2)

                    loss = model.contrastive_loss(proj1, proj2)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    views = batch['views']
                    view1, view2 = views[0].to(device), views[1].to(device)

                    _, proj1 = model(view1)
                    _, proj2 = model(view2)

                    loss = model.contrastive_loss(proj1, proj2)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            logger.info(f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }
                torch.save(checkpoint, best_model_path)
                logger.info(f"Saved best model, validation loss: {best_val_loss:.4f}")

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }
            torch.save(checkpoint, last_model_path)

            train_state = {
                'epoch': epoch + 1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'time_elapsed': time.time() - start_time
            }

            with open(train_state_path, 'w') as f:
                json.dump(train_state, f)

            if (epoch + 1) % 10 == 0:
                torch.save(checkpoint, checkpoint_dir / f'ssl_model_epoch_{epoch + 1}.pt')

    except KeyboardInterrupt:
        logger.info("Training interrupted by user! Saving current state for later resumption...")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss if 'avg_val_loss' in locals() else float('inf'),
        }
        torch.save(checkpoint, last_model_path)

        train_state = {
            'epoch': epoch,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'time_elapsed': time.time() - start_time
        }

        with open(train_state_path, 'w') as f:
            json.dump(train_state, f)

        logger.info(f"Saved interrupt checkpoint, can resume training with --resume flag")
        return model

    # 绘制训练曲线
    plot_training_curves(train_state, checkpoint_dir, plot_type='ssl')

    torch.save(model.state_dict(), checkpoint_dir / 'final_ssl_model.pt')

    training_time = time.time() - start_time
    logger.info(f"SSL training completed in {training_time / 60:.2f} minutes")

    return model