# -*-coding:utf-8-*-
# author lyl
import paddle
import paddle.nn as nn
from resnet import ResNet18
from dataset import get_dataset
from dataset import get_dataloader
from utils import AverageMeter


def train_one_epoch(model, dataloader, criterion, optimizer, total_epoch):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_id, batch in enumerate(dataloader):
        image, label = batch[0], batch[1]

        out = model(image)
        loss = criterion(out, label)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        pred = nn.functional.softmax(out, axis=1)
        acc = paddle.metric.accuracy(pred, label.unsqueeze(-1))

        batch_size  = image.shape[0]
        loss_meter.update(loss.numpy()[0], batch_size)
        acc_meter.update(acc.numpy()[0], batch_size)
        if batch_size > 0:
            print(f"----Batch {batch_id}, loss={loss_meter.avg}, acc={acc_meter.avg}")


def validation(model, dataloader, criterion):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_id, batch in enumerate(dataloader):
        image, label = batch[0], batch[1]

        out = model(image)
        loss = criterion(out, label)

        pred = nn.functional.softmax(out, axis=1)
        acc = paddle.metric.accuracy(pred, label.unsqueeze(-1))

        batch_size = image.shape[0]
        loss_meter.update(loss.numpy()[0], batch_size)
        acc_meter.update(acc.numpy()[0], batch_size)
        if batch_size > 0:
            print(f"----Batch {batch_id}, loss={loss_meter.avg}, acc={acc_meter.avg}")

def main():
    total_epoch = 200
    batch_size = 16

    model = ResNet18(num_classes=10)
    train_dataset = get_dataset(mode='train')
    train_dataloader = get_dataloader(dataset=train_dataset, mode='train', batch_size=batch_size)
    val_dataset = get_dataset(mode='test')
    val_dataloader = get_dataloader(dataset=val_dataset, mode='test', batch_size=1)

    criterion = nn.CrossEntropyLoss()
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(0.02, total_epoch)

    optimizer = paddle.optimizer.Momentum(learning_rate=scheduler,
                                          parameters=model.parameters(),
                                          momentum=0.9,
                                          weight_decay=5e-4)

    save_freq = 50
    test_freq = 10
    for epoch in range(1, total_epoch+1):
        train_one_epoch(model, train_dataloader, criterion, optimizer, total_epoch)
        scheduler.step()

        if epoch % test_freq == 0 or epoch == total_epoch:
            validation(model, val_dataloader, criterion)

        if epoch % save_freq == 0 or epoch == total_epoch:
            paddle.save(model.state_dict(), f"./resnet18_ep{epoch}.pdparams")
            paddle.save(optimizer.state_dict(), f"./renet18_ep{epoch}.pdot")


if __name__ == '__main__':
    main()