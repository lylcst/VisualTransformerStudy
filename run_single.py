import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from paddle.vision.transforms import ToTensor
from vi_transformers import VisualTransformer, main


num_epoch = 50
batch_size = 8
learning_rate=0.0001


transform = ToTensor()
cifat10_train = paddle.vision.datasets.Cifar10(mode='train', transform=transform)
cifat10_test = paddle.vision.datasets.Cifar10(mode='test', transform=transform)


def accuarcy(pred_logits, label):
    pred = paddle.argmax(pred_logits, -1)
    acc = paddle.sum((pred == label).astype(paddle.float32))

    return acc


def eval(model, eval_dataloader):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    global_loss = 0.
    global_acc = 0.
    data_size = 0
    batch_len = len(eval_dataloader)
    for batch_id, data in enumerate(eval_dataloader):
        image = data[0]
        label = paddle.to_tensor(data[1])

        out = model(image)[:, 0]

        loss = loss_fn(out, label.unsqueeze(-1))
        global_loss += loss.item()
        acc = accuarcy(out, label)
        global_acc += acc.cpu().item()
        data_size += len(label)
    
    accs = global_acc / data_size
    losses = global_loss / batch_len
    print('accuracy: {}, losses: {}'.format(accs, losses))


def save_model(model, epoch_id):
    # paddle.save(model.state_dict(), '/data/lyl/vit/data/model_{}.pdparams'.format(epoch_id))
    pass


def train(model, train_dataloader, eval_dataloader):
    
    optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch_id in range(num_epoch):
        model.train()
        for batch_id, data in enumerate(train_dataloader):
            image = data[0]
            label = paddle.to_tensor(data[1])

            out = model(image)[:, 0]

            loss = loss_fn(out, label.unsqueeze(-1))
            loss.backward()

            optimizer.step()
            optimizer.clear_grad()

            if batch_id > 0 and batch_id % 50 == 0:
                print('epoch_id: {}, batch_id:{}, loss:{}'.format(epoch_id, batch_id, loss.cpu().item()))
        
        eval(model, eval_dataloader)
        save_model(model, epoch_id)
            

if __name__ == '__main__':
    train_dataloader = DataLoader(cifat10_train, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(cifat10_test, shuffle=True, batch_size=batch_size)
    model = VisualTransformer(image_size=32, patch_size=8, in_channels=3, embed_dim=768, num_layers=6, num_classes=10, dropout=0.1)
    # model.set_state_dict(paddle.load('/data/lyl/vit/data/model_8.pdparams'))
    train(model, train_dataloader, eval_dataloader)

