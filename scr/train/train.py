import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from open_dataset import prepare_datasets
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
import segmentation_models_pytorch as smp


def save_checkpoints(epoch, model_state_dict, optimizer_state_dict, mean_loss, model_name, scheduler_state_dict):
    model_dst = f'/media/user/SP PHD U3/TSU_dataset/logs/{model_name}'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler': scheduler_state_dict,
        'loss': mean_loss,
    }, model_dst)

BATCH_SIZE = 1
WRITER_EPOCH = 10
start_epoch = 1
EPOCHS = 10  #
resolution=512

# Указать путь к чекпоинту для продолжения обучения с него
continue_with = None
# !!!!!!!!!!!!!!1
acc = 0.01
IoU = 0.01
# 1!!!!!!!!!!!!!!!!!
train_dataset, val_dataset_big = prepare_datasets(resolution=resolution)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader_big = DataLoader(val_dataset_big, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

del train_dataset
del val_dataset_big


iou_metric = MulticlassJaccardIndex(
    num_classes=10,
    average='none',
    ignore_index=None
).cpu()

accuracy_metric = MulticlassAccuracy(
    num_classes=10,
    average='micro'
).cpu()

model = smp.Segformer(
    encoder_name="resnext101_32x16d",
    encoder_weights=None,
    classes=10,
    in_channels=3,
    decoder_segmentation_channels=128,
    decoder_attn_channels=128,       # Attention для точных границ
).cuda()

optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4, weight_decay=1e-3)  # 0.001 для дообучения 0.00005
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, max_lr=1e-4, base_lr=1e-10, step_size_up=792*5, # 10/215
                                               step_size_down=792*45, mode='triangular', cycle_momentum=False) # на одну эпоху больше


class ComboLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = {'dice': 0.5, 'ce': 0.3, 'focal': 0.2}

        self.weights = weights
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', alpha=0.25, gamma=2.0)

    def forward(self, outputs, targets):
        """
        outputs: [B, C, H, W] — логиты от модели
        targets: [B, C, H, W] — one-hot маска
        """
        # Преобразуем one-hot -> индекс классов
        targets_argmax = targets.argmax(dim=1).long()  # [B, H, W]

        loss = 0.0

        if self.weights.get('dice', 0) > 0:
            # DiceLoss ожидает индексную маску при mode='multiclass'
            loss += self.weights['dice'] * self.dice_loss(outputs, targets_argmax)

        if self.weights.get('ce', 0) > 0:
            loss += self.weights['ce'] * self.ce_loss(outputs, targets_argmax)

        if self.weights.get('focal', 0) > 0:
            loss += self.weights['focal'] * self.focal_loss(outputs, targets_argmax)

        return loss

criterion = ComboLoss(weights={'dice': 0.6, 'ce': 0.3, 'focal': 0.1}).cuda()

if continue_with:
    checkpoint = torch.load(continue_with, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    # # # # Вытаскиваем из имени файла цихры и конвертим в int, добавляем единичку
    start_epoch = checkpoint['epoch']
    print(f'Предобученная модель успешно загружена, начинаем с {start_epoch} эпохи')
    if start_epoch >= 225:
        optimizer = torch.optim.RAdam(model.parameters(), lr=1e-5, weight_decay=1e-2)  # 0.001 для дообучения 0.00005
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, max_lr=1e-5, base_lr=1e-10, step_size_up=13200,
                                                      step_size_down=565*215, mode='triangular', cycle_momentum=False)

top_acc = 0
top_mean_loss = 200
best_val_iou = 0.0
overall_iou = 0.0
mean_ious = []

# Определяем параметры нормализации один раз
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

def normalize_tensor(tensor):
    """Нормализация тензора на GPU"""
    tensor = tensor.to('cuda')
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD

for epoch in range(start_epoch, EPOCHS + 1):

    if epoch == 50:
        optimizer = torch.optim.RAdam(model.parameters(), lr=1e-5, weight_decay=1e-2)  # 0.001 для дообучения 0.00005
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, max_lr=1e-5, base_lr=1e-10, step_size_up=792*5,
                                                      step_size_down=792*45, mode='triangular', cycle_momentum=False)


    for phase in 'train val_big'.split():
        if phase == 'train':
            model.train()
            torch.set_grad_enabled(True)
            loader = train_loader

        elif phase == 'val_big':
            model.eval()
            torch.set_grad_enabled(False)
            loader = val_loader_big

        if phase != 'train':
            iou_metric.reset()
            accuracy_metric.reset()


        running_ans = []
        running_pred = []
        running_loss = []
        running_acc = []

        running_acc_1 = []
        all_ious = []

        step = 0
        for batch in tqdm(loader, desc=f'{phase} loader in {epoch} epoch:'):
            X, y = batch
            X_normalized = normalize_tensor(X)
            output = model(X_normalized)
            loss = criterion(output, y.cuda())
            running_loss.append(loss.item())

            step += 1
            if phase != 'train' and ((epoch % WRITER_EPOCH == 0) or epoch == 1):
                running_ans = y.detach().to(torch.bool)
                running_pred = output.detach()

                running_pred = (torch.eq(running_pred, running_pred.max(1)[0].unsqueeze(1).repeat(1, 10, 1, 1)))
                running_acc_1.append(running_pred[running_ans])

                # Вычисляем IoU для каждого класса
                pred_classes = output.argmax(dim=1).cpu().detach() # [B, H, W]
                target_classes = y.argmax(dim=1).cpu()  # [B, H, W]
                accuracy_metric.update(pred_classes, target_classes)
                iou_metric.update(pred_classes, target_classes)

                del running_ans
                del running_pred

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                scheduler.get_last_lr()


        mean_loss = sum(running_loss) / len(running_loss)
        del running_loss

        if phase != 'train' and mean_loss < top_mean_loss:
            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            scheduler_state_dict = scheduler.state_dict()
            save_checkpoints(epoch, model_state_dict, optimizer_state_dict, mean_loss,
                             'top_loss.pt', scheduler_state_dict)
            top_mean_loss = mean_loss

        if phase != 'train' and ((epoch % WRITER_EPOCH == 0) or epoch == 1):
            class_ious = iou_metric.compute()  # [6] - IoU для каждого класса
            overall_iou = class_ious.mean().item()
            acc_2 = accuracy_metric.compute().item()

            print(f'точность {phase}_acc_2', f'{acc_2:.4f}')
            print(f'IoU {phase}_iou', f'{overall_iou:.4f}')
            print("Имена классов: ['Фон', 'Водные объекты', 'Дороги', 'Залежи(деревья)', 'Залежи(кустарник)', 'Залежи(травы)', "
                  "'Земли под трубопроводами, ЛЭП, связи, иными коммуникациями', "
                  "'Леса и древесно-кустарниковая растительность (естественная)',\n 'Пахотные земли', "
                  "'Участки, неиспользуемые ввиду особенностей рельефа, увлажнения, нарушенные и деградированные земли']")
            print(f'  Class IoUs: {[f"{iou:.4f}" for iou in class_ious.cpu().numpy()]}')

            # Сбрасываем метрики для следующей эпохи
            iou_metric.reset()
            accuracy_metric.reset()
            acc = torch.cat(running_acc_1, dim=0).to(torch.float32).mean().cpu()
            print(f'точность {phase}_acc', acc)
            del running_acc_1
            if acc > top_acc:
                model_state_dict = model.state_dict()
                optimizer_state_dict = optimizer.state_dict()
                scheduler_state_dict = scheduler.state_dict()
                save_checkpoints(epoch, model_state_dict, optimizer_state_dict, mean_loss,
                                 'top_model.pt', scheduler_state_dict)
                top_acc = acc

        if epoch and epoch % 25 == 0:
            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            scheduler_state_dict = scheduler.state_dict()
            save_checkpoints(epoch, model_state_dict, optimizer_state_dict, mean_loss,
                             f'{epoch:05d}.pt', scheduler_state_dict)

        torch.cuda.empty_cache()

torch.cuda.empty_cache()
