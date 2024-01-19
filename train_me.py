## train_me.py
from multiprocessing import freeze_support
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from network_me_rnn import ESP_PCT_Net
from dataset_me_new_npy_2 import Parsing_Pose_PC
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Import LR scheduler
from sklearn.metrics import confusion_matrix
import os
import sys
from thop import profile, clever_format
import torch
from thop import profile
from thop import clever_format

np.set_printoptions(threshold=sys.maxsize)

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
# 定义类别名称
class_names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']         

def train_model(num_epochs, learning_rate, batchsize):
    freeze_support()
    start_time = time.time()
    if torch.cuda.is_available():
        device = 'cuda:%d' % (0)
    else:
        device = 'cpu'
    print("device is {}".format(device))
    
    # num_epochs = 300
    # learning_rate = 0.0002
    # batchsize = 16
    length_size = 25
    num_class = 36
    early_stop_patience = 200  # 如果连续多少个epoch验证性能不提升，就停止训练
    best_validation_loss = float('inf')  # 初始化最佳验证集损失值
    counter = 0  # 用于计数连续多少个epoch验证性能没有提升
    writer1 = SummaryWriter(r'.\logs')
    path = r".\models"
    import os

    if not os.path.exists(path):
        os.mkdir(path)

    model = ESP_PCT_Net(num_class=num_class).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_data = Parsing_Pose_PC()
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=5)
    test_data = Parsing_Pose_PC(train=False)
    test_loader = DataLoader(test_data, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=5)

    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    modelLast = r'.\models\model_599.pth'
    if os.path.exists(modelLast):
        # checkpoint = torch.load(modelLast)
        # model.load_state_dict(checkpoint['model'])
        # Load the checkpoint
        checkpoint = torch.load(modelLast)
        # Get the state_dict meant for the model
        model_state_dict = checkpoint['model']
        # Remove the profiling keys from the state_dict
        model_state_dict = {k: v for k, v in model_state_dict.items() if "total_ops" not in k and "total_params" not in k}
        # Load the model state_dict
        model.load_state_dict(model_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Load epoch {} success！'.format(start_epoch))
    else:
        start_epoch = 0
        print('No saved model, start from begin')

    loss_total = 0
    for epoch in range(start_epoch, num_epochs):
        print("epoch: {}".format(epoch + 1))
        training_loss = []
        train_classify_accu_epoch = []

        model.train()
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            data = np.asarray(data)
            batch_size, seq_len, point_num, dim = data.shape  # batch_size16,25,96,5
            data_ti_in = np.reshape(data, (batch_size * seq_len, point_num, dim))  # 16×25,96,5
            data_ti = torch.tensor(data_ti_in, dtype=torch.float32, device=device).squeeze()  # 16×25,96,5
            target0 = torch.tensor(target, dtype=torch.long, device=device).squeeze()  # 16，25,96

            optimizer.zero_grad()
            h0 = torch.zeros((6, batchsize, 96//2), dtype=torch.float32, device=device)  # 3，16，96
            c0 = torch.zeros((6, batchsize, 96//2), dtype=torch.float32, device=device)
            
            predict_action = model(data_ti, h0, c0, num_class, batchsize, length_size)

            loss = F.cross_entropy(predict_action, target0)  # Use cross_entropy loss directly
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_norm as needed

            # classify accu
            pred_choice = predict_action.argmax(1)
            correct = pred_choice.eq(target0.data).cpu().sum()
            train_classify_accu = correct.item() / float(batch_size)  # npoints 96
            train_classify_accu_epoch.append(train_classify_accu)

        training_loss = np.mean(training_loss)
        train_classify_accu_epoch = np.mean(train_classify_accu_epoch)
        print('train_loss: {}'.format(training_loss))
        print('train_classify_accu: {}'.format(train_classify_accu_epoch))

        # if (epoch + 1) % 25 == 0:
        #     state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        #     torch.save(state, path + "/model_{}.pth".format(epoch))

        ### evaluation
        model.eval()
        eval_loss = []
        eval_classify_accu = []
        # 定义两个空列表，用于存储所有的预测标签和真实标签
        all_pred = []
        all_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
                data = np.asarray(data)
                batch_size, seq_len, point_num, dim = data.shape
                data_ti_in = np.reshape(data, (batch_size * seq_len, point_num, dim))

                data_ti = torch.tensor(data_ti_in, dtype=torch.float32, device=device).squeeze()
                target0 = torch.tensor(target, dtype=torch.long, device=device).squeeze()

                optimizer.zero_grad()
                h0 = torch.zeros((6, batchsize, 96//2), dtype=torch.float32, device=device)
                c0 = torch.zeros((6, batchsize, 96//2), dtype=torch.float32, device=device)
                length_size = 25
                predict_action = model(data_ti, h0, c0, num_class, batchsize, length_size)

                loss = F.cross_entropy(predict_action, target0)
                eval_loss.append(loss.item())

                # classify accu
                pred_choice = predict_action.argmax(1)
                correct = pred_choice.eq(target0.data).cpu().sum()
                eval_classify_accu0 = correct.item() / float(batch_size)  # npoints 96
                eval_classify_accu.append(eval_classify_accu0)
                # 将本批次的预测标签和真实标签添加到列表中
                all_pred.extend(pred_choice.cpu().numpy())
                all_true.extend(target0.cpu().numpy())
            # 将列表转换为numpy数组
            all_pred = np.array(all_pred)
            all_true = np.array(all_true)

            # 定义类别名称
            class_names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']

            # # Find unique classes in true labels
            # unique_classes = np.unique(all_true)
            # 在真实标签和预测标签中查找独特的类
            unique_classes = np.unique(np.concatenate((all_true, all_pred)))

            # Filter class names based on unique classes
            filtered_class_names = [class_names[i] for i in unique_classes]

            # 调用classification_report函数，输出每个类别的评估指标
            report = classification_report(all_true, all_pred, target_names=filtered_class_names)
            with open("report.txt", "a") as file:
                file.write(report)
                file.write("\n")
                file.write("\n")
            file.close
            # print(report)

            # 计算confusion matrix
            conf_mat = confusion_matrix(all_true, all_pred, labels=unique_classes)
            with open("Confusion_Matrix.txt", "a") as file:
                file.write(f"Epoch: {epoch}\n")
                file.write("Confusion Matrix:\n")
                file.write(np.array2string(conf_mat, threshold=sys.maxsize))
                file.write("\n")
                file.write("\n")
            file.close

            # Use the `profile` function from `thop` to calculate FLOPs
            flops, params = profile(model, inputs=(data_ti, h0, c0, 36, batch_size, 25), verbose=False)
            # Format the output to make it more readable
            flops, params = clever_format([flops, params], "%.3f")
            print('test FLOPs: {}'.format(flops))
            print('test Params: {}'.format(params))

            eval_loss = np.mean(eval_loss)
            eval_classify_accu = np.mean(eval_classify_accu)

            print('eval_loss: {}'.format(eval_loss))
            print('eval_classify_accu: {}'.format(eval_classify_accu))
            
            # 检查验证性能是否提升
            if eval_loss < best_validation_loss:
                best_validation_loss = eval_loss
                counter = 0
            else:
                counter += 1

            # # 如果连续多于early_stop_patience个epoch验证性能没有提升，停止训练
            if counter >= early_stop_patience:
                print(f'Early stopping after {epoch+1} epochs without improvement.')
                break

        writer1.add_scalar(tag='train_loss', scalar_value=training_loss, global_step=epoch)
        writer1.add_scalar(tag='train_classify_accu', scalar_value=train_classify_accu_epoch, global_step=epoch)

        writer1.add_scalar(tag='eval_loss', scalar_value=eval_loss, global_step=epoch)
        writer1.add_scalar(tag='eval_classify_accu', scalar_value=eval_classify_accu, global_step=epoch)        

        if (epoch + 1) % 25 == 0:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, path + "/model_{}.pth".format(epoch))

    end_time = time.time()

    print("训练总时长：{}".format(end_time - start_time))
    writer1.close()
    
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, path + "/model_{}.pth".format(epoch))
    return model,best_validation_loss

if __name__ == '__main__':
    open("Confusion_Matrix.txt", "w")
    open("report.txt", "w")
    train_model(num_epochs = 700, learning_rate = 0.0002, batchsize = 16)