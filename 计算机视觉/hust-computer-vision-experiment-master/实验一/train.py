"""
@Description :   模型训练
@Author      :   tqychy 
@Time        :   2023/12/05 19:32:15
"""

"""
TODO:
标准差归一化
正则化
"""

# 训练
import datetime
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from utils import *
from net import FCNet
from dataset import generate_dataset

def get_loss(criterion, params):
    """
    计算损失函数
    :param l1_weight: L1 损失权重
    :param l2_weight: L2 损失权重
    """

    
def training(verbose=True, save_param=True, save_fig=True):
    """
    模型训练函数
    :param verbose: 是否打印输出
    :param save_param: 是否保存模型参数
    :param save_fig: 是否画图
    :returns tuple[float, float]: 准确率和 macro f1 分数
    """
    set_seed(params["random_seed"])
    data = pd.read_csv("dataset.csv", header=0)
    train_dataset, valid_dataset = generate_dataset(
        data[["data1", "data2"]].values, data["label"].values)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=params["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(
        dataset=valid_dataset, batch_size=len(valid_dataset))

    net = FCNet(2, params["hidden_layers_list"], 4,
                params["activate_func"], params["dropout"]).to(device)
    net.init_weights()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=params["lr"])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    save_path = os.path.join(
        "params", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    training_losses_per_epoch = []
    training_losses_per_batch = []
    training_f1_per_epoch = []
    training_f1_per_batch = []
    training_acc_per_epoch = []
    training_acc_per_batch = []
    valid_losses = []
    valid_f1s = []
    valid_accs = []
    max_acc = 0.
    max_f1 = 0.

    for i in range(params["epoches"]):

        mean_loss = 0.
        mean_f1 = 0.
        mean_acc = 0.
        if verbose:
            print(f"Epoch {1 + i} training:")
        net.train()
        for feature, label in tqdm(train_dataloader, disable=not verbose):
            inputs, labels = feature.float().to(device), label.to(device)
            outputs = net(inputs)

            _, prediction = torch.max(outputs.data, 1)
            y_true, y_pred = labels.cpu().numpy(), prediction.cpu().numpy()
            training_f1, training_acc = f1_score(
                y_true, y_pred, average="macro"), accuracy_score(y_true, y_pred)
            mean_f1 += training_f1
            mean_acc += training_acc
            training_f1_per_batch.append(training_f1)
            training_acc_per_batch.append(training_acc)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            l1_loss, l2_loss = 0, 0
            for param in net.parameters():
                l1_loss += torch.norm(param, 1)
                l2_loss += torch.norm(param, 2)
            loss = loss + params["l1_weight"] * l1_loss + params["l2_weight"] * l2_loss
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()
            training_losses_per_batch.append(loss.item())

        training_losses_per_epoch.append(mean_loss / len(train_dataloader))
        training_f1_per_epoch.append(mean_f1 / len(train_dataloader))
        training_acc_per_epoch.append(mean_acc / len(train_dataloader))
        if verbose:
            print(f"Epoch {1 + i} validation:")
        net.eval()
        checkpoint = {
            "net": net.state_dict(),
            "optim": optimizer.state_dict()
        }
        save(checkpoint, save_path, f"Epoch{1 + i}.pth", save_param)
        with torch.no_grad():
            for feature, label in valid_dataloader:
                inputs, labels = feature.float().to(device), label.to(device)
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                l1_loss, l2_loss = 0, 0
                for param in net.parameters():
                    l1_loss += torch.norm(param, 1)
                    l2_loss += torch.norm(param, 2)
                # print(f"loss:{loss}, l1_loss:{l1_loss}, l2_loss:{l2_loss}")
                loss = loss + params["l1_weight"] * l1_loss + params["l2_weight"] * l2_loss
                
                valid_losses.append(loss.item())

                _, prediction = torch.max(outputs.data, 1)
                y_true, y_pred = labels.cpu().numpy(), prediction.cpu().numpy()
                valid_f1, valid_acc = f1_score(
                    y_true, y_pred, average="macro"), accuracy_score(y_true, y_pred)
                if verbose:
                    print(f"f1:{valid_f1}")
                    print(f"acc:{valid_acc}")
                valid_f1s.append(valid_f1)
                valid_accs.append(valid_acc)

                if max_acc < valid_acc:
                    checkpoint = {
                        "net": net.state_dict(),
                        "optim": optimizer.state_dict()
                    }
                    save(checkpoint, save_path, "best.pth", save_param)
                    max_acc = valid_acc
                    max_f1 = max(valid_f1, max_f1)

        if params["early_stop"] and early_stop(valid_accs, params["patience"]):
            break

    if save_fig:
        # 绘图函数
        # 训练损失函数及评价指标(batch)
        fig = plt.figure()
        # plt.title("训练损失函数及评价指标", fontdict=fontdict)
        ax1 = fig.add_subplot(111, label="1")
        ax2 = fig.add_subplot(111, label="2", frame_on=False)

        ax1.plot(range(len(training_losses_per_batch)),
                 training_losses_per_batch, "g", label="loss")
        ax1.set_xlabel("batch number", fontdict)
        ax1.set_ylabel("loss value", fontdict)
        ax1.legend()

        ax2.plot(range(len(training_acc_per_batch)),
                 training_acc_per_batch, "purple", label="acc")
        ax2.plot(range(len(training_f1_per_batch)),
                 training_f1_per_batch, "b", label="macro f1")
        ax2.set_xticks([])
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.set_ylabel("metrics", fontdict=fontdict)
        ax2.legend()

        plt.savefig("./images/训练损失函数及评价指标(batch).pdf")
        plt.clf()

        # 训练验证损失函数
        fig = plt.figure()
        # plt.title("训练损失函数及评价指标", fontdict=fontdict)
        min_training_loss,  min_training_loss_id= min(training_losses_per_epoch), training_losses_per_epoch.index(min(training_losses_per_epoch))
        min_valid_loss,  min_valid_loss_id= min(valid_losses), valid_losses.index(min(valid_losses))
        plt.plot(range(len(training_losses_per_epoch)),
                 training_losses_per_epoch, "g", label="train", marker=".")
        plt.plot(range(len(valid_losses)), valid_losses,
                 "purple", label="valid", marker="*")
        plt.plot(range(len(training_losses_per_epoch)), [min_training_loss for _ in range(len(training_losses_per_epoch))], "r", linestyle="--")
        plt.plot(range(len(valid_losses)), [min_valid_loss for _ in range(len(valid_losses))], "r", linestyle="--")
        plt.scatter(min_training_loss_id, min_training_loss, c="orange", marker="s")
        plt.scatter(min_valid_loss_id, min_valid_loss, c="r", marker="s")
        plt.text(0, min_training_loss + 0.2, f"min training loss is {round(min_training_loss, 3)}")
        plt.text(0, min_valid_loss - 0.3, f"min valid loss is {round(min_valid_loss, 3)}")
        plt.xlabel("batch number")
        plt.ylabel("loss values")
        plt.legend()

        plt.savefig("./images/训练验证损失函数.pdf")
        plt.clf()

        # 训练验证评价指标
        fig = plt.figure()
        # plt.title("训练验证评价指标", fontdict=fontdict)
        min_training_acc,  min_training_acc_id= max(training_acc_per_epoch), training_acc_per_epoch.index(max(training_acc_per_epoch))
        min_valid_acc,  min_valid_acc_id= max(valid_accs), valid_accs.index(max(valid_accs))
        plt.plot(range(len(training_acc_per_epoch)),
                 training_acc_per_epoch, "g", label="train acc", marker=".")
        plt.plot(range(len(valid_accs)), valid_accs, "purple", label="valid acc", marker="*")
        plt.plot(range(len(training_acc_per_epoch)), [min_training_acc for _ in range(len(training_acc_per_epoch))], "orange", linestyle="--")
        plt.plot(range(len(valid_accs)), [min_valid_acc for _ in range(len(valid_accs))], "r", linestyle="--")
        plt.scatter(min_training_acc_id, min_training_acc, c="orange", marker="s")
        plt.scatter(min_valid_acc_id, min_valid_acc, c="r", marker="s")
        plt.text(0, min_training_acc + 0.005, f"max training acc is {round(min_training_acc, 3)}")
        plt.text(0, min_valid_acc + 0.005, f"max valid acc is {round(min_valid_acc, 3)}")
        plt.xlabel("batch number")
        plt.ylabel("metrics")
        plt.legend()
        plt.savefig("./images/训练验证acc.pdf")
        plt.clf()

        min_training_f1,  min_training_f1_id= max(training_f1_per_epoch), training_f1_per_epoch.index(max(training_f1_per_epoch))
        min_valid_f1,  min_valid_f1_id= max(valid_f1s), valid_f1s.index(max(valid_f1s))
        plt.plot(range(len(training_f1_per_epoch)),
                 training_f1_per_epoch, "g", label="train macro f1", marker=".")
        plt.plot(range(len(valid_f1s)), valid_f1s, "purple", label="valid macro f1", marker="*")
        plt.plot(range(len(training_f1_per_epoch)), [min_training_f1 for _ in range(len(training_f1_per_epoch))], "orange", linestyle="--")
        plt.plot(range(len(valid_f1s)), [min_valid_f1 for _ in range(len(valid_f1s))], "r", linestyle="--")
        plt.scatter(min_training_f1_id, min_training_f1, c="orange", marker="s")
        plt.scatter(min_valid_f1_id, min_valid_f1, c="r", marker="s")
        plt.text(0, min_training_f1 + 0.005, f"max training f1 is {round(min_training_f1, 3)}")
        plt.text(0, min_valid_f1 + 0.005, f"max valid f1 is {round(min_valid_f1, 3)}")
        plt.xlabel("batch number")
        plt.ylabel("metrics")
        plt.legend()
        plt.savefig("./images/训练验证f1.pdf")
        plt.clf()

    if verbose:
        print(f"max acc:{max_acc} max f1:{max_f1}")
    return max_acc, max_f1


if __name__ == "__main__":
    device = params["device"]
    run_grid_search, test_layer_nums, test_neural_nums, test_lr, test_l1_loss, test_act_func = False, False, False, False, True, False

    training(False, False, True)

    if run_grid_search:
        activate_funcs = [nn.LeakyReLU, nn.ReLU, nn.Sigmoid, nn.GELU]
        dropouts = [0, 0.4, 0.6]
        l2_losses = [0, 1e-3, 1e-2, 1e-1]
        l1_losses = [0, 1e-4, 1e-3, 1e-2]
        batch_sizes = [3600, 360, 36]
        hidden_layer_lists = [[500], [100], [50], [10]]
        lrs = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        best_param = {}
        max_acc, max_f1 = 0., 0.
        for activate_func in activate_funcs:
            params["activate_func"] = activate_func
            for dropout in dropouts:
                params["dropout"] = dropout
                for l2_loss in l2_losses:
                    params["l2_weight"] = l2_loss
                    for l1_loss in l1_losses:
                        params["l1_weight"] = l1_loss
                        for batch_size in batch_sizes:
                            params["batch_size"] = batch_size
                            for layer_list in hidden_layer_lists:
                                params["hidden_layers_list"] = layer_list
                                for lr in lrs:
                                    params["lr"] = lr
                                    print(f"Using params\n{params}")
                                    acc, f1 = training(False, False, False)
                                    print(f"acc:{acc}, f1:{f1}")
                                    if max_acc < acc:
                                        max_acc = acc
                                        best_param = params
                                        max_f1 = f1

        print(f"max acc:{max_acc}, max f1:{max_f1}")
        print(f"Best params are \n{best_param}")

    if test_layer_nums:
        hidden_layer_lists = [[50 for _ in range(i)] for i in range(1, 11)]
        accs = []
        for i, hidden_layer_list in enumerate(hidden_layer_lists):
            params["hidden_layers_list"] = hidden_layer_list
            params["lr"] = 1 / (1 + i)
            print(f"Case {i}: {hidden_layer_list}")
            acc, f1 = training(False, False, False)
            print(f"acc: {acc} f1: {f1}")
            accs.append(acc)

        plt.plot(range(1, 1 + len(hidden_layer_lists)), accs, "g", marker="*")
        for i, acc in enumerate(accs):
            plt.text(1 + i, acc + 0.005, f"{round(acc, 3)}")
        plt.xlabel("layer nums")
        plt.ylabel("valid acc")
        plt.savefig("./images/不同的网络层数.pdf")
        plt.clf()

    if test_neural_nums:
        hidden_layer_lists = [[i] for i in np.arange(10, 110, 10)]
        accs = []
        for i, hidden_layer_list in enumerate(hidden_layer_lists):
            params["hidden_layers_list"] = hidden_layer_list
            # params["lr"] = 1 / (1 + i)
            print(f"Case {i}: {hidden_layer_list}")
            acc, f1 = training(False, False, False)
            print(f"acc: {acc} f1: {f1}")
            accs.append(acc)

        plt.plot(range(1, 1 + len(hidden_layer_lists)), accs, "purple", marker="*")
        for i, acc in enumerate(accs):
            plt.text(1 + i, acc, f"{round(acc, 3)}")
        plt.xlabel("neural nums")
        plt.xticks(range(len(accs)), np.arange(0, 100, 10))
        plt.ylabel("valid acc")
        plt.savefig("./images/不同的神经元个数.pdf")
        plt.clf()

    if test_lr:
        lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 50]
        epoches = [2000, 1500, 1000, 500, 200, 200, 200, 100, 100]
        accs = []
        for i, lr in enumerate(lrs):
            params["lr"] = lr
            params["epoches"] = epoches[i]
            print(f"Case {i}: {lr}")
            acc, f1 = training(False, False, False)
            print(f"acc: {acc} f1: {f1}")
            accs.append(acc)

        plt.plot(range(len(accs)), accs, "#784315", marker="*")
        for i, acc in enumerate(accs):
            plt.text(i, acc, f"{round(acc, 3)}")
        plt.xlabel("lr")
        plt.xticks(range(len(accs)), lrs)
        plt.ylabel("valid acc")
        plt.savefig("./images/不同的学习率.pdf")
        plt.clf()


    if test_l1_loss:
        l1_losses = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        accs = []
        for i, l1 in enumerate(l1_losses):
            params["l1_weight"] = l1
            print(f"Case {i}: {l1}")
            acc, f1 = training(False, False, False)
            print(f"acc: {acc} f1: {f1}")
            accs.append(acc)

        plt.plot(range(len(accs)), accs, "b", marker="*")
        for i, acc in enumerate(accs):
            plt.text(i, acc, f"{round(acc, 3)}")
        plt.xlabel("L1 weight")
        plt.xticks(range(len(accs)), l1_losses)
        plt.ylabel("valid acc")
        plt.savefig("./images/不同的l1_weight.pdf")
        plt.clf()

    if test_act_func:
        activate_funcs = [nn.LeakyReLU, nn.ReLU, nn.Sigmoid, nn.GELU]
        for i, activate_func in enumerate(activate_funcs):
            params["activate_func"] = activate_func
            print(f"Case {i}: {activate_func}")
            acc, f1 = training(False, False, False)
            print(f"acc: {acc} f1: {f1}")


