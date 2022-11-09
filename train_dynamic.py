import os
import torch
import torch.nn as nn
import argparse
import time
import pickle
import tensorboard_logger as tb_logger
from datasets.datasets_loader import get_loaders_new
from helper.loops import train_one_epoch_dynamic, validate
from models.backbone.Signal import model_dict
from datasets import datasets_dict
from helper.create import create_optimizer, create_scheduler


def parse_args():
    parser = argparse.ArgumentParser('the argument for training')

    # regular parameters
    parser.add_argument("--print_freq", type=int, default=10, help="the frequency to print")
    parser.add_argument("--save_freq", type=int, default=100, help="the frequency to save")
    parser.add_argument('--batch_size', type=int, default=32, help="the batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="the num of workers to load data")
    parser.add_argument("--epochs", type=int, default=500, help="the total train epoch")

    # optimizer parameters
    parser.add_argument("--optimizer_name", type=str, default="adam", choices=["adam", "sgd", "adamw", "rmsprop"],
                        help='Optimizer name')
    parser.add_argument("--opt_eps", type=float, default=1e-8, help="Optimizer Epsilon")
    parser.add_argument("--opt_betas", type=str, default=None, help="Optimizer Betas, use opt default")
    parser.add_argument("--momentum", type=float, default=0.9, help="Optimizer momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="the norm to weight")
    parser.add_argument("--lr", type=float, default=1e-3, help="the init learning rate")
    parser.add_argument("--amp", type=int, default=0, choices=[0, 1], help="using amp to train or not")

    # schedule parameters
    parser.add_argument("--lr_scheduler", type=str, default="step", choices=["step", "mstep", "exp", "cos", "reduce"],
                        help="the learning rate scheduler")
    parser.add_argument("--lr_decay_epochs", type=str, default="150,250,350", help="the epoch to adjust the lr")
    parser.add_argument("--lr_decay_rate", type=float, default=0.9, help="decay rate for learning rate")
    parser.add_argument("--patience", type=int, default=20, help="the metric to adjust ReduceLROnPlateau")

    # loss parameters
    parser.add_argument("--loss_name", type=str, default="cross_entropy",
                        choices=["cross_entropy", "smooth_cross_entropy", "jsd_loss", "enhanced_loss"])

    # dataset parameters
    parser.add_argument("--work_dir", type=str, default=r'D:\深度学习\测试数据\湖大螺旋锥齿轮新箱体test\测试数据集',
                        help="the path root of data")
    parser.add_argument("--datasets", type=str, default="hnu_datasets", choices=["hnu_dataset",
                                                                                 "xjtu_dataset",
                                                                                 "dds_dataset",
                                                                                 ],
                        help="the dataset for training")
    parser.add_argument("--size", type=int, default=100, help="Number of all samples")
    parser.add_argument('--train_size_use', type=str, default="300,20",
                        help="the dataset size of each type during training preprocess")
    parser.add_argument('--test_size', type=int, default=100,
                        help="the dataset size of each type during testing preprocess")
    parser.add_argument("--step", type=int, default=500, help="the overlap of two samples")
    parser.add_argument("--length", type=int, default=1024, help="the length of each sample")
    parser.add_argument("--use_ratio", type=int, default=0, choices=[0, 1],
                        help=" Whether to specify the proportion of training samples")
    parser.add_argument("--ratio", type=float, default=0.5,
                        help=" Ratio of training samples, should be (0,1) and only works when opt.use_ratio is True")
    parser.add_argument("-t", "--trail", type=int, default=0, help="the experiment id")

    # model parameters
    parser.add_argument("--model", type=str, default="convformer_s",
                        choices=["convformer_s", "convformer_m", "convformer_l",
                                 "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                                 "vgg11", "vgg13", "vgg16", "vgg19",
                                 "ehcnn_24_16", "ehcnn_30_32", "ehcnn_24_16_dilation",
                                 "vit_base", "vit_middle_16", 'vit_middle_32',
                                 'max_vit_tiny_16', 'max_vit_tiny_32', 'max_vit_small_16', 'max_vit_small_32',
                                 'localvit_base_patch16_type1', 'localvit_base_patch16_type2',
                                 ' localvit_middle1_patch16_type1', 'localvit_middle12_patch16_type1'],
                        help="the name of model")

    parser.add_argument("--num_cls", type=int, default=8, help="the classification classes")
    parser.add_argument("-ic", "--input_channel", type=int, default=3, help="the input channel of input data")
    parser.add_argument("--layer_args", type=str, default='100,64,32', help="the hidden layer neurons")
    opt = parser.parse_args()

    # add lr_decay_epochs for lr scheduler
    decay_iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in decay_iterations:
        opt.lr_decay_epochs.append(int(it))

    # add h_args for classifier
    if not opt.layer_args:
        opt.h_args = None
    else:
        h_layer_args = opt.layer_args.split(",")
        opt.h_args = list([])
        for it in h_layer_args:
            opt.h_args.append(int(it))

    # add opt betas for optimizer(like adam)
    if opt.opt_betas:
        opt.betas = []
        for it in opt.opt_betas.split(","):
            opt.betas.append(float(it))
    if not opt.opt_betas:
        opt.betas = None

    # add train_size for imbalanced sample training
    list_ = opt.train_size_use.split(",")
    if len(list_) == 1:
        train_size = int(list_[0])
        opt.train_size = [train_size] * opt.num_cls
    else:
        if len(list_) == 2:
            opt.train_size = [int(list_[0])] + [int(list_[1])] * (opt.num_cls - 1)
        elif len(list_) == opt.num_cls:
            opt.train_size = []
            for it in list_:
                opt.train_size.append(int(it))
        else:
            raise ValueError('the train size should be 2 or num classes')
    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}_train_ratio{}_amp_use_{}'.format(opt.model, opt.datasets, opt.lr,
                                                                                     opt.weight_decay,
                                                                                     opt.trail, opt.ratio,
                                                                                     bool(opt.amp))

    opt.save_path = './save'
    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    train_result = {'accuracy': [],
                    'loss': [],
                    'lr': [],
                    'sample_weight': []}
    test_result = {'accuracy': [],
                   'loss': []}
    best_acc = 0
    best_epoch = 0
    opt = parse_args()
    model = model_dict[opt.model](h_args=opt.h_args, in_c=opt.input_channel, num_cls=opt.num_cls)
    datasets_using = datasets_dict[opt.datasets]

    print("==>Loading data...")
    train_loader, test_loader = get_loaders_new(opt, MyDatasets=datasets_using)

    # create optimizer, lr_scheduler, loss_function, scaler
    optimizer = create_optimizer(model, opt)
    lr_scheduler = create_scheduler(optimizer, opt)

    sample_weight = torch.tensor(opt.train_size, dtype=torch.float)
    sample_weight = (sample_weight.max() / sample_weight).cuda()
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch in range(1, opt.epochs + 1):
        print("==>the training is going ")
        time_1 = time.time()
        train_acc, train_loss, cur_lr, sample_weight = train_one_epoch_dynamic(epoch, train_loader, model, criterion,
                                                                               optimizer, opt, sample_weight)
        criterion.weight = sample_weight.cuda()
        lr_scheduler.step()
        time_2 = time.time()
        train_result["accuracy"].append(train_acc)
        train_result["loss"].append(train_loss)
        train_result["lr"].append(cur_lr)
        train_result["sample_weight"].extend(sample_weight.cpu().detach().numpy())
        print("the {} epoch, total train time{:.2f}".format(epoch, time_2 - time_1))

        test_acc, test_loss = validate(test_loader, model, criterion, opt)
        test_result["accuracy"].append(test_acc)
        test_result['loss'].append(test_loss)
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {'epoch': epoch,
                          'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'best_acc': best_acc}
            best_epoch = epoch
        if epoch % opt.save_freq == 0:
            print("==>Saving the regular model")
            regular_stare = {'epoch': epoch,
                             'model': model.state_dict(),
                             'optimizer': optimizer.state_dict()}
            regular_save_file = os.path.join(opt.save_folder, 'checkpoint_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(regular_stare, regular_save_file)

    final_state = {'opt': opt,
                   'model': model.state_dict(),
                   'optimizer': optimizer.state_dict()}

    # save the last checkpoint file
    print("==> Saving the last model")
    final_save_file = os.path.join(opt.save_folder, 'last_checkpoint_{epoch}.pth'.format(epoch=opt.epochs))
    torch.save(final_state, final_save_file)

    # save the best checkpoint file
    print("==> Saving the best model")
    best_save_file = os.path.join(opt.save_folder, 'best_checkpoint_{epoch}.pth'.format(epoch=best_epoch))
    torch.save(best_state, best_save_file)

    # save the train_result and test_result by using pickle
    print('==> Saving the acc, loss and lr during training')
    train_save_pkl = os.path.join(opt.save_folder, 'train_result.pkl')
    with open(train_save_pkl, "wb") as tf:
        pickle.dump(train_result, tf)

    '''
    load pkl
    with oepn(train_save_pkl, "rb") as tf:
        dict_ = pickle.load(tf)
    '''
    print('==> Saving the test acc and loss')
    test_save_pkl = os.path.join(opt.save_folder, "test_result.pkl")
    with open(test_save_pkl, "wb") as tf:
        pickle.dump(test_save_pkl, tf)


if __name__ == "__main__":
    main()
