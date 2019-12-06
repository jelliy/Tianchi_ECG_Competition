# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 18:45

@ author: javis
'''
import os


class Config:
    # for data_process.py
    train_root = r'../data'
    #test_root = r'../data'
    test_root = r'/tcdata'
    train_dir = os.path.join(train_root, 'hf_round2_train')
    train_label = os.path.join(train_root, 'hf_round2_label.txt')
    test_dir = os.path.join(test_root, 'hf_round2_testA')
    test_label = os.path.join(test_root, 'hf_round2_subA.txt')
    arrythmia = os.path.join(train_root, 'hf_round2_arrythmia.txt')
    cur_dir = ""
    # for train
    # 训练的模型名称
    model_name = 'resnet34'
    # 在第几个epoch进行到下一个state,调整lr
    stage_epoch = [25, 30, 40]
    # 训练时的batch大小
    batch_size = 64
    # label的类别数
    num_classes = 34
    # 最大训练多少个epoch
    max_epoch = 256
    # 目标的采样长度
    target_point_num = 2048
    # 保存模型的文件夹
    ckpt = 'ckpt'
    # 保存提交文件的文件夹
    sub_dir = 'submit'
    # 初始的学习率
    lr = 1e-3
    # 保存模型当前epoch的权重
    current_w = 'current_w.pth'
    # 保存最佳的权重
    best_w = 'best_w.pth'
    # 学习率衰减 lr/=lr_decay
    lr_decay = 10


config = Config()
