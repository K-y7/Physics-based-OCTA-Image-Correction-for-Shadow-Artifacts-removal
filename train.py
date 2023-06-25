import os
import random
import shutil
import yaml
from attrdict import AttrMap
import time

import torch
from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F

from data_manager import TrainDataset,ValDataset
from SpA_Former import Generator
from models.dis.dis import Discriminator
import utils
from utils import gpu_manage, save_image, checkpoint
from eval import test
from log_report import LogReport
from log_report import TestReport


def train(config):
    gpu_manage(config)

    ### DATASET LOAD ###
    print('===> Loading datasets')
    
    train_dataset = TrainDataset(config)
    validation_dataset = ValDataset(config)
    print('train dataset:', len(train_dataset))
    print('validation dataset:', len(validation_dataset))

    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config.threads, batch_size=config.batchsize, shuffle=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads, batch_size=config.validation_batchsize, shuffle=False)
    
    
    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)#图片生成器

    if config.gen_init is not None:
        param = torch.load(config.gen_init)
        gen.load_state_dict(param)
        print('load {} as pretrained model'.format(config.gen_init))

    dis = Discriminator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)#分类器

    if config.dis_init is not None:
        param = torch.load(config.dis_init)
        dis.load_state_dict(param)
        print('load {} as pretrained model'.format(config.dis_init))

    # setup optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)
    opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)
    # 这两行代码是为了设置Adam优化器的参数。其中，Adam优化器是一种高效的梯度下降算法，可用于深度学习模型的训练中。
    # 这里的gen和dis分别是生成器和判别器的类对象，即模型。使用.gen.parameters()和dis.parameters()可以获取它们的参数，然后用这些参数来初始化对应的优化器。
    # lr表示学习率，betas是Adam优化器中的两个参数，用来调整梯度下降的速度，weight_decay是权重衰减项，用于防止过拟合。
    real_a = torch.FloatTensor(config.batchsize, config.in_ch, config.width, config.height)
    real_b = torch.FloatTensor(config.batchsize, config.out_ch, config.width, config.height)
    M = torch.FloatTensor(config.batchsize, config.width, config.height)
    # 这段代码中，定义了三个浮点类型的Tensor张量，分别为real_a、real_b和M。其中，real_a代表包含输入图像的Tensor张量，real_b代表包含输出图像的Tensor张量，M代表包含二值图像的Tensor张量。
    # 这三个张量的形状和大小分别由batchsize、in_ch、out_ch、width和height的值来确定。在这个过程中，torch.FloatTensor函数被用于创建基于浮点数的Tensor张量

    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()
    criterionSoftplus = nn.Softplus()
    # 这些代码定义了三种损失函数，分别是：L1损失函数、均方误差损失函数和Softplus损失函数。在深度学习中，损失函数用于衡量预测结果与真实结果的差异，通常是优化模型的目标之一。
    # L1损失函数计算绝对误差，均方误差损失函数计算平方误差，而Softplus损失函数则在误差较小时施加一些惩罚，可以将其视为一种平滑的损失函数。

    if config.cuda:
        gen = gen.cuda()
        dis = dis.cuda()
        criterionL1 = criterionL1.cuda()
        criterionMSE = criterionMSE.cuda()
        criterionSoftplus = criterionSoftplus.cuda()
        real_a = real_a.cuda()
        real_b = real_b.cuda()
        M = M.cuda()

    real_a = Variable(real_a)
    real_b = Variable(real_b)
    # 如果config.cuda为True，则将gen、dis、criterionL1、criterionMSE和criterionSoftplus存储至cuda设备上。
    # 同时，将real_a、real_b和M也存储至cuda设备上。最后将real_a和real_b用Variable进行包装。

    logreport = LogReport(log_dir=config.out_dir)
    validationreport = TestReport(log_dir=config.out_dir)
    # 用于输出训练和验证结果的日志报告。
    print('===> begin')
    start_time=time.time()
    # main
    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()
        for iteration, batch in enumerate(training_data_loader, 1):

            real_a_cpu, real_b_cpu, M_cpu = batch[0], batch[1], batch[2]
            # 在该循环中，首先获取训练数据集中的一批数据，分别赋值给变量real_a_cpu、real_b_cpu和M_cpu。
            real_a.resize_(real_a_cpu.size()).copy_(real_a_cpu)
            real_b.resize_(real_b_cpu.size()).copy_(real_b_cpu)
            M.resize_(M_cpu.size()).copy_(M_cpu)
            # 将这些数据分别复制到变量real_a、real_b和M中，

            att, fake_b = gen.forward(real_a)
            # 调用生成器（gen对象）的forward方法来生成输出att和fake_b。

            ################
            ### Update D ###
            ################

            opt_dis.zero_grad()
            # opt_dis.zero_grad() 是将判别器的梯度清零。这个操作的作用是可以避免在优化过程中累积梯度导致梯度爆炸或梯度消失的问题。
            # 这个操作一般在每次优化判别器时都需要执行。

            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab.detach())
            batchsize, _, w, h = pred_fake.size()
            # 在这段代码中，`fake_ab` 是将真实图像 `real_a` 和生成的图像 `fake_b` 进行拼接、组成的新图像。
            # `pred_fake` 是使用 `dis`（鉴别器）对这个新图像进行判断，得到的预测值。
            # `batchsize`、`_`（这个下划线是因为这个位置的值我们并不需要，所以用下划线代替）、`w`、`h` 则分别表示预测值的批量大小、通道数、宽度和高度。

            loss_d_fake = torch.sum(criterionSoftplus(pred_fake)) / batchsize / w / h
            # 是计算生成器的损失函数。其中，pred_fake 表示生成器生成的图像；criterionSoftplus 是 softplus 激活函数作为损失函数；
            # batchsize、w、h 分别表示训练中的批次大小、图像宽度和高度。该公式通过计算生成器生成图像与真实图像之间的差异来更新生成器的权重，使其能够更好地生成真实的图像。


            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = dis.forward(real_ab)
            loss_d_real = torch.sum(criterionSoftplus(-pred_real)) / batchsize / w / h
            # 将real_a和real_b在通道维度上拼接起来，得到一个形状为（batchsize，2，w，h）的张量real_ab。
            # 然后将其输入到discriminator网络中得到输出pred_real，然后用criterionSoftplus计算出-discriminator(real_ab)的平均值作为loss_d_real。
            # Combined loss
            loss_d = loss_d_fake + loss_d_real

            loss_d.backward()
            # loss_d是由loss_d_fake和loss_d_real相加得到的，loss_d_fake和loss_d_real分别是鉴别器对生成器生成的假图像和数据集中真实图像的判别损失，
            # loss_d通过反向传播算法将误差传递并更新相关参数。
            if epoch % config.minimax == 0:
                opt_dis.step()
            # 如果当前的epoch能够被config.minimax整除，那么就执行opt_dis.step()，也就是对判别器的优化器进行一次参数更新。
            ################
            ### Update G ###
            ################

            opt_gen.zero_grad()
            # opt_gen.zero_grad() 的意思是将生成器的梯度设为零。这是为了避免梯度累加，以便进行下一轮迭代。
            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab)
            loss_g_gan = torch.sum(criterionSoftplus(-pred_fake)) / batchsize / w / h
            # 将real_a和fake_b在channel上合并成一个tensor fake_ab，然后用判别器dis来预测fake_ab的真假标签pred_fake，接着计算生成器的对抗损失loss_g_gan，
            # 其中-criterionSoftplus(-pred_fake)是为了把生成器的目标从最小化loss_g_gan变成最大化pred_fake，
            # 损失的值除以batchsize、w和h是为了使得不同分辨率的图像可以进行比较。最后需要反向传播求解生成器的梯度。
            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b) * config.lamb
            loss_g_att = criterionMSE(att[:,0,:,:], M)
            loss_g = loss_g_gan + loss_g_l1 + loss_g_att
            # 首先定义三个loss：loss_g_l1为生成图像与实际图像的L1损失（也称为平均绝对误差），loss_g_att为属性向量与目标属性之间的均方误差，
            # loss_g_gan为生成器在鉴别器方面的损失。然后将三个loss加权求和，得到总损失loss_g，
            # 并将其反向传播到生成器参数上以更新参数，使生成器产生更逼真、与目标风格更匹配的图像。

            loss_g.backward()
            # loss_g.backward() 的作用是计算生成器网络的损失函数对各参数的梯度并反向传播，以便更新参数。也就是说，这一行代码是用来进行生成器网络的训练和参数优化的关键步骤。
            opt_gen.step()

            # log # 是用来更新生成器的权重参数，使其尽可能地让生成图像越来越接近真实图像。
            if iteration % 10 == 0:
                print("===> Epoch[{}]({}/{}): loss_d_fake: {:.4f} loss_d_real: {:.4f} loss_g_gan: {:.4f} loss_g_l1: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss_d_fake.item(), loss_d_real.item(), loss_g_gan.item(), loss_g_l1.item()))
                # 每经过 10 次迭代就输出一次损失函数的值，包括 loss_d_fake，loss_d_real，loss_g_gan 和 loss_g_l1。
                # 其中，Epoch 为当前迭代的轮数，iteration 为当前迭代的次数，len(training_data_loader) 为总的迭代次数，
                # loss_d_fake 为判别器对假图像的判别损失，loss_d_real 为判别器对真实图像的判别损失，loss_g_gan 为生成器的对抗损失，loss_g_l1 为生成器的 L1 损失。
                log = {}
                log['epoch'] = epoch
                log['iteration'] = len(training_data_loader) * (epoch-1) + iteration
                log['gen/loss'] = loss_g.item()
                log['dis/loss'] = loss_d.item()

                logreport(log)
                # 该段代码是记录训练过程中的损失值信息。将当前训练 epoch 数、iteration 数、生成器损失值和鉴别器损失值记录在字典 log 中，
                # 并通过 logreport() 函数汇报信息。这有助于跟踪训练损失值的变化趋势，以便进行调整和优化模型。
        print('epoch', epoch, 'finished, use time', time.time() - epoch_start_time)
        with torch.no_grad():
            log_validation = test(config, validation_data_loader, gen, criterionMSE, epoch)
            validationreport(log_validation)
        print('validation finished')
        if epoch % config.snapshot_interval == 0:
            checkpoint(config, epoch, gen, dis)
        # 这段代码输出了当前 epoch 的信息并计算了验证集的指标。
        # 使用 torch.no_grad() 是为了避免在验证集上进行反向传播。如果当前 epoch 是快照周期，则会保存当前模型的 checkpoint。
        logreport.save_lossgraph()
        validationreport.save_lossgraph()
    print('training time:', time.time() - start_time)
        # 保存训练过程中的损失值图表和验证过程中的损失值图表，并打印训练时间。

if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    utils.make_manager()
    n_job = utils.job_increment()
    config.out_dir = os.path.join(config.out_dir, '{:06}'.format(n_job))
    os.makedirs(config.out_dir)
    print('Job number: {:04d}'.format(n_job))

    # 保存本次训练时的配置
    shutil.copyfile('config.yml', os.path.join(config.out_dir, 'config.yml'))

    train(config)
