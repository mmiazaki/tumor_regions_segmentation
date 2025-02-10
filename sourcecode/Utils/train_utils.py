import datetime
import os
import time
import csv

import torch.optim as optim
from torch.autograd import Variable

from sourcecode.Utils.unet_model import *



def get_loss_function(loss_function, device = "cuda"):
    match loss_function:
        case 'BCELoss':
            return nn.BCELoss().to(device)
        case 'L1Loss':
            return nn.L1Loss().to(device)
        case 'MSELoss':
            return nn.MSELoss().to(device)
        case 'HuberLoss':
            return nn.HuberLoss().to(device)
        case 'SmoothL1Loss':
            return nn.SmoothL1Loss().to(device)
        case _:
            logger.info("Invalid Loss Function")
            return None
    # Loss Functions that didn't work well (odd results):
    # - nn.PoissonNLLLoss().to(device)
    # - nn.HingeEmbeddingLoss().to(device) # target in [-1 1]
    # - nn.SoftMarginLoss().to(device) # target in [-1 1]



def get_optimizer(optimizer, model_parameters):
    match optimizer:
        case 'Adam':
            return optim.Adam(model_parameters)
        case 'Adadelta':
            return optim.Adam(model_parameters)
        case 'Adagrad':
            return optim.Adam(model_parameters)
        case 'AdamW':
            return optim.Adam(model_parameters)
        case 'Adamax':
            return optim.Adam(model_parameters)
        case 'ASGD':
            return optim.Adam(model_parameters)
        case 'NAdam':
            return optim.Adam(model_parameters)
        case 'RAdam':
            return optim.Adam(model_parameters)
        case 'RMSprop':
            return optim.Adam(model_parameters)
        case 'Rprop':
            return optim.Adam(model_parameters)
        case 'SGD':
            return optim.Adam(model_parameters)
        case _:
            logger.info("Invalid Optimizer")
            return None


def train_model_with_validation(dataloaders,
                                model=None,
                                patch_size=(640, 640),
                                n_epochs=1,
                                start_epoch=1,
                                batch_size=1,
                                use_cuda=True,
                                output_dir="../../models",
                                augmentation_strategy="random",
                                augmentation_operations=[None],
                                dataset_name="",
                                loss_function="BCELoss",
                                optimizer_algorithm="Adam",
                                result_file_csv="../../datasets/training_accuracy_loss.csv",
                                model_saving_frequency=('all', 0)):
    # Checking for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_cuda else "cpu"
    logger.info('Running on: {} | GPU available? {}'.format(device, torch.cuda.is_available()))

    torch.cuda.empty_cache()
    if model is None:
        model = UNet(in_channels=3, out_channels=1, padding=True, img_input_size=patch_size).to(device)

    if augmentation_strategy in ["no_augmentation", "color_augmentation", "inpainting_augmentation"]:
        augmentation = augmentation_strategy
    elif augmentation_strategy == "solo":
        aug_tmp = augmentation_operations
        if None in aug_tmp:
            aug_tmp.remove(None)
        augmentation = aug_tmp[0]
    else:
        augmentation = "{}_{}_operations".format(augmentation_strategy, len(augmentation_operations) - 1)

    with open(result_file_csv, mode='a+') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(
            ['model', 'augmentation', 'phase', 'epoch', 'loss', 'accuracy', 'TP', 'TN', 'FP', 'FN', 'date', 'time(sec)',
             'transformations'])

    criterion = get_loss_function(loss_function, device)
    optimizer = get_optimizer(optimizer_algorithm, model.parameters())
    optimizer.zero_grad()

    best_loss = 1.0
    best_acc = 0.0

    since = time.time()
    qtd_images = 0

    for epoch in range(start_epoch, n_epochs + 1):

        time_elapsed = time.time() - since

        logger.info("")
        logger.info("-" * 20)
        logger.info('Epoch {}/{} {} ({:.0f}m {:.0f}s) {}'.format(epoch, n_epochs, augmentation, time_elapsed // 60,
                                                                 time_elapsed % 60,
                                                                 datetime.datetime.now()))
        logger.info("-" * 20)

        # Each epoch has a training and validation phase
        epoch_loss = {}
        epoch_acc = {}
        epoch_tp = {}
        epoch_tn = {}
        epoch_fp = {}
        epoch_fn = {}
        for phase in ['train', 'test']:

            model.train()
            # if phase == 'train':
            #    model.train()  # Set model to training mode
            # else:
            #    model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_accuracy = 0
            running_tp = 0
            running_tn = 0
            running_fp = 0
            running_fn = 0
            n_images = len(dataloaders[phase].dataset)
            for batch_idx, (data, target, fname, original_size) in enumerate(dataloaders[phase]):

                logger.info(
                    "\tfname: '{}' {}/{} :: {} :: Epoch {}/{}".format(fname[0], (batch_idx + 1), n_images, phase, epoch,
                                                                      n_epochs))

                data = Variable(data.to(device))
                target = Variable(target.to(device)).unsqueeze(1)
                # target = Variable(target.to(device))
                # print('X     --> {}'.format(data.size()))
                # print('y     --> {}'.format(target.size()))
                # print('          {}'.format(target))

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):

                    output = model(data)
                    # output = model(data).squeeze(0)
                    # print('y_hat --> {}'.format(output.size()))
                    # print('          {}'.format(output))

                    loss = criterion(output, target)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    torch.cuda.empty_cache()

                    if use_cuda:
                        preds = torch.zeros(output.size(), dtype=torch.double).cuda()
                    else:
                        preds = torch.zeros(output.size(), dtype=torch.double)
                    preds[output >= 0.5] = 1.0

                    # statistics
                    tp = torch.sum(torch.logical_and(target.data == 1, target.data == preds)).detach().cpu().numpy()
                    tn = torch.sum(torch.logical_and(target.data == 0, target.data == preds)).detach().cpu().numpy()
                    fp = torch.sum(torch.logical_and(target.data == 1, target.data != preds)).detach().cpu().numpy()
                    fn = torch.sum(torch.logical_and(target.data == 0, target.data != preds)).detach().cpu().numpy()
                    acc = torch.sum(preds == target.data).detach().cpu().numpy() / (
                                data.size(0) * data.size(-1) * data.size(-2))
                    running_loss += loss.item() * data.size(0)
                    running_accuracy += acc
                    running_tp += tp
                    running_tn += tn
                    running_fp += fp
                    running_fn += fn

                    qtd_images = (batch_idx + 1) * len(data) if phase == 'train' else qtd_images

#                if batch_idx == 0:
#                    break

            epoch_loss[phase] = running_loss / len(dataloaders[phase].dataset)
            epoch_acc[phase] = running_accuracy / len(dataloaders[phase].dataset)
            epoch_tp[phase] = running_tp
            epoch_tn[phase] = running_tn
            epoch_fp[phase] = running_fp
            epoch_fn[phase] = running_fn

        filename = save_model(output_dir, model, dataset_name, patch_size, epoch, qtd_images, batch_size,
                              loss_function, optimizer_algorithm, augmentation, model_saving_frequency)

        if epoch_loss[phase] < best_loss:
            best_loss = epoch_loss[phase]
        if epoch_acc[phase] > best_acc:
            best_acc = epoch_acc[phase]

        logger.info("-" * 20)

        with open(result_file_csv, mode='a+') as csv_file:
            time_elapsed = time.time() - since
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for phase in ['train', 'test']:
                print('[{}] Loss: {:.6f}'.format(phase, epoch_loss[phase]))
                csv_writer.writerow(
                    [filename, augmentation, phase, epoch, epoch_loss[phase], epoch_acc[phase], epoch_tp[phase],
                     epoch_tn[phase], epoch_fp[phase], epoch_fn[phase], datetime.datetime.now(), time_elapsed,
                     str(augmentation_operations).replace(",", "")])

    time_elapsed = time.time() - since
    logger.info('-' * 20)
    logger.info('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best accuracy: {}'.format(best_acc))



def save_model(model_dir, model, dataset_name, patch_size, epoch, qtd_images, batch_size, loss_function,
                       optimizer_algorithm, augmentation, model_saving_frequency):
    filename = '{}_{}x{}_Epoch-{}_Images-{}_Batch-{}_{}_{}_{}.pth'.format(dataset_name, patch_size[0],
                                                                                patch_size[1],
                                                                                epoch, qtd_images, batch_size,
                                                                                loss_function, optimizer_algorithm,
                                                                                augmentation)
    if model_saving_frequency is not None:
        if model_saving_frequency[0] == 'all':
            save_model_file(filename, model_dir, model, dataset_name, epoch, batch_size, loss_function, optimizer_algorithm)
        elif model_saving_frequency[0] == 'every':
            if epoch % model_saving_frequency[1] == 0:
                save_model_file(filename, model_dir, model, dataset_name, epoch, batch_size, loss_function, optimizer_algorithm)
        elif model_saving_frequency[0] == 'last':
            save_model_file(filename, model_dir, model, dataset_name, epoch, batch_size, loss_function, optimizer_algorithm)
            if epoch - model_saving_frequency[1] >= 1:
                delete_model(model_dir, dataset_name, patch_size, epoch - model_saving_frequency[1], qtd_images, batch_size, loss_function, optimizer_algorithm, augmentation)
    return filename



def delete_model(model_dir, dataset_name, patch_size, epoch, imgs, batch_size, loss_function, optimizer_algorithm, augmentation):
    """
    Delete the trained model
    """
    filename = '{}_{}x{}_Epoch-{}_Images-{}_Batch-{}_{}_{}_{}.pth'.format(dataset_name, patch_size[0],
                                                                                patch_size[1],
                                                                                epoch, imgs, batch_size,
                                                                                loss_function, optimizer_algorithm,
                                                                                augmentation)
    filepath = os.path.join(model_dir, filename) if model_dir is not None else filename
    if os.path.exists(filepath):
        os.remove(filepath)



def save_model_file(filename, model_dir, model, dataset_name, epoch, batch_size, loss_function, optimizer_algorithm):
    """
    Save the trained model
    """
    logger.info("Saving the model: '{}'".format(filename))

    filepath = os.path.join(model_dir, filename) if model_dir is not None else filename
    with open(filepath, 'wb') as f:
        torch.save({
            'epoch': epoch,
            'batch_size': batch_size,
            'dataset': dataset_name,
            'model_in_channels': model.model_input_channels(),
            'model_out_channels': model.model_output_channels(),
            'model_up_mode': model.model_up_mode(),
            'model_padding': model.model_padding(),
            'criterion': loss_function,
            'optimizer': optimizer_algorithm,
            'model_state_dict': model.state_dict()
        }, f)



def load_model(file_path='../models/ORCA.pth', img_input_size=(640, 640), use_cuda=True):
    # Checking for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_cuda else "cpu"

    # load the trained model
    checkpoint = torch.load(file_path) if torch.cuda.is_available() else torch.load(file_path, map_location=lambda storage, loc: storage)

    model_in_channels = checkpoint['model_in_channels']
    model_out_channels = checkpoint['model_out_channels']
    model_up_mode = checkpoint['model_up_mode']
    model_padding = key_check('model_padding', checkpoint, True)

    # recreate the model
    with torch.no_grad():
        model = UNet(in_channels=model_in_channels,
                     out_channels=model_out_channels,
                     up_mode=model_up_mode,
                     padding=model_padding,
                     img_input_size=img_input_size).to(device) if use_cuda else UNet(in_channels=model_in_channels,
                                                                                     out_channels=model_out_channels,
                                                                                     up_mode=model_up_mode,
                                                                                     padding=model_padding,
                                                                                     img_input_size=img_input_size)
        model.load_state_dict(checkpoint['model_state_dict'])

    logger.info('\t Model loaded on: {} / {} / {} / {} / {} params -> {}'.format(device,
                                                                                 model_in_channels,
                                                                                 model_out_channels,
                                                                                 img_input_size,
                                                                                 count_parameters(model),
                                                                                 file_path))
    return model



def key_check(key, arr, default):
    if key in arr.keys():
        return arr[key]
    return default



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
