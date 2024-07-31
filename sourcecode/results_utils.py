import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def save_results(csv_input, csv_output):
    path = Path(csv_input)
    if not path.is_file():
        with open(csv_output, mode='w+') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['model', 'augmentation', 'phase', 'epoch', 'loss', 'accuracy', 'TP', 'TN', 'FP', 'FN', 'date', 'transformations'])

    with open(csv_output, mode='a+') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        print('[{}] Loss: {:.6f}'.format(phase, epoch_loss[phase]))
        csv_writer.writerow([filename, augmentation, phase, epoch, epoch_loss[phase], epoch_acc[phase], epoch_tp[phase], epoch_tn[phase], epoch_fp[phase], epoch_fn[phase], datetime.datetime.now(), str(augmentation_operations).replace(",", "")])


def plot_graph(csv_filename, fig_filename, legend=False, show=False, title='FCN Training - RCAug'):
    xdata1 = []
    ydata1 = []
    ydata2 = []
    ydata3 = []
    ydata4 = []
    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row != []:
                #print(row[4], row[2])
                if row[4] != 'loss':
                    if row[2] == 'train':
                        ydata1.append(float(row[4]))
                    else:
                        ydata2.append(float(row[4]))
                if row[5] != 'accuracy':
                    if row[2] == 'train':
                        ydata3.append(float(row[5]))
                    else:
                        ydata4.append(float(row[5]))
        for i in range(1, 401):
            xdata1.append(i)

    plt.rcParams["font.family"] = "serif"
    fig = plt.figure()
    fig.set_figheight(3.5)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xdata1, ydata1, color='lime', label='Train Loss', linewidth=0.8)
    ax.plot(xdata1, ydata2, color='darkorange', label='Test Loss', linewidth=0.8)
    ax.plot(xdata1, ydata3, color='mediumblue', label='Train Accuracy', linewidth=0.8)
    ax.plot(xdata1, ydata4, color='red', label='Test Accuracy', linewidth=0.8)

    ax = plt.gca()
    ax.set_xlim([0, 400])
    ax.set_ylim([0, 1])
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Loss')
    plt.title(title)
    plt.grid()
    if legend:
        plt.legend(loc="lower left", bbox_to_anchor=(0.66, 0.43))
    plt.tight_layout()
    plt.savefig(fig_filename)
    if show:
        plt.show()

if __name__ == '__main__':
    csv_filename = '../datasets/ORCA_512x512/training/Test1-completo--orca_training_accuracy_loss_all.csv'
    fig_filename = '../datasets/ORCA_512x512/training/Test1-plot.png'
    plot_graph(csv_filename, fig_filename, True, True)