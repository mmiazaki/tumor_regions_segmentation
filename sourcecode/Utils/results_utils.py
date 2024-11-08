import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path


# def calculate_avg_results(csv_input, csv_output, dataset_name):
#     #wsi_image	patch_image	class	auc	accuracy	precision	f1/dice	jaccard	sensitivity/recall	specificity	pixels	tp	tn	fp	fn
#     df = pd.read_csv(csv_input)
#
#     measures = ['accuracy', 'precision', 'f1/dice', 'jaccard', 'sensitivity/recall', 'specificity']
#     avg = [dataset_name]
#     for m in measures:
#         avg.append(df[m].mean())
#         #print("{}: {}".format(m, avg[-1]))
#     dfm = pd.DataFrame([avg], columns=['dataset']+measures)
#     dfm.to_csv(csv_output, mode = 'a', index = False, header = not os.path.isfile(csv_output), sep = ";", decimal= ",")


def calculate_results(csv_input, csv_output_epoch, csv_output_final, dataset_name):
    #csv_input: model,augmentation,phase,epoch,loss,accuracy,TP,TN,FP,FN,date,transformations
    df = pd.read_csv(csv_input)

    # train = df.loc[(df['phase'] == 'train'), 'TP':'FN'].astype(float)
    test = df.loc[(df['phase'] == 'test'), 'TP':'FN'].astype(float)

    # accuracy_train = (train["TP"] + train["TN"]) / (train["TP"] + train["TN"] + train["FP"] + train["FN"])
    # precision_train = train["TP"] / (train["TP"] + train["FP"])
    # f1_train = (2 * train["TP"]) / (2 * train["TP"] + train["FP"] + train["FN"])
    # iou_train = train["TP"] / (train["TP"] + train["FP"] + train["FN"])
    # sensitivity_train = train["TP"] / (train["TP"] + train["FN"])
    # specificity_train = train["TN"] / (train["TN"] + train["FP"])

    accuracy_test = (test["TP"] + test["TN"]) / (test["TP"] + test["TN"] + test["FP"] + test["FN"])
    precision_test = test["TP"] / (test["TP"] + test["FP"])
    f1_test = (2 * test["TP"]) / (2 * test["TP"] + test["FP"] + test["FN"])
    iou_test = test["TP"] / (test["TP"] + test["FP"] + test["FN"])
    sensitivity_test = test["TP"] / (test["TP"] + test["FN"])
    specificity_test = test["TN"] / (test["TN"] + test["FP"])

    dfm1 = pd.concat([accuracy_test, precision_test, f1_test, iou_test, sensitivity_test, specificity_test], axis=1)
    dfm1.reset_index(inplace=True, drop=True)
    dfm1.rename(columns={0: "accuracy", 1: "precision", 2: "f1", 3: "iou", 4: "sensitivity", 5: "specificity"}, inplace=True)

    dfm1.to_csv(csv_output_epoch, index=False)
    #dfm.plot()
    #plt.show()

    measures = ['accuracy', 'precision', 'f1', 'iou', 'sensitivity', 'specificity']
    min = [dataset_name, 'min']
    mean = [dataset_name, 'mean']
    max = [dataset_name, 'max']
    last = [dataset_name, 'last']
    for m in measures:
        min.append(dfm1[m].min())
        mean.append(dfm1[m].mean())
        max.append(dfm1[m].max())
        last.append(dfm1[m][-1:].item())
        print('{} (min/mean/max/last): ({} / {} / {} / {})'.format(m, dfm1[m].min(), dfm1[m].mean(), dfm1[m].max(), dfm1[m][-1:].item()))

    dfm2 = pd.DataFrame([min, mean, max, last], columns=['dataset', 'type']+measures)
    dfm2.to_csv(csv_output_final, mode = 'a', index = False, header = not os.path.isfile(csv_output_final), sep = ";", decimal= ",")




def plot_graph(csv_filename, fig_filename, n_epochs, legend=False, show=False, title='FCN-RCAug'):
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
        #for i in range(1, 401):
        #    xdata1.append(i)
        xdata1 = list(range(1, n_epochs+1))

    plt.rcParams["font.family"] = "serif"
    fig = plt.figure()
    fig.set_figheight(3.5)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xdata1, ydata1, color='lime', label='Train Loss', linewidth=0.8)
    ax.plot(xdata1, ydata2, color='darkorange', label='Test Loss', linewidth=0.8)
    ax.plot(xdata1, ydata3, color='mediumblue', label='Train Accuracy', linewidth=0.8)
    ax.plot(xdata1, ydata4, color='red', label='Test Accuracy', linewidth=0.8)

    ax = plt.gca()
    ax.set_xlim([0, n_epochs])
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
