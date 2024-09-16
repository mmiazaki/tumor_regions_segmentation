from sourcecode.results_utils import *

if __name__ == '__main__':
    dir = '../../results/_r001-ORCA512/'
    csv_input = 'orca_training_accuracy_loss_all.csv'
    csv_output_epoch = 'epoch_measures.csv'
    csv_output_final = 'final_measures.csv'

    dataset = [
        '012-ORCA512-BCELoss-Full',
        '013-ORCA512-L1Loss-Full',
        '014-ORCA512-MSELoss-Full',
        '015-ORCA512-HuberLoss-Full',
        '016-ORCA512-SmoothL1Loss-Full',
        '020-ORCA512-BCELoss-random8',
        '022-ORCA512-BCELoss',
        '023-ORCA512-BCELoss-random8',
        '024-ORCA512-BCELoss-random9',
        '025-ORCA512-BCELoss-random8',
        '026-ORCA512-L1Loss-random9']

    for ds in dataset:
        print('::: Ploting/calculating measures: '+ds+' :::')
        plot_graph(dir+ds+'/'+csv_input, dir+ds+'.png', 400, False, False, 'FCN - RCAug - '+ds)
        calculate_results(dir + ds + '/' + csv_input, dir + ds + '/' + csv_output_epoch, dir + csv_output_final, ds)
