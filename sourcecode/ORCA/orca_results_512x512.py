from sourcecode.results_utils import *
from sourcecode.evaluation_utils import *

if __name__ == '__main__':
    execute1 = False
    execute2 = True

    if execute1:
        csv_dir = '../../results/_r001-ORCA512/'
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
            print('::::: Plotting/calculating measures: ' + ds + ' :::::')
            plot_graph(csv_dir+ds+'/'+csv_input, csv_dir+ds+'.png', 400, False, False, 'FCN - RCAug - '+ds)
            calculate_results(csv_dir + ds + '/' + csv_input, csv_dir + ds + '/' + csv_output_epoch, csv_dir + csv_output_final, ds)



    # if execute2:
    #     # Evaluation dataset
    #     csv_dir = '../../results/evaluation/'
    #     csv_output = '../../results/evaluation/final_avg_measures.csv'
    #
    #     dataset = [
    #         '022-ORCA512-BCELoss-random9',
    #         '023-ORCA512-BCELoss-random8',
    #         '024-ORCA512-BCELoss-random9',
    #         '025-ORCA512-BCELoss-random8',
    #         '026-ORCA512-L1Loss-random9',
    #         '027-ORCA512-L1Loss-random8',
    #         '028-ORCA512-MSELoss-random8',
    #         '029-ORCA512-MSELoss-random9']
    #
    #     for ds in dataset:
    #         print('::::: Calculating measures (evaluation): ' + ds + ' :::::')
    #         csv_input = csv_dir+ds+'.csv'
    #         calculate_avg_results(csv_input, csv_output, ds)
