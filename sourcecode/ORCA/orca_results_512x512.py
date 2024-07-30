from sourcecode.plot_graph_utils import *

if __name__ == '__main__':
    dir = '../../results/'
    filename = [
        '001-ORCA512-BCELoss',
        '002-ORCA512-BCELoss',
        '003-ORCA512-L1Loss',
        '004-ORCA512-MSELoss',
        '005-ORCA512-PoissonNLLLoss',
        '006-ORCA512-HuberLoss',
        '007-ORCA512-SmoothL1Loss',
        '008-ORCA512-L1Loss',
        '009-ORCA512-HingeEmbeddingLoss',
        '010-ORCA512-SoftMarginLoss']

    for fname in filename:
        print('Ploting: '+fname)
        plot_graph(dir+fname+'.csv', dir+fname+'.png', False, False, 'FCN Training - RCAug - '+fname[4:])
