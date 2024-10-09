import pandas as pd
from statistics import mode, median, stdev
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import time

# Load the data
data = pd.read_csv('data/data_seqmol_BindingDB.csv')
print(len(data))

def analysis_proteins(data):
    # Get the length of the sequences
    seq_len = data['Sequence'].apply(lambda x: len(x)).tolist()
    data['Seq_len'] = seq_len
    
    # Get the mean length of the sequences
    mean_len = sum(seq_len)/len(seq_len)
    print(mean_len)

    # Get the mode length of the sequences
    mode_len = mode(seq_len)
    print(mode_len)

    # Get the median length of the sequences
    median_len = median(seq_len)
    print(median_len)
    
    # Get the standard deviation of the lengths
    std_len = stdev(seq_len)
    print(std_len)
    
    # Plot the distribution of the lengths
    sns.displot(seq_len, kde=True)
    plt.savefig('data/seq_len_hist.png')
    
    # Plot a boxplot of the lengths and check for outliers
    plt.figure(figsize=(4, 10))
    sns.boxplot(seq_len)
    plt.savefig('data/seq_len_boxplot.png')
    
    # Detect outliers
    q1 = data['Seq_len'].quantile(0.25)
    q3 = data['Seq_len'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    print(lower_bound, upper_bound)
    
    # Remove outliers from the data
    data = data[data['Seq_len'] < upper_bound]
    
    # Plot the distribution of the lengths without outliers
    sns.displot(data['Seq_len'], kde=True)
    plt.savefig('data/seq_len_hist_filt.png')
    # Get the metrics
    seq_len = data['Seq_len'].tolist()
    mean_len = sum(seq_len)/len(seq_len)
    print(mean_len)
    mode_len = mode(seq_len)
    print(mode_len)
    median_len = median(seq_len)
    print(median_len)
    std_len = stdev(seq_len)
    print(std_len)
    # Plot boxplot without outliers
    plt.figure(figsize=(4, 10))
    sns.boxplot(seq_len)
    plt.savefig('data/seq_len_boxplot_filt.png')
    
    
def analysis_molecules():
    # Get the length of the sequences
    seq_len = data['SMILES'].apply(lambda x: len(x)).tolist()

    # Filter the sequences with length smaller than 200
    #seq_len = [x for x in seq_len if x < 200]
    #print(len(seq_len))
    
    # Get the mean length of the sequences
    mean_len = sum(seq_len)/len(seq_len)
    print(mean_len)

    # Get the mode length of the sequences
    mode_len = mode(seq_len)
    print(mode_len)

    # Get the median length of the sequences
    median_len = median(seq_len)
    print(median_len)

    # Plot the distribution of the lengths
    sns.displot(seq_len, kde=True)
    plt.savefig('data/mol_len_hist_filt.png')
    

def calculate_molecular_weight():
    # Calculate the molecular weight of the molecules
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mw = data['SMILES'].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x))).tolist()
    sns.displot(mw, kde=True)
    plt.savefig('data/mol_weight_hist_filt.png')
    
    # To be considered small molecules, they should be smaller than 900 Da
    small_molecules = [x for x in mw if x < 900]
    print(len(small_molecules))

def scatterplot_mw_len():
    # Scatterplot of molecular weight and length of the molecules
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mw = data['SMILES'].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x))).tolist()
    seq_len = data['SMILES'].apply(lambda x: len(x)).tolist()
    
    plt.figure(figsize=(10, 6))
    p = sns.regplot(x=mw, y=seq_len, scatter_kws={'s':1, 'alpha':0.1}, line_kws={'color':'red'})
    #calculate slope and intercept of regression equation
    slope, intercept, r, p, sterr = scipy.stats.linregress(x=p.get_lines()[0].get_xdata(),
                                                        y=p.get_lines()[0].get_ydata())
    print(intercept, slope)
    #add regression equation to plot
    plt.text(2, 95, 'y = ' + str(round(intercept,3)) + ' + ' + str(round(slope,3)) + 'x')
    plt.xlabel('Molecular Weight')
    plt.ylabel('Length')
    plt.savefig('data/scatter_mw_len_filt.png')

# EXECUTIONS
time0 = time.time()
#analysis_proteins(data)
#analysis_molecules()
#calculate_molecular_weight()
#scatterplot_mw_len()

# Calculate how many sequences have a length smaller than 790
#print(len(data[data['Sequence'].apply(lambda x: len(x) < 790)]))

# calculate how many molecules have a length smaller than 80
#print(len(data[data['SMILES'].apply(lambda x: len(x) < 80)])) 

time1 = time.time() - time0
print(time1)