import pandas as pd
import matplotlib.pyplot as plt

def plot_train_metrics(csv,x_variable, y_variables, out, x_label='Epochs', y_label='Metrics', title='Training Metrics'):
    
    df = pd.read_csv(csv)
    
    # plot the training metrics
    plt.figure(figsize=(10, 6))
    for y_variable in y_variables:
        plt.plot(df[x_variable], df[y_variable], label=y_variable)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(out)
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot training metrics from a CSV file.')
    parser.add_argument('--csv', type=str, help='Path to the CSV file containing training metrics')
    parser.add_argument('--x_var', type=str, help='Column name for the x-axis variable')
    parser.add_argument('--y_vars', nargs='+', type=str, help='Column names for the y-axis variables')
    parser.add_argument('--out', type=str, default='training_metrics.png', help='Output filename for the plot')
    parser.add_argument('--x_label', type=str, default='Epochs', help='Label for the x-axis')
    parser.add_argument('--y_label', type=str, default='Metrics', help='Label for the y-axis')
    parser.add_argument('--title', type=str, default='Training Metrics', help='Title of the plot')

    args = parser.parse_args()
    
    plot_train_metrics(args.csv, args.x_var, args.y_vars, args.out, \
                       x_label=args.x_label, y_label=args.y_label, title=args.title)
