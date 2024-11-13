import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def main():
    parser = argparse.ArgumentParser(description='Plot delta_ln_pi against eta times delta_reward and perform linear regression.')
    parser.add_argument('--csv_file', type=str, default='./../../online_evaluation_results_1731485961.csv', help='Path to the CSV file containing the data.')
    parser.add_argument('--eta', type=float, default=1e4, help='Value of eta used in the calculations.')
    parser.add_argument('--output', type=str, default='delta_ln_pi_vs_eta_delta_reward.png', help='Path to save the plot image.')
    args = parser.parse_args()

    csv_file = args.csv_file
    eta = args.eta

    # Load the data
    df = pd.read_csv(csv_file)

    # Extract the relevant columns
    delta_reward = df['delta_reward'].values
    delta_ln_pi = df['delta_ln_pi'].values

    # Compute eta times delta_reward
    eta_delta_reward = eta * delta_reward

    # Perform linear regression with delta_ln_pi as x and eta_delta_reward as y
    slope, intercept, r_value, p_value, std_err = linregress(delta_ln_pi, eta_delta_reward)

    # Generate points for the best fit line
    x_fit = np.linspace(min(delta_ln_pi), max(delta_ln_pi), 100)
    y_fit = slope * x_fit + intercept

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(delta_ln_pi, eta_delta_reward, color='blue', label='Data Points')

    # Plot the best fit line
    plt.plot(x_fit, y_fit, color='red', label='Best Fit')

    # Plot the ideal line y = x
    plt.plot(x_fit, x_fit, color='green', linestyle='--', label='Ideal (y = x)')

    # Labels and title
    plt.xlabel(r'$\Delta \ln \pi$')
    plt.ylabel(r'$\eta \cdot \Delta r$')
    plt.title(r'Plot of $\eta \cdot \Delta r$ vs $\Delta \ln \pi$')
    plt.legend()
    plt.grid(True)

    # Display the plot
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()

    # Print the regression statistics
    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")
    print(f"R-squared: {r_value**2}")

if __name__ == "__main__":
    main()