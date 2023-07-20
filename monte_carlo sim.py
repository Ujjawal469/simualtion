import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

error=[]
def monte_carlo_option_pricing(S, K, r, sigma, T, num_paths ,N):
    option_values = []
    for M in range(1000,num_paths,1000 ):
        # Generate random price paths
        dt =T/N
        z = np.random.standard_normal((N, M))

        # Computation of defined variables
        nudt=(r - 0.5 * sigma**2) * dt
        volsdt=sigma * np.sqrt(dt)
        lns= np.log(S)

        delta_lnst=nudt +volsdt*z
        lnst=lns + np.cumsum(delta_lnst,axis=0)
        lnst= np.concatenate((np.full(shape=(1,M),fill_value=lns),lnst))

        # Price_paths calculation same_as = S * np.exp(lnst)
        price_paths = np.exp(lnst)


        # Calculate option payoffs
        call_payoffs = np.maximum(price_paths - K, 0)

        # Calculate Discount
        discount_factor = np.exp(-r * dt)
        option_value = discount_factor * np.sum(call_payoffs[-1])/M

        # Calcualtion of standard error
        si=np.sqrt(np.sum( (call_payoffs[-1] - option_value)**2)/(M-1))
        SE = si/np.sqrt(M)
        error.append(SE)

        option_values.append(option_value)

    return option_values

S = 100  # Current price of the underlying asset
K = 105  # Strike price
r = 0.025  # Risk-free interest rate
sigma = 0.2  # Volatility of the underlying asset
T = 2/3  # Time to expiration in years
num_paths = 70000 # Number of simulated price paths
N= 63 # no.of trading days or no. of time steps
# must be ideally lower doesnt matter in overall result as much
# but N large gives accurate result but is computationally costlier


option_values = monte_carlo_option_pricing(S, K, r, sigma, T, num_paths,N)

# Plot convergence
plt.plot(option_values)
plt.xlabel('NO. of simulated price paths * 10^-3-->')
plt.ylabel('Option Value-->',)
plt.title('Convergence of Monte Carlo Simulation')

plt.show()
l= option_values.pop()
print('the call value ='+ str(l))

# error values
plt.plot(error)
plt.ylabel('+/- standard deviation')
plt.xlabel('no.of simulation * 10^-3--->')
plt.show()
l2 = error.pop()
print('the error SE is upto '+'+/-'+ str(l2))