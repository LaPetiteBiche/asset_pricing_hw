import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
f = open("test.txt", 'w')
sys.stdout = f

# First bullet point
# Import csv
df = pd.read_csv('data/raw/HPR_daily.csv')
df2 = pd.read_csv('data/raw/Prices_daily.csv')

# Clean the data
df2.columns = ['date', 'msft', 'xom', 'ge', 'jpm', 'intc', 'c', 'spindx','1','2','3','4','5']
df.columns = ['date', 'msft', 'xom', 'ge', 'jpm', 'intc', 'c', 'vwretd', 'sprtrn','1','2','3']
df2 = df2.drop([0])
df = df.drop([0])
df2 = df2.drop(['1', '2', '3', '4', '5'], axis=1)
df = df.drop(['1', '2', '3'], axis=1)
df2 = df2.dropna()
df = df.dropna()

# Change object type
df2['date'] = pd.to_datetime(df2['date'])
cols = df2.columns[df2.dtypes.eq('object')]
df2[cols] = df2[cols].apply(pd.to_numeric, errors='coerce')
df['date'] = pd.to_datetime(df['date'])
cols = df.columns[df.dtypes.eq('object')]
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

# Save years into list (full years)
years = []
test = 0
for i in df['date'].dt.year :
    if i != test :
        years.append(i)
    test = i

# Set the index to date to select only the date of interest
df2 = df2.set_index(['date'])
df = df.set_index(['date'])

# Function to transform daily return to annual return


def annual_return_test(cp, year):
    a_r = df[[cp]].loc[f'{year}-01-01':f'{year}-12-31']
    deb = 100.0
    for i, row in a_r.iterrows():
        deb *= (1 + row)
    return((deb-100)/100)


# Only MSFT XOM and GE and until 2002-12-31, compute annual returns for each company and each year
df3 = df[['msft','xom','ge']].loc[:'2002-12-31']
years_reduced = list(years)
years_reduced[:] = [x for x in years_reduced if x < 2003]


y = np.array([])
yy = np.array([])
for col in df3.columns:
    for year in years_reduced:
        z = annual_return_test(col, year).to_numpy()
        y = np.concatenate((y, z), axis=0)
    if yy.size == 0:
        yy = y
    else:
        yy = np.c_[yy,y]
    y = np.array([])

# Print annual returns by company (1 company is 1 column) and print the matrix size
print("annual returns : " + f'{yy}')
print(yy.shape)

# Compute expected return (mean of the annual returns)


def expected_return(matrix):
   num_rows, num_cols = matrix.shape
   exp_ret = []
   for i in range(num_cols):
      z = 0
      for x in matrix[:,i]:
         z += x
      exp_ret.append(z/num_rows)
   return(exp_ret)


exp_ret = expected_return(yy)
print("Expected returns : " + f'{exp_ret}')

# Create empty array for portfolio weights
weights = np.zeros(3)

# Cov matrix annual returns
cov_ret = np.cov(np.transpose(yy))
print("cov : " + f'{cov_ret}')

# Compute return and std if equal-weighted (e)   --> TO DO return results to do covariance
for i in range(3):
    weights[i] = 1/3


def print_result(weights, exp_ret, cov_ret):
   print ("Weights : " + f'{weights}')
   print ("Return : " + f'{np.matmul(weights,np.transpose(exp_ret))}')
   var_midterm = np.matmul(weights,cov_ret)
   print ("Variance : " + f'{np.matmul(np.transpose(var_midterm), weights)}')
   #Sharpe Ratio with rf = 2%
   print("Sharpe Ratio : " + f'{(np.matmul(weights,np.transpose(exp_ret))-0.02)/np.matmul(np.transpose(var_midterm), weights)}')
   print("")

print("")
print("----Portfolio 1/3 1/3 1/3----")
print_result(weights, exp_ret, cov_ret)

# Compute return and std if weights 0.8, 0.4, -0.2
weights[0] = 0.8
weights[1] = 0.4
weights[2] = -0.2
print("----Portfolio 0.8, 0.4, -0.2----")
print_result(weights, exp_ret, cov_ret)

#Covariance with portfolio e
for i in range(3):
    weights[i] = 1/3
print("Covariance with portfolio e")
print(np.matmul(np.matmul(np.array([0.8, 0.4, -0.2]), cov_ret),weights))

# Min Var Portfolio
one = np.ones(3)
up = np.matmul(np.linalg.inv(cov_ret),one)
down1 = np.matmul(np.matmul(np.transpose(one),np.linalg.inv(cov_ret)),one)
mv_weights = up/down1
print("-----Minimum Variance Portfolio-----")
print_result(mv_weights, exp_ret, cov_ret)

# Efficient Portfolio return MSFT
A = np.matmul(np.matmul(np.transpose(exp_ret),np.linalg.inv(cov_ret)),exp_ret)
B = np.matmul(np.matmul(one,np.linalg.inv(cov_ret)),exp_ret)
C = np.matmul(np.matmul(one,np.linalg.inv(cov_ret)),np.transpose(one))
lambda1 = (C * exp_ret[0] - B )/ (A*C - B**2)
lambda2 = (A - B * exp_ret[0]) / (A*C - B**2)
w2 = np.matmul((lambda1 * np.linalg.inv(cov_ret)),exp_ret) + np.matmul((lambda2 * np.linalg.inv(cov_ret)),np.transpose(one))
print("-----Efficient Portfolio return = MSFT-----")
print_result(w2, exp_ret, cov_ret)

#Cov with Min Var portfolio
print("Covariance with MV portfolio")
print(np.matmul(np.matmul(mv_weights, cov_ret),w2))

#Efficient Frontier Plot
def Plot_EF(exp_ret, cov_ret, one):
    A = np.matmul(np.matmul(np.transpose(exp_ret), np.linalg.inv(cov_ret)), exp_ret)
    B = np.matmul(np.matmul(one, np.linalg.inv(cov_ret)), exp_ret)
    C = np.matmul(np.matmul(one, np.linalg.inv(cov_ret)), np.transpose(one))
    x = np.array([])
    y = np.array([])
    for i in range(500):
        lambda1 = (C * (i/1000) - B) / (A * C - B ** 2)
        lambda2 = (A - B * (i/1000)) / (A * C - B ** 2)
        weights = np.matmul((lambda1 * np.linalg.inv(cov_ret)), exp_ret) + np.matmul((lambda2 * np.linalg.inv(cov_ret)),np.transpose(one))
        ret = np.matmul(weights, np.transpose(exp_ret))
        var_midterm = np.matmul(weights, cov_ret)
        var = np.matmul(np.transpose(var_midterm), weights)
        x =np.append(x,ret)
        y =np.append(y,var)
    return (x,y)
x_graph, y_graph = Plot_EF(exp_ret, cov_ret, one)
x = [i for i in x_graph]
y = [i for i in y_graph]
plt.scatter(y,x)
plt.title("Efficient Frontier MSFT, GE, JPM 1990-2002 ")
plt.xlabel("variance")
plt.ylabel("return")
x1 = np.array([])
x2 = np.array([])
for i in range(3):
    x1 = np.append(x1, exp_ret[i])
    x2 = np.append(x2, cov_ret[i][i])
x = [i for i in x1]
y = [i for i in x2]
plt.scatter(y,x, marker = 'P')

plt.show()
f.close()
