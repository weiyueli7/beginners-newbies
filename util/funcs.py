import numpy as np
import pandas as pd

def least_squares_regression(X, y):
    """ Find the least squares regression plane using the normal equations """
    return np.linalg.solve(X.T @ X, X.T @ y)


def mse_with_vars(df, vars, f) :    
    assert len(vars) == len(f)

    data = df.get(vars).values.tolist()

    design_mat = []
    for row in data:
        mat_row = [1]
        for i in range(len(row)):
            mat_row.append(f[i](row[i]))
        design_mat.append(mat_row)
    design_mat = np.array(design_mat)

    # design_mat = np.array([[1, f[0](x), f[1](y)] for x, y, rk in data])
    observ_vec = np.array(df['INCWAGE'].values)

    print(type(design_mat))
    w = least_squares_regression(design_mat, observ_vec)

    err_vec = observ_vec - design_mat @ w
    mse = np.dot(err_vec, err_vec) / len(data)

    return w, mse