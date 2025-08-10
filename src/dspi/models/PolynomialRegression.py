import numpy as np

class PolynomialRegression:
    def __init__(self):
        pass

    def fit(self, x, y, order):
        """
        Fits a polynomial regression model to the input data.
        
        :param x: array-like, independent variable.
        :param y: array-like, dependent variable.
        :param order: int, the polynomial degree.
        :return: coeffs: array, polynomial coefficients in ascending order.
        """
        # 验证输入
        if len(x) != len(y):
            raise ValueError("The size of x and y arrays are different")
        
        # 构建 Vandermonde 矩阵
        X = np.vander(x, N=order+1, increasing=True)
        
        # 使用最小二乘法求解系数
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        
        return coeffs

    def predict(self, x, coeffs):
        """
        Predicts the y values using the fitted polynomial model.
        
        :param x: array-like, independent variable.
        :param coeffs: array, polynomial coefficients in ascending order.
        :return: y_pred: array, predicted y values.
        """
        y_pred = np.polyval(coeffs[::-1], x)
        return y_pred
