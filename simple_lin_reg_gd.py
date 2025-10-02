import pandas as pd
import numpy as np

class SimpleLinearRegressionGD:
    def __init__(self, df, alpha=0.01, epochs=1000):
        self.df = df.copy()
        self.alpha = alpha
        self.epochs = epochs
        self.t0 = 0
        self.t1 = 0
        self.mse = None
        self.mean = None
        self.std = None

    def preprocess(self):
        arr = self.df.values
        self.mean = arr[:,0].mean()
        self.std = arr[:,0].std()
        self.features = (arr[:,0] - self.mean) / self.std
        self.target = arr[:,1]
        return arr

    def train(self):
        n = len(self.features)
        for _ in range(self.epochs):
            y_pred = self.t0 + self.t1 * self.features
            resid = y_pred - self.target
            self.t0 -= self.alpha * np.sum(resid) / n
            self.t1 -= self.alpha * np.sum(resid * self.features) / n
            self.mse = np.sum(resid**2) / (2 * n)

    def predict(self, x):
        """x: raw feature value, not scaled"""
        x_scaled = (x - self.mean) / self.std
        return self.t0 + self.t1 * x_scaled

    def summary(self):
        print("Final Model Parameters:")
        print(f"Theta0 = {self.t0:.3f}, Theta1 = {self.t1:.3f}, MSE = {self.mse:.2f}")
        print("\nDataset Info:")
        print(self.df.describe())

