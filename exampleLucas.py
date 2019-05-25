from backtesting import evaluateHist
from strategy import Strategy
from order import Order
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

class EMAVG(Strategy):
  def __init__(self):
    self.signal = 0
    self.prices = []
    self.window = 40
    self.ema = 0
    self.prev_ema = 0
    self.sm_coef = 2
    self.counter = 0
    
  def push(self, event):
    price = event.price[3]
    self.prices.append(price)
    orders = []

    if len(self.prices) > self.window:
      if self.ema == 0:
        self.prev_ema = sum(self.prices[-(self.window + 1):-1])/self.window
      mult = self.sm_coef/(1 + self.window)
      self.ema = price*(mult) + self.prev_ema*(1 - mult)

    if self.ema > self.prev_ema and self.signal != 1:
      if self.signal == -1:
        orders.append(Order(event.instrument, 1, 0))
      orders.append(Order(event.instrument, 1, 0))
      self.signal = 1
    elif self.ema < self.prev_ema and self.signal != -1:
      if self.signal == 1:
        orders.append(Order(event.instrument, -1, 0))
      orders.append(Order(event.instrument, -1, 0))
      self.signal = -1

    self.prev_ema = self.ema
    self.counter += 1
    return orders


class LinRegStrat(Strategy):
  def __init__(self):
    self.signal = 0
    self.prices = []
    self.train_set = []
    self.labels = []
    self.train_size = 2000
    self.is_trained = False
    self.regr = linear_model.LinearRegression()
    self.pred_label = 0
    self.prev_pred_label = 0
    self.counter = 0
    self.window = 4

    
  def push(self, event):
    price = event.price[3]
    self.prices.append(price)
    orders = []

    
    if len(self.prices) == self.window:
      if len(self.train_set) < self.train_size:
        print(len(self.train_set))
        if len(self.train_set) > 0:
          self.labels.append(price)
        self.train_set.append(self.prices)
      else:
        if not self.is_trained:
          self.labels.append(price)
          labels = np.asarray(self.labels, dtype=np.float32)

          train_set = np.asarray(self.train_set, dtype=np.float32)
          print(train_set.shape, labels.shape)
          self.regr.fit(train_set, labels)
          self.is_trained = True
        self.prev_pred_label = self.pred_label
        self.pred_label = self.regr.predict(self.prices)

        if self.pred_label > self.prev_pred_label and self.signal != 1:
          if self.signal == -1:
            orders.append(Order(event.instrument, 1, 0))
          orders.append(Order(event.instrument, 1, 0))
          self.signal = 1
        elif self.pred_label < self.prev_pred_label and self.signal != -1:
          if self.signal == 1:
            orders.append(Order(event.instrument, -1, 0))
          orders.append(Order(event.instrument, -1, 0))
          self.signal = -1

      del self.prices[0]
      self.counter += 1

    return orders

print(evaluateHist(LinRegStrat(), {'IBOV':'^BVSP.csv'}))

