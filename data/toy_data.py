import datetime
import numpy as np


loc1 = [[datetime.datetime(2019, 1, 1, 19, 15), 20, 5],
        [datetime.datetime(2019, 1, 1, 16, 0), 10, 2],
        [datetime.datetime(2019, 1, 1, 12, 15), 8.0, 3]
        ]

loc2 = [[datetime.datetime(2019, 1, 1, 18, 15), 20, 4],
        [datetime.datetime(2019, 1, 1, 15, 0), 7.2, 1.5],
        [datetime.datetime(2019, 1, 1, 11, 15), 8.0, 3]
        ]

locations = [loc1, loc2]

days = 365*4
price_values = np.tile(np.concatenate((np.linspace(1,0,48),np.linspace(0,1,48))), days)
print(np.concatenate((np.linspace(1,0,48),np.linspace(0,1,48))))
price_keys = [datetime.datetime(2016,1,1,0,0) + datetime.timedelta(hours=0.25*x) for x in range(0, days*96)]
price = {k:v for k, v in zip(price_keys, price_values)}
print(price)

# states are represented as dictionaries: 
# state["time"] is the current time
# state["stations"] is a list of stations
# each station is a dictionary with keys 
# "is_car", "des_char", "per_char", "curr_dur"
