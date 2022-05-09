import pandas as pd
import numpy as np

consumption_train_norm = np.load("../data/example_nondeterminist_cons_train.npy")[0:1 * 365 * 24]
consumption_valid_norm = np.load("../data/example_nondeterminist_cons_train.npy")[365 * 24:2 * 365 * 24]
consumption_test_norm = np.load("../data/example_nondeterminist_cons_test.npy")[0:1 * 365 * 24]

print("conso normée 1","consumption_train_norm", consumption_train_norm ,"consumption_valid_norm", consumption_valid_norm, "consumption_test_norm",consumption_test_norm)

# Scale consumption profile in [0,2.1kW] --> average max per day = 1.7kW, average per day is 18.3kWh
consumption_train = consumption_train_norm * 2.1
consumption_valid = consumption_valid_norm * 2.1
consumption_test = consumption_test_norm * 2.1

print("conso 2","max consumption_train", np.max(consumption_train) ,"max consumption_valid", np.max(consumption_valid), "max consumption_test", np.max(consumption_test))

# Get production profile in W/Wp in [0,1]
production_train_norm = np.load("../data/BelgiumPV_prod_train.npy")[0:1 * 365 * 24]
production_valid_norm = np.load("../data/BelgiumPV_prod_train.npy")[
                             365 * 24:2 * 365 * 24]  # determinist best is 110, "nondeterminist" is 124.9
production_test_norm = np.load("../data/BelgiumPV_prod_test.npy")[
                            0:1 * 365 * 24]  # determinist best is 76, "nondeterminist" is 75.2

print("PV 1", "production_train_norm", production_train_norm, "production_valid_norm", production_valid_norm, "production_test_norm", production_test_norm)

# Scale production profile : 12KWp (60m^2) et en kWh
production_train = production_train_norm * 12000. / 1000.
production_valid = production_valid_norm * 12000. / 1000.
production_test = production_test_norm * 12000 / 1000

print("PV 2", "production_train", np.max(production_train), "production_valid", np.max(production_valid), "production_test", np.max(production_test))