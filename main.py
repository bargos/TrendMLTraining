# Importação e bibliotecas
from functions import *

import pickle

reset = '\033[0m'        # Reseta a formatação
red = '\033[31m'         # Texto em vermelho
green = '\033[32m'       # Texto em verde
yellow = '\033[33m'      # Texto em amarelo
blue = '\033[34m'        # Texto em azul
magenta = '\033[35m'     # Texto em magenta
cyan = '\033[36m'        # Texto em cyan
white = '\033[37m'       # Texto em branco

# Previsão de têndencia com base nos indicadores
# Treina um modelo para calcular tendência com base nos indicadores


features=['Volume', 'Open', 'rsi', 'ema_5_t', 'sma_5_t', 'ema_10_t']#, 'sma_10_t',
#           'stoc_t', 'bol_t', 'adx_t', 'macd_t', 'macd', 'chai_t']


"""
features = ['Open', 'sma_10_t', 'bol_t', 'macd_t']

data = data_inter(Interval.in_monthly)
Y = data['profit2_t']
X = data[list(features)]

x_train, x_test, y_train, y_test = split_custom(X,Y)

norm = MinMaxScaler().fit(x_train)

X_train_norm = norm.transform(x_train)

X_test_norm = norm.transform(x_test)

best_hparams = {'max_depth': 9, 'max_features': 1, 'min_samples_leaf': 3, 'min_samples_split': 5,
                 'n_estimators': 64, 'n_jobs': 5}

ml_rf = train_RF(X_train_norm, y_train, **best_hparams)

predictions = ml_rf.predict(X_test_norm)

performance(X, x_train, y_train, y_test, predictions, ml_rf)

with open('Resultados/ml_RF.pkl', 'wb') as file:
    pickle.dump(ml_rf, file)
"""


"""
results_1 = pd.read_csv('Resultados/Resultado_1.csv')
results_2 = pd.read_csv('Resultados/Resultado_2.csv')
results_3 = pd.read_csv('Resultados/Resultado_3.csv')
results_4 = pd.read_csv('Resultados/Resultado_4.csv')

results_f = pd.concat([results_1, results_2]).reset_index(drop=True)
results_f = pd.concat([results_f, results_3]).reset_index(drop=True)
results_f = pd.concat([results_f, results_4]).reset_index(drop=True)
results_f.to_csv(f'Resultados/Resultado_Final.csv', index=False)
"""

print(recursive_test(features))


