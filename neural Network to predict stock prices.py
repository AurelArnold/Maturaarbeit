#Libraries importieren
import numpy as np
import pandas_datareader as web
import pandas as pd
import matplotlib.pyplot as plt


#Neuronales Netzwerk als Klasse definieren

class neuralNetwork:

    def __init__(self, ninputs, nhidden, noutputs = 1):
        #anzahl Knoten pro input-, hidden-, und output layer
        self.n_inputs = ninputs
        self.n_hidden = len(nhidden)
        self.n_outputs = noutputs
        self.nN_structure = [self.n_inputs] + nhidden + [self.n_outputs]
        #weight "W" und bias "B" dictionarys definieren
        self.W = {}
        self.B = {}
        #den dictionarys "W" und "B" werden "key: values pairs" (Schlüssel-Objekt-Paare) hinzugefügt
        #diese entsprechen den weights und biases zwichen zwei layers
        for i in range(self.n_hidden + 1):
            #die weights werden mit zufälligen Zahlen gefüllt, welche eine definierte Normalverteilung um den Mittelpunkt 0 aufweisen
            self.W[i + 1] = np.random.normal(0.0, pow(self.nN_structure[i], - 0.5), (self.nN_structure[i],self.nN_structure[i + 1]))
            #die biases werden mit dem Wert 0 gefüllt
            self.B[i + 1] = np.zeros((1,self.nN_structure[i + 1]))

#die Aktivierungsfunktion wird definiert
    def sigmoid(self, input):
        return 1.0  / (1.0 + np.exp(-input))

#definieren des FeedForward Algorithmus
    def feedforward(self, inputs):
        #zwei libraries definieren
        #"A" speichert die gewichtete Summe der inputs (oder outputs des vorherigen layer) addiert mit dem dazugehörigen bias
        self.A = {}
        #"H" speichert die outputs der layers (sigmoid("A"))
        self.H = {}
        self.H[0] = inputs.reshape(1, -1)
        for i in range(self.n_hidden + 1):
            #die Gewichtete Summe der inputs berechnen und mit den bias addieren
            self.A[i + 1] = np.matmul(self.H[i], self.W[i + 1]) + self.B[i + 1]
            #die outputs der layers mit der sigmoid-Funktion berechnen
            self.H[i + 1] = self.sigmoid(self.A[i + 1])
        final_output = self.H[self.n_hidden + 1]
        return final_output

#die Ableitung der sigmoid-Funktion für den backpropagation Algorithmus berechnen
    def deriv_sigmoid(self, output):
        return output * (1 - output)

#definieren des backpropagation Algorithmus
    def backpropagation(self, inputs, actual_value):
        #die inputs werden durch den FeedForward Algorithmus gelassen
        self.feedforward(inputs)
        #drei libraries definieren
        #"dW" speichert die Differenz der weights um welche diese verändert werden müssen
        self.dW = {}
        #"dB" speichert die Differenz der biases um welche diese verändert werden müssen
        self.dB = {}
        #"error" speichert den error (Fehler) jedes einzelnen Knoten
        self.error = {}
        L = self.n_hidden + 1
        self.error[L] = (self.H[L] - actual_value)
        for i in range (L, 0, -1):
            #"dW" ist:
            #transponierte Matrix von den outputs der vorherigen Knoten * error (des nächsten Knoten) * deriv_sigmoid(outputs der vorherigen Knoten)
            self.dW[i] = np.matmul(self.H[i - 1].T, np.multiply(self.error[i], self.deriv_sigmoid(self.H[i])))
            #"dB" ist:
            #error (des nächsten Knoten) * deriv_sigmoid(outputs der vorherigen Knoten)
            self.dB[i] = np.multiply(self.error[i], self.deriv_sigmoid(self.H[i]))
            #der "error" berechnet sich aus dem error (des nächsten Knoten) multipliziert mit der transponierten Matrix der weights
            self.error[i - 1] = np.matmul(self.error[i], self.W[i].T)

#definieren des mean square error (mittlere quadratische Abweichung)
    def mean_squared_error(self, predictions, actual_values_list):
        return np.square(np.subtract(actual_values_list, predictions)).mean()

#definieren des trainings Algorithmus
    def train(self, inputs_list, actual_values_list, epochs, learning_rate, initialise=True, display_loss=False):
        #wenn initialise auf "True" gesetzt ist werden die weights und biases nochmals mit zufälligen Zahlen gefüllt (wie bei __init__())
        #bei "False" kann das neural Network weiter trainiert werden
        if initialise:
            for i in range(self.n_hidden+1):
                self.W[i+1] = np.random.normal(0.0, pow(self.nN_structure[i], - 0.5), (self.nN_structure[i],self.nN_structure[i + 1]))
                self.B[i+1] = np.zeros((1, self.nN_structure[i+1]))
        #wenn display_loss auf "true"gesetzt ist, wird  ein Diagramm dargestellt mit den epochs als x-Achse und dem mean square error als y-Achse
        #hier wird ein dictionarys definiert
        if display_loss:
            loss = {}
        #die weights und biases werden angepasst
        for e in range(epochs):
            dW = {}
            dB = {}
            for i in range(self.n_hidden+1):
                dW[i+1] = np.zeros((self.nN_structure[i], self.nN_structure[i+1]))
                dB[i+1] = np.zeros((1, self.nN_structure[i+1]))
            for inputs, actual_value in zip(inputs_list, actual_values_list):
                self.backpropagation(inputs, actual_value)
                for i in range(self.n_hidden+1):
                    dW[i+1] += self.dW[i+1]
                    dB[i+1] += self.dB[i+1]
            for i in range(self.n_hidden+1):
                #die learning_rate gewichtet die Änderungen welche an den weights und biases vorgenommen wird
                self.W[i+1] -= learning_rate * dW[i+1]
                self.B[i+1] -= learning_rate * dB[i+1]
            #der mean square error wird ausgerechnet
            if display_loss:
                Y_pred = self.predict(inputs_list)
                loss[e] = self.mean_squared_error(Y_pred, actual_values_list)
        #der mean square error wird in einem Diagramm dargestellt
        if display_loss:
            plt.plot(list(loss.values()))
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.show()

#den Algorithmus zum vorhersagen von Daten definieren
    def predict(self, inputs_list):
        Y_pred = []
        for inputs in inputs_list:
            y_pred = self.feedforward(inputs)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()

#den root-mean-square error defieren
def rmse(predictions, actual_values):
    return np.sqrt(((predictions - actual_values) ** 2).mean())

#den mean absolute percentage error definieren
def mape(predictions, actual_values):
    return np.mean(np.abs((actual_values - predictions) / actual_values)) * 100



#Variabeln definieren
aktienkurs = 'MSFT'
start_datum = '2010-01-01'
end_datum = '2020-01-01'
trainings_datensatz = 0.8
benutzter_trainings_datensatz = 1
NN_structure = [7, 13, 13, 1]
epochs = 100
learning_rate = 0.01

#Daten mit der Library "pandas_datareader" von "Yahoo! Finance" importieren
Daten = web.DataReader(aktienkurs, 'yahoo', start_datum, end_datum)
#Isolieren des Börsenschluss-Preis (Close price)
close = Daten.filter(['Close'])

#Erhalten aller Wochentage
alle_wochentage = pd.date_range(start = start_datum, end = end_datum, freq = 'B')
close = close.reindex(alle_wochentage)
#Lücken werden mit dem Wert des vorherigen Tages gefüllt
close = close.fillna(method = 'ffill')
#Falls es immer noch Lücken hat (an der ersten Stelle) werden diese mit dem Wert des nächsten Tages gefüllt
close = close.fillna(method = 'bfill')
#In ein Numpy array konvertieren
close = close.values


#Daten in Trainings und Test Daten aufspalten
trainings_ende = int(np.floor(len(close) * trainings_datensatz))
trainings_start = int(np.floor(trainings_ende * (1 - benutzter_trainings_datensatz)))
test_start = trainings_ende
test_ende = len(close)
trainings_daten = close[np.arange(trainings_start, trainings_ende),:]
test_daten = close[np.arange(test_start, test_ende),:]

#Daten zwischen 0 und 1 skalieren
s_trainings_daten = (trainings_daten - np.amin(trainings_daten)) / (np.amax(trainings_daten) - np.amin(trainings_daten))
s_test_daten = (test_daten - np.amin(test_daten)) / (np.amax(test_daten) - np.amin(test_daten))

#die trainings inputs und die Werte zum überprüfen der Vorhersagen (actual values) werden in Form gebracht
s_training_inputs = np.zeros(((len(s_trainings_daten) - NN_structure[0]), NN_structure[0]))
for x in range(len(s_trainings_daten) - NN_structure[0]):
    for i in range(NN_structure[0]):
        s_training_inputs[x,i] = s_trainings_daten[x + i]
s_training_actual_values = np.zeros((len(s_trainings_daten) - NN_structure[0]))
for x in range(len(s_trainings_daten) - NN_structure[0]):
  s_training_actual_values[x] = s_trainings_daten[x+5]

#die test inputs und die Werte zum überprüfen der Vorhersagen (actual values) werden in Form gebracht
s_test_inputs = np.zeros(((len(s_test_daten) - NN_structure[0]), NN_structure[0]))
for x in range(len(s_test_daten) - NN_structure[0]):
    for i in range(NN_structure[0]):
        s_test_inputs[x,i] = s_test_daten[x + i]
s_test_actual_values = np.zeros((len(s_test_daten) - NN_structure[0]))
for x in range(len(s_test_daten) - NN_structure[0]):
  s_test_actual_values[x] = s_test_daten[x+5]


#Neuronales Netzwerk trainieren
nn = neuralNetwork(NN_structure[0], NN_structure[1:len(NN_structure)-1], NN_structure[len(NN_structure)-1])
nn.train(s_training_inputs, s_training_actual_values, epochs, learning_rate, display_loss=True)

#Neuronales Netzwerk ausführen
s_training_prediction = nn.predict(s_training_inputs)
s_test_prediction = nn.predict(s_test_inputs)

#Skalierung rückgängig machen
training_predictions = s_training_prediction * (np.amax(trainings_daten) - np.amin(trainings_daten)) + np.amin(trainings_daten)
training_actual_values = s_training_actual_values * (np.amax(trainings_daten) - np.amin(trainings_daten)) + np.amin(trainings_daten)
test_predictions = s_test_prediction * (np.amax(test_daten) - np.amin(test_daten)) + np.amin(test_daten)
test_actual_values = s_test_actual_values * (np.amax(test_daten) - np.amin(test_daten)) + np.amin(test_daten)

#die Genauigkeit der trainings daten und der test daten mit dem rmse und dem mape berechnen
rmse_train = rmse(training_predictions, training_actual_values)
mape_train = mape(training_predictions, training_actual_values)
rmse_test = rmse(test_predictions, test_actual_values)
mape_test = mape(test_predictions, test_actual_values)

#Ausgabe des Programms
print("Deep Feedforward Neural Network zur Vorhersage von Aktien der Firma", aktienkurs)
print("structure: ", NN_structure)
print("epochs: ", epochs)
print("learning rate: ", learning_rate)
print()
print("Trainingsdaten")
print("RMSE: ", rmse_train)
print("MAPE: ", mape_train, "%")
print()
print("Testdaten")
print("RMSE: ", rmse_test)
print("MAPE: ", mape_test, "%")
