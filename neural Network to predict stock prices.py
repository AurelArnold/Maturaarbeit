#Libraries importieren
import numpy as np
import pandas_datareader as web
import pandas as pd
import matplotlib.pyplot as plt


#Neuronales Netzwerk als Klasse definieren

class neuronalesNetzwerk:

    def __init__(self, neingaben, nverborgen, nausgaben = 1):
        #anzahl Knoten pro Eingabe-, Verborgene-, und Ausgabe Schicht
        self.n_eingaben = neingaben
        self.n_verborgen = len(nverborgen)
        self.n_ausgaben = nausgaben
        self.nN_aufbau = [self.n_eingaben] + nverborgen + [self.n_ausgaben]
        #Gewichtungs "W" und Bias "B" dictionarys definieren
        self.W = {}
        self.B = {}
        #den dictionarys "W" und "B" werden "key: values pairs" (Schlüssel-Objekt-Paare) hinzugefügt
        #diese entsprechen den Gewichtungen und Bias zwichen zwei layers
        for i in range(self.n_verborgen + 1):
            #die Gewichtungen werden mit zufälligen Zahlen gefüllt, welche eine definierte Normalverteilung um den Mittelpunkt 0 aufweisen
            self.W[i + 1] = np.random.normal(0.0, pow(self.nN_aufbau[i], - 0.5), (self.nN_aufbau[i],self.nN_aufbau[i + 1]))
            #die Bias werden mit dem Wert 0 gefüllt
            self.B[i + 1] = np.zeros((1,self.nN_aufbau[i + 1]))

#die Aktivierungsfunktion wird definiert
    def sigmoid(self, eingabe):
        return 1.0  / (1.0 + np.exp(-eingabe))

#definieren des FeedForward Algorithmus
    def feedforward(self, eingaben):
        #zwei libraries definieren
        #"V" speichert das Aktivierungpotenzial (die Summe der gewichteten Ausgaben von den Knoten der vorherigen Schicht, addiert mit dem dazugehörigen Bias)
        self.V = {}
        #"O" speichert die Ausgaben der Schicht (sigmoid("V"))
        self.O = {}
        self.O[0] = eingaben.reshape(1, -1)
        for i in range(self.n_verborgen + 1):
            #die Summe der gewichteten Ausgaben von den Knoten der vorherigen Schicht berechnen und mit den Bias addieren
            self.V[i + 1] = np.matmul(self.O[i], self.W[i + 1]) + self.B[i + 1]
            #die Sigmoidfunktion auf die Ausgaben der Schicht anwenden
            self.O[i + 1] = self.sigmoid(self.V[i + 1])
        finale_ausgabe = self.O[self.n_verborgen + 1]
        return finale_ausgabe

#die Ableitung der Sigmoidfunktion für den backpropagation Algorithmus berechnen
    def deriv_sigmoid(self, ausgabe):
        return ausgabe * (1 - ausgabe)

#definieren des backpropagation Algorithmus
    def backpropagation(self, eingaben, korrekte_daten):
        #die Eingaben werden durch den FeedForward Algorithmus gelassen
        self.feedforward(eingaben)
        #drei libraries definieren
        #"dW" speichert die Differenz der Gewichte um welche diese verändert werden müssen
        self.dW = {}
        #"dB" speichert die Differenz der Bias um welche diese verändert werden müssen
        self.dB = {}
        #"E" speichert den Fehler jedes einzelnen Knoten
        self.E = {}
        L = self.n_verborgen + 1
        self.E[L] = (self.O[L] - korrekte_daten)
        for i in range (L, 0, -1):
            #"dW" ist:
            #transponierte Matrix der Ausgaben von den Knoten der vorherigen Schicht * Fehler (von den Knoten der nächsten Schicht) * deriv_sigmoid(von den Asugben der nächsten Schicht)
            self.dW[i] = np.matmul(self.O[i - 1].T, np.multiply(self.E[i], self.deriv_sigmoid(self.O[i])))
            #"dB" ist:
            #Fehler (von den Knoten der nächsten Schicht) * deriv_sigmoid(von den Asugben der nächsten Schicht)
            self.dB[i] = np.multiply(self.E[i], self.deriv_sigmoid(self.O[i]))
            #der Fehler "E" berechnet sich aus dem Fehler (der Knoten von der nächsten Schicht) multipliziert mit der transponierten Matrix der Gewichtungen
            self.E[i - 1] = np.matmul(self.E[i], self.W[i].T)

#definieren des mean square error (mittlere quadratische Abweichung)
    def mean_squared_error(self, vorhersagen, korrekte_daten_liste):
        return np.square(np.subtract(korrekte_daten_liste, vorhersagen)).mean()

#definieren des trainings Algorithmus
    def train(self, eingabe_liste, korrekte_daten_liste, wiederholungen, lernfaktor, initialise=True, display_loss=False):
        #wenn initialise auf "True" gesetzt ist werden die Gewichtungen und Biases nochmals mit zufälligen Zahlen gefüllt (wie bei __init__())
        #bei "False" kann das neurale Netzwerk weiter trainiert werden
        if initialise:
            for i in range(self.n_verborgen+1):
                self.W[i+1] = np.random.normal(0.0, pow(self.nN_aufbau[i], - 0.5), (self.nN_aufbau[i],self.nN_aufbau[i + 1]))
                self.B[i+1] = np.zeros((1, self.nN_aufbau[i+1]))
        #wenn display_loss auf "true"gesetzt ist, wird  ein Diagramm dargestellt mit den epochs als x-Achse und dem mean square error als y-Achse
        #hier wird ein dictionarys definiert
        if display_loss:
            loss = {}
        #die Gewichtungen und Bias werden angepasst
        for e in range(wiederholungen):
            dW = {}
            dB = {}
            for i in range(self.n_verborgen+1):
                dW[i+1] = np.zeros((self.nN_aufbau[i], self.nN_aufbau[i+1]))
                dB[i+1] = np.zeros((1, self.nN_aufbau[i+1]))
            for eingaben, korrekte_daten in zip(eingabe_liste, korrekte_daten_liste):
                self.backpropagation(eingaben, korrekte_daten)
                for i in range(self.n_verborgen+1):
                    dW[i+1] += self.dW[i+1]
                    dB[i+1] += self.dB[i+1]
            for i in range(self.n_verborgen+1):
                #die learning_rate gewichtet die Änderungen welche an den Gewichtungen und Bias vorgenommen wird
                self.W[i+1] -= lernfaktor * dW[i+1]
                self.B[i+1] -= lernfaktor * dB[i+1]
            #der mean square error wird ausgerechnet
            if display_loss:
                Y_vorhersagen = self.predict(eingabe_liste)
                loss[e] = self.mean_squared_error(Y_vorhersagen, korrekte_daten_liste)
        #der mean square error wird in einem Diagramm dargestellt
        if display_loss:
            plt.plot(list(loss.values()))
            plt.xlabel('wiederholungen')
            plt.ylabel('Mean Squared Error')
            plt.show()

#den Algorithmus zum vorhersagen von Daten definieren
    def predict(self, eingabe_liste):
        Y_vorhersagen = []
        for eingaben in eingabe_liste:
            y_vorhersagen= self.feedforward(eingaben)
            Y_vorhersagen.append(y_vorhersagen)
        return np.array(Y_vorhersagen).squeeze()

#den root-mean-square error defieren
def rmse(vorhersagen, korrekte_daten):
    return np.sqrt(((vorhersagen - korrekte_daten) ** 2).mean())

#den mean absolute percentage error definieren
def mape(vorhersagen, korrekte_daten):
    return np.mean(np.abs((korrekte_daten - vorhersagen) / korrekte_daten)) * 100



#Variabeln definieren
aktienkurs = 'GOOG'
start_datum = '2010-01-01'
end_datum = '2020-01-01'
trainings_datensatz = 0.8
benutzter_trainings_datensatz = 1
NN_aufbau = [7, 7, 7, 7, 1]
wiederholungen = 1000
lernfaktor = 0.001

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
s_training_eingaben = np.zeros(((len(s_trainings_daten) - NN_aufbau[0]), NN_aufbau[0]))
for x in range(len(s_trainings_daten) - NN_aufbau[0]):
    for i in range(NN_aufbau[0]):
        s_training_eingaben[x,i] = s_trainings_daten[x + i]
s_training_korrekte_daten = np.zeros((len(s_trainings_daten) - NN_aufbau[0]))
for x in range(len(s_trainings_daten) - NN_aufbau[0]):
  s_training_korrekte_daten[x] = s_trainings_daten[x+5]

#die test inputs und die Werte zum überprüfen der Vorhersagen (actual values) werden in Form gebracht
s_test_eingaben = np.zeros(((len(s_test_daten) - NN_aufbau[0]), NN_aufbau[0]))
for x in range(len(s_test_daten) - NN_aufbau[0]):
    for i in range(NN_aufbau[0]):
        s_test_eingaben[x,i] = s_test_daten[x + i]
s_test_korrekte_daten = np.zeros((len(s_test_daten) - NN_aufbau[0]))
for x in range(len(s_test_daten) - NN_aufbau[0]):
  s_test_korrekte_daten[x] = s_test_daten[x+5]


#Neuronales Netzwerk trainieren
nn = neuronalesNetzwerk(NN_aufbau[0], NN_aufbau[1:len(NN_aufbau)-1], NN_aufbau[len(NN_aufbau)-1])
nn.train(s_training_eingaben, s_training_korrekte_daten, wiederholungen, lernfaktor, display_loss=True)

#Neuronales Netzwerk ausführen
s_training_vorhersagen = nn.predict(s_training_eingaben)
s_test_vorhersagen = nn.predict(s_test_eingaben)

#Skalierung rückgängig machen
training_vorhersagen = s_training_vorhersagen * (np.amax(trainings_daten) - np.amin(trainings_daten)) + np.amin(trainings_daten)
training_korrekte_daten = s_training_korrekte_daten * (np.amax(trainings_daten) - np.amin(trainings_daten)) + np.amin(trainings_daten)
test_vorhersagen = s_test_vorhersagen * (np.amax(test_daten) - np.amin(test_daten)) + np.amin(test_daten)
test_korrekte_daten = s_test_korrekte_daten * (np.amax(test_daten) - np.amin(test_daten)) + np.amin(test_daten)

#die Genauigkeit der trainings daten und der test daten mit dem rmse und dem mape berechnen
rmse_training = rmse(training_vorhersagen, training_korrekte_daten)
mape_training = mape(training_vorhersagen, training_korrekte_daten)
rmse_test = rmse(test_vorhersagen, test_korrekte_daten)
mape_test = mape(test_vorhersagen, test_korrekte_daten)

#Ausgabe des Programms
print("Deep Feedforward Neural Network zur Vorhersage von Aktien der Firma", aktienkurs)
print("structure: ", NN_aufbau)
print("wiederholungen: ", wiederholungen)
print("lernfaktor: ", lernfaktor)
print()
print("Trainingsdaten")
print("RMSE: ", rmse_training)
print("MAPE: ", mape_training, "%")
print()
print("Testdaten")
print("RMSE: ", rmse_test)
print("MAPE: ", mape_test, "%")
