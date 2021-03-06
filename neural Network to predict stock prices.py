#Libraries importieren
import numpy as np
import pandas_datareader as web
import pandas as pd
import matplotlib.pyplot as plt


#Neuronales Netzwerk als Klasse definieren

class neuronalesNetzwerk:

    def __init__(self, neingaben, nverborgen, nausgaben = 1):
        #Anzahl Knoten pro Eingabeschicht, verborgene Schicht, und Ausgabeschicht
        self.n_eingaben = neingaben
        self.n_verborgen = len(nverborgen)
        self.n_ausgaben = nausgaben
        self.nN_aufbau = [self.n_eingaben] + nverborgen + [self.n_ausgaben]
        #Gewichtungs "W" und Bias "B" dictionarys definieren
        self.W = {}
        self.B = {}
        #den dictionarys "W" und "B" werden "key: values pairs" (Schlüssel-Objekt-Paare) hinzugefügt
        #diese entsprechen der Position zwischen zwei Schichten, und den Gewichtungen und Bias an dieser Position
        for i in range(self.n_verborgen + 1):
            #die Gewichtungen werden mit zufälligen Zahlen gefüllt, welche eine Normalverteilung vom Kehrbruch der Wurzel von den Anzahl der Eingaben des Knotens um den Mittelpunkt 0 aufweisen
            self.W[i + 1] = np.random.normal(0.0, pow(self.nN_aufbau[i], - 0.5), (self.nN_aufbau[i],self.nN_aufbau[i + 1]))
            #die Bias werden mit dem Wert 0 gefüllt
            self.B[i + 1] = np.zeros((1,self.nN_aufbau[i + 1]))

#die Aktivierungsfunktion wird definiert
    def sigmoid(self, eingabe):
        return 1.0  / (1.0 + np.exp(-eingabe))

#definieren des feedforward-Algorithmus
    def feedforward(self, eingaben):
        #zwei libraries definieren
        #"V" speichert das Aktivierungpotenzial (die Summe der gewichteten Ausgaben von den Knoten der vorherigen Schicht, addiert mit dem dazugehörigen Bias)
        self.V = {}
        #"O" speichert die Ausgaben der Schicht (sigmoid("V"))
        self.O = {}
        self.O[0] = eingaben.reshape(1, -1)
        for i in range(self.n_verborgen + 1):
            #das Aktivierungspotenzial berechnen
            self.V[i + 1] = np.matmul(self.O[i], self.W[i + 1]) + self.B[i + 1]
            #die Sigmoidfunktion auf die Ausgaben der Schicht anwenden
            self.O[i + 1] = self.sigmoid(self.V[i + 1])
        finale_ausgabe = self.O[self.n_verborgen + 1]
        return finale_ausgabe

#die Ableitung der Sigmoidfunktion für den backpropagation-Algorithmus berechnen
    def ableitung_sigmoid(self, ausgabe):
        return ausgabe * (1 - ausgabe)

#definieren des backpropagation-Algorithmus
    def backpropagation(self, eingaben, korrekte_daten):
        #die Eingaben werden durch den feedforward-Algorithmus gelassen
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
            #transponierte Matrix der Ausgaben von den Knoten der vorherigen Schicht * Fehler (von den Knoten der nächsten Schicht) * Ableitung von Sigmoid (von den Ausgaben der nächsten Schicht)
            self.dW[i] = np.matmul(self.O[i - 1].T, np.multiply(self.E[i], self.ableitung_sigmoid(self.O[i])))
            #"dB" ist:
            #Fehler (von den Knoten der nächsten Schicht) * Ableitung von Sigmoid (von den Ausgaben der nächsten Schicht)
            self.dB[i] = np.multiply(self.E[i], self.ableitung_sigmoid(self.O[i]))
            #der Fehler "E" berechnet sich aus dem Fehler (der Knoten von der nächsten Schicht) multipliziert mit der transponierten Matrix der Gewichtungen
            self.E[i - 1] = np.matmul(self.E[i], self.W[i].T)

#definieren des mean squared error (mittlere quadratische Abweichung)
    def mean_squared_error(self, vorhersagen, korrekte_daten_liste):
        return np.square(np.subtract(korrekte_daten_liste, vorhersagen)).mean()

#definieren des Trainingsalgorithmus
    def train(self, eingabe_liste, korrekte_daten_liste, wiederholungen, lernfaktor, initialise=True, display_loss=False):
        #wenn "initialise" auf "True" gesetzt ist, werden die Gewichtungen und Biases nochmals mit zufälligen Zahlen gefüllt (wie beim Konstruktor der Klasse)
        #bei "False" kann das neurale Netzwerk weiter trainiert werden
        if initialise:
            for i in range(self.n_verborgen+1):
                self.W[i+1] = np.random.normal(0.0, pow(self.nN_aufbau[i], - 0.5), (self.nN_aufbau[i],self.nN_aufbau[i + 1]))
                self.B[i+1] = np.zeros((1, self.nN_aufbau[i+1]))
        #wenn "display_loss" auf "True" gesetzt ist, wird  eine Grafik mit den Wiederholungen als x-Achse und dem mean squared error als y-Achse dargestellt
        #hier wird ein dictionary definiert
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
                #der Lernfaktor gewichtet die Änderungen welche an den Gewichtungen und Bias vorgenommen werden
                self.W[i+1] -= lernfaktor * dW[i+1]
                self.B[i+1] -= lernfaktor * dB[i+1]
            #der mean squared error wird ausgerechnet
            if display_loss:
                Y_vorhersagen = self.predict(eingabe_liste)
                loss[e] = self.mean_squared_error(Y_vorhersagen, korrekte_daten_liste)
        #der mean squared error wird in einer Grafik dargestellt
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


#den mean absolute percentage error definieren
def mape(vorhersagen, korrekte_daten):
    return np.mean(np.abs((korrekte_daten - vorhersagen) / korrekte_daten)) * 100


#Variabeln definieren
aktienkurs = 'GOOG'
startdatum = '2010-01-01'
enddatum = '2020-01-01'
testdatensatz = 0.1
training_test_verhältniss = 85/15
anzahl_neuronen = 20
NN_aufbau = [7, anzahl_neuronen, anzahl_neuronen, anzahl_neuronen, 1]
wiederholungen = 9000
lernfaktor = 0.0025


#Daten mit der Library "pandas_datareader" von "Yahoo! Finance" importieren
Daten = web.DataReader(aktienkurs, 'yahoo', startdatum, enddatum)
#Isolieren des Börsenschluss-Preis (Close price)
close = Daten.filter(['Close'])

#Erhalten aller Wochentage
alle_wochentage = pd.date_range(start = startdatum, end = enddatum, freq = 'B')
close = close.reindex(alle_wochentage)
#Lücken werden mit dem Wert des vorherigen Tages gefüllt
close = close.fillna(method = 'ffill')
#Falls es immer noch Lücken hat (an der ersten Stelle) werden diese mit dem Wert des nächsten Tages gefüllt
close = close.fillna(method = 'bfill')
#In ein NumPy array konvertieren
close = close.values

#Daten in Trainings- und Testdaten aufspalten
trainings_ende = int(np.floor(len(close) * (1-testdatensatz)))
trainings_start = int(np.floor(len(close) * (1 - ((testdatensatz*training_test_verhältniss)+testdatensatz))))
test_start = trainings_ende
test_ende = len(close)
trainingsdaten = close[np.arange(trainings_start, trainings_ende),:]
testdaten = close[np.arange(test_start, test_ende),:]

#Daten zwischen 0 und 1 skalieren
s_trainingsdaten = (trainingsdaten - np.amin(trainingsdaten)) / (np.amax(trainingsdaten) - np.amin(trainingsdaten))
s_testdaten = (testdaten - np.amin(testdaten)) / (np.amax(testdaten) - np.amin(testdaten))

#die trainings Eingaben und die Werte zum Überprüfen der Vorhersagen (korrekte Daten) werden in Form gebracht
s_training_eingaben = np.zeros(((len(s_trainingsdaten) - NN_aufbau[0]), NN_aufbau[0]))
for x in range(len(s_trainingsdaten) - NN_aufbau[0]):
    for i in range(NN_aufbau[0]):
        s_training_eingaben[x,i] = s_trainingsdaten[x + i]
s_training_korrekte_daten = np.zeros((len(s_trainingsdaten) - NN_aufbau[0]))
for x in range(len(s_trainingsdaten) - NN_aufbau[0]):
  s_training_korrekte_daten[x] = s_trainingsdaten[x+5]

#Dasselbe wird mit den Testdaten getan
s_test_eingaben = np.zeros(((len(s_testdaten) - NN_aufbau[0]), NN_aufbau[0]))
for x in range(len(s_testdaten) - NN_aufbau[0]):
    for i in range(NN_aufbau[0]):
        s_test_eingaben[x,i] = s_testdaten[x + i]
s_test_korrekte_daten = np.zeros((len(s_testdaten) - NN_aufbau[0]))
for x in range(len(s_testdaten) - NN_aufbau[0]):
  s_test_korrekte_daten[x] = s_testdaten[x+5]

#neuronales Netzwerk trainieren
nn = neuronalesNetzwerk(NN_aufbau[0], NN_aufbau[1:len(NN_aufbau)-1], NN_aufbau[len(NN_aufbau)-1])
nn.train(s_training_eingaben, s_training_korrekte_daten, wiederholungen, lernfaktor, display_loss=True)

#Vorhersagen der Testdaten berechnen
s_test_vorhersagen = nn.predict(s_test_eingaben)

#Skalierung rückgängig machen
test_vorhersagen = s_test_vorhersagen * (np.amax(testdaten) - np.amin(testdaten)) + np.amin(testdaten)
test_korrekte_daten = s_test_korrekte_daten * (np.amax(testdaten) - np.amin(testdaten)) + np.amin(testdaten)

#die Genauigkeit der Vorhersagen der Testdaten mit dem MAPE berechnen
mape_test = mape(test_vorhersagen, test_korrekte_daten)

#Ausgabe des Programms
print("künstliches neuronales Netzwerk zum Vorhersagen von Aktien des Unternehmens", aktienkurs)
print("Konfiguration: ", NN_aufbau)
print("Wiederholungen: ", wiederholungen)
print("Lernfaktor: ", lernfaktor)
print(training_test_verhältniss)
print()
print("Genauigkeit von den Vorhersagen der Testdaten")
print("MAPE: ", mape_test, "%")
