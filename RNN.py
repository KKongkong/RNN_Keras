import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.advanced_activations import LeakyReLU


def sub_Sampler(sample, subSampleLength):
    '''
    Pass a window of size 1 x sampleLength 
    over the data and return every chunk
    '''
    if subSampleLength <= 0:
        print("subSampleLength must be greater than or equal to 0")
        return -1
    
    #i is lower bound, j is upper bound
    i = 0
    j = subSampleLength 
    fullSampleLength = len(sample)
    
    if subSampleLength > fullSampleLength:
        print("subSampleLength provided was longer than the sample itself")
        return -1
    
    #move the window along the sample and save every sub sample
    subSamples = []
    while j <= fullSampleLength:
        subSamples.append(sample[i:j])
        i += 1
        j += 1
    
    return np.array(subSamples)


def read_Data(sampleLength=120, trainSplit=0.9):
    '''
    read in the data provided in the csv file
    return train and test data
    
    sampleLength controls how long each sample fed to the RNN will be
    trainSplit is a percentage for how much data will be used for training
    '''
    
    f = pd.read_csv("passenger-miles-mil-flown-domest.csv")
    
    #don't need month and year, only need the data to be in the right order
    f = f.drop("Month",1)
    
    #get all the values for the number of domestic flight miles
    Y = f.as_matrix(
        columns=["Passenger miles (Mil) flown domestic U.K. Jul. ?62-May ?72"])\
        .flatten()
    
    #set up X axis
    length = len(Y)
    X = np.linspace(0, length-1, num=length, endpoint=True)
    
    #set up interpolation function to provide more data to the network
    f = interp1d(X, Y, kind='cubic', copy=False)
    
    #n is how many steps between each integer
    #for example, setting to 10 produces values [0, 0.1, 0.2, 0.3, 0.4, ...]
    #setting to 2 produces [0, 0.5, 1, 1.5, 2, ...]
    n=10
    newX = np.linspace(0, length-1, num=(length*n)-(n-1), endpoint=True)
    
    #interpolate
    newY = f(newX)
    
    #one way of normalizing
    #inX = (newY-np.mean(newY))/np.std(newY)
    
    #another way of normalizing
    inX = newY/np.max(newY)
    
    #ensure all samples are of size "sampleLength"
    subSamples = sub_Sampler(inX, sampleLength)
    
    #error checking for the return of subSamples
    if type(subSamples) == int:
        if subSamples == -1:
            sys.exit(1)
    elif subSamples.shape[0] == 0:
        sys.exit(1)
    
    #split into training data and testing data
    split = int(trainSplit*subSamples.shape[0])
    trainingData = subSamples[:split, :]
    
    #ensure randomness
    np.random.shuffle(trainingData)
    
    #yTrain is the last value of each of the samples and xTrain is the series 
    #leading up to the last value (which is going to be predicted)
    xTrain = trainingData[:, :-1]
    yTrain = trainingData[:, -1]
    
    #testing data for evaluating the generalizability of the model
    xTest = subSamples[split:, :-1]
    yTest = subSamples[split:, -1]
    
    #reshape to be accepted by the network
    xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
    xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))  
    
    return xTrain, yTrain, xTest, yTest


def RNN(layers=[1,10,10,1], dropoutPercent=0.2, opt="Nadam", lossFunc="mse", 
        leakyAlpha=0.3):
    '''
    takes layers and builds an RNN in the same shape, returns the model
    
    
    if layers = [1, 20, 40, 1]
    then there's 1 input into the first LSTM which has 20 units
    those are then fed into a LSTM of 40 units, final output has size 1
    
    dropoutPercent - the percent of nodes ignored from the previous layer
    
    opt - optimizer, provide any of the accepted optimizer strings for keras
    
    lossFunc - the loss function, provide any of the accepted loss strings for
               keras
    
    leakyAlpha - the learning rate for the leakyRelu layer learning how leaky
                 it should be
    '''
    #using the Keras Sequential model
    m = Sequential()
    
    #add the first LSTM, (separated as it needs an input shape)
    m.add(LSTM(layers[1], 
               return_sequences=True, 
               input_shape=(None, layers[0])))
    
    #apply dropout to help prevent overfitting
    m.add(Dropout(dropoutPercent))
    
    #add all but the last of the LSTM layers
    for i in range(2, len(layers) - 2):
        m.add(LSTM(
            layers[i],
            return_sequences=True))
        m.add(Dropout(dropoutPercent))
    
    #need return sequences to be false in order to have a final prediction
    m.add(LSTM(
        layers[-2],
        return_sequences=False))
    m.add(Dropout(dropoutPercent))  
    
    #fully connected layer, then a leakyReLU activation function to allow
    #some gradient signal to pass through even when the gradient is less
    #than 0, prevent dead neurons
    m.add(Dense(layers[-1]))
    
    m.add(LeakyReLU(alpha=leakyAlpha))

    #compile and return model
    #default is mean squared error and the Nadam optimizer 
    #(Adam RMSprop + nesterov momentum)
    m.compile(loss=lossFunc, optimizer=opt)    
    return m


def train_Net(xTrain, yTrain, xTest, yTest, model, 
              batchSize=500, numEpochs=3000, verbose=1, showStep=10):
    '''
    trains the provided model with the training data and verifies the
    generalizability of the model by graphing the training data versus
    the model's prediction, returns the trained model.
    
    If verbose is set to 1, then this graph will be displayed every "showStep"
    time steps, if verbose is set to 0, then the graph will only be displayed
    at the end of training
    
    
    xTrain, yTrain, xTest, yTest - training and testing data, received from
                                   read_Data()
    
    model - model to be trained, built by RNN()
    
    batchSize - how large each batch should be
    
    numEpochs - how many time steps the training occurs for
    
    verbose - controls how much information is displayed
    
    showStep - controls how often the model's prediction is graphed against the
               actual test data, only occurs if verbose is set to 1
    
    '''
    if verbose==1:
        for i in range(0, numEpochs, showStep):
            #run "showStep" more training steps
            model.fit(
                xTrain,
                yTrain,
                validation_split=0.05,
                batch_size=512,
                epochs=showStep+i,
                initial_epoch = i)
            
            #calculate the model's prediction on the test data
            predicted = model.predict(xTest)
            predicted = np.reshape(predicted, (predicted.size,))
            
            #set x axis
            length = len(predicted)
            x_s = np.linspace(0, length-1, num=length, endpoint=True)
            
            #plot the test data against the prediction
            plt.clf()
            plt.plot(x_s, predicted, "-" , x_s, yTest, "-")
            plt.legend(labels=["Predicted", "Test data"])
            plt.draw()
            plt.pause(0.0001)  
            
    else:
        #run all training steps
        model.fit(
            xTrain,
            yTrain,
            validation_split=0.05,
            batch_size=512,
            epochs=numEpochs,
            verbose=2)
        
        #calculate the model's prediction on the test data
        predicted = model.predict(xTest)
        predicted = np.reshape(predicted, (predicted.size,))
        
        #set x axis
        length = len(predicted)
        x_s = np.linspace(0, length-1, num=length, endpoint=True)
        
        #plot the test data against the prediction
        plt.clf()
        plt.plot(x_s, predicted, "-" , x_s, yTest, "-")
        plt.legend(labels=["predicted", "actual"])
        plt.show()
        
    #return trained model
    return model


sampleLength = 120
xTrain, yTrain, xTest, yTest = read_Data(sampleLength)
model = RNN([1, sampleLength, sampleLength, 1])
trainedModel = train_Net(xTrain, yTrain, xTest, yTest, model)
