# Diabetes model using PyTorch
# Uses the data file:  diabetes.csv
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# This code assumes that the data file is in the same dir as this python file.
data_file_name = 'data/diabetes.csv'
model_saved_name = 'model/PytorchDiabetesModel.pt'


df = pd.read_csv(data_file_name)
X = df.drop('Outcome' , axis = 1) #independent Feature
y = df['Outcome'] #dependent Feature

#note data is not normalized in this example
#train the model

X_train,X_test,y_train,y_test = train_test_split(X,y , test_size =0.2,random_state=0)

# Creating Tensors (multidimensional matrix) x-input data  y-output data
X_train=torch.FloatTensor(X_train.values)
X_test=torch.FloatTensor(X_test.values)
y_train=torch.LongTensor(y_train.values)
y_test=torch.LongTensor(y_test.values)

#Create the Model
class ANN_model(nn.Module):
    def __init__(self,input_features=8,hidden1=20, hidden2=10,out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features,hidden1)
        self.f_connected2 = nn.Linear(hidden1,hidden2)
        self.out = nn.Linear(hidden2,out_features)
        
    def forward(self,x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x

    def save(self, model_path):
        torch.save(model.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    
torch.manual_seed(20)
model = ANN_model()

# Backward Propagation - loss and optimizer
loss_function = nn.CrossEntropyLoss()   #CrossEntropyLoss also used in Tensorflow
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)  #note Tensorflow also uses Adam

epochs=500
final_losses=[]
for i in range(epochs):
    i= i+1
    y_pred=model.forward(X_train)
    loss=loss_function(y_pred,y_train)
    final_losses.append(loss)
    #if i % 10 == 1:
    #    print("Epoch number: {} and the loss : {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    


#Accuracy - comparing the results from the test data

predictions = []
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_pred = model(data)
        #predictions.append(y_pred.argmax().item())
        predictions.append(y_pred.argmax())
        
score = accuracy_score(y_test , predictions)  # Simply calculates number of hits / length of y_test
#print(score)

# save model
model.save(model_saved_name)


def predict(dataset):
    predict_data = dataset
    predict_data_tensor = torch.tensor(predict_data)      #Convert input array to tensor
    prediction_value    = ann_model(predict_data_tensor)  # This is a tensor

    # Dict for textual display of prediction
    outcomes            = {0: 'No diabetes',1:'Diabetes Predicted'}

    # From the prediction tensor, get the index of the max value ( Either 0 or 1)
    prediction_index   = prediction_value.argmax().item()
    prediction = outcomes[prediction_index]
    #return(outcomes[prediction_index])
    #return prediction
    return {'prediction': prediction}

