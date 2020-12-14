# autoECG
Model and transfer learning model based on Automatic diagnosis of the 12-lead ECG using a deep neural network, Ribeiro, et al.
# 1. Background
  We tried to build an advanced model according to the paper Automatic diagnosis of the 12-lead ECG using a deep neural network. This paper uses CNN and RNN models to do
the classification. The dataset in the original paper is very similar to what we have. The original data, acquired from the Telehealth Network of Minas Gerais (TNMG) [1]
, also contains 12 leads. With each lead, the duration of the ECG recording is between 7s to 10s, with frequency ranging from 300 HZ to 600 HZ. To make input of the same size, the data was zero-padded and each lead has 4096 numbers. The dataset size is very large, containing 2,322,513 ECG records from 1,676,384 patients. The training set contains 98% of the data and the validation set contains 2% of the data.
In the paper, the DNN architecture is structured as follows:  

![alt text](/modelstructure.jpg)   


  The network consists of a convolutional layer followed by four residual blocks with two convolutional layers per block. Max Pooling and 1*1 convolutional network are included in the skip connections to make the dimensions match those output from the main branch. The output of the last layer was then fed into a fully connected dense layer with sigmoid
activation function. The result in this paper is attractive. The minimum F1 score achieved is 0.8970 (for 1dAVb) and the maximum F1 score is 1.00 (for LBBB). For all types of disease, the DNN prediction accuracy outperforms humans.  

![alt text](/modelresult.jpg)  


# 2. Modified Structure
  The original model consists of four residual blocks and the overall structure is shown in the following graph. The output has six entries, each containing a probability between 0 and 1, and can be understood as the probability of a given abnormality to be present. The abnormalities it predicts are: 1st degree AV block(1dAVb), right bundle branch block (RBBB), left bundle branch block (LBBB), sinus bradycardia (SB), atrial fibrillation (AF), sinus tachycardia (ST). The abnormalities are not mutually exclusive, so the probabilities do not necessarily sum to one.  
  While in our cases, we only need to determine a certain ECG data whether is normal or abnormal (also the two probabilities sum to one), so our new output should be a single probability. In order to fulfill this goal, we modified the final dense layer by changing the unit number from six to one while still using the ‘sigmoid’ activation function. Moreover, we used the binary cross entropy function as the new loss function.  
  The coding file of the modified model is in the autoECG-GPU.ipynb file under the root directory.

  
  
  
  
# 3. Data Generator



# 4. Transfer Learning
  Because the performance of the model was pretty good, and the training set used by the paper was super large, we assumed the original model learned enough information about how to diagnose an ECG, which suggested using transfer learning and fine-tuning would be a promising and practical step to take. We basically froze the first 5 layers making them untrainable and freed the rest layers’s weights (the pretrained model in total had 50 layers) and we built our own neural network followed by the 45th layer of the pretrained one. We added three 1d convolutional layers and three dense layers followed by the 45th layer. The structure can be viewed as follow:  
 
![alt text](/transferlearning.png)    
  
  The coding file of Transfer Learning is in the Transfer.ipynb file under the root directory.

[1] Ribeiro, A.H., Ribeiro, M.H., Paixão, G.M.M. et al. Automatic diagnosis of the 12-lead
ECG using a deep neural network.
Nat Commun 11, 1760 (2020). https://doi.org/10.1038/s41467-020-15432-4
