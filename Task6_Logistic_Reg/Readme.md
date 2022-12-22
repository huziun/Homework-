# CatNotCat


X_train.shape=  
(209, 64, 64, 3)

X_test.shape=  (50, 64, 64, 3)

Y_train.shape=  (209, 1)

Y_test.shape=  (50, 1)

Number of training examples: m_train = 209

Number of testing examples: m_test = 50

Height/Width of each image: num_px = 64

Each image is of size: (64, 64, 3)

y = [1], it's a 'cat' picture.

y = [0], it's a 'non-cat' picture.

Label 1 count: 72

Label 0 count: 137

train_set_x_flatten shape: (209, 12288)

test_set_x_flatten shape: (50, 12288)

sanity check after reshaping: [17 31 56 22 33]

sigmoid([0, 2]) = [0.5        0.88079708]

w = [[0. 0.]]

b = 0

dJ_dw = [[0.99845601 2.39507239]]

dJ_db = 0.001455578136784208

cost = 5.801545319394553

w = [[0.19033591 0.12259159]]

b = 1.9253598300845747

dw = [[0.67752042 1.41625495]]

db = 0.21919450454067652

predictions = 
    
    [[1]
     [1]
     [0]]

Cost after iteration 0: 0.693147

Cost after iteration 100: 0.571977

Cost after iteration 200: 0.528957

Cost after iteration 300: 0.497042

Cost after iteration 400: 0.471093

Cost after iteration 500: 0.449030

Cost after iteration 600: 0.429769

Cost after iteration 700: 0.412656

Cost after iteration 800: 0.397258

Cost after iteration 900: 0.383269

Cost after iteration 1000: 0.370460

Cost after iteration 1100: 0.358659

Cost after iteration 1200: 0.347727

Cost after iteration 1300: 0.337555

Cost after iteration 1400: 0.328052

Cost after iteration 1500: 0.319144

Cost after iteration 1600: 0.310767

Cost after iteration 1700: 0.302869

Cost after iteration 1800: 0.295403

Cost after iteration 1900: 0.288331

Cost after iteration 2000: 0.281618

Cost after iteration 2100: 0.275233

Cost after iteration 2200: 0.269152

Cost after iteration 2300: 0.263349

Cost after iteration 2400: 0.257804

Cost after iteration 2500: 0.252499

Cost after iteration 2600: 0.247417


Cost after iteration 2700: 0.242542

Cost after iteration 2800: 0.237862

Cost after iteration 2900: 0.233364

train accuracy= 96.172%

test accuracy= 74.000%

y_predicted = 1 (true label = 1) , you predicted that it is a cat picture.

![](https://cdn.discordapp.com/attachments/753602758023839817/1055214670874693742/image.png)
![](https://cdn.discordapp.com/attachments/753602758023839817/1055214796968038400/image.png)
![](https://cdn.discordapp.com/attachments/753602758023839817/1055214911359295589/image.png)
![](https://media.discordapp.net/attachments/753602758023839817/1055212622477275267/image.png?width=779&height=663)


# Cancer Dataset

X_cancer.shape= (569, 30)

Breast cancer dataset

Accuracy of Logistic regression classifier on training set: 0.96

Accuracy of Logistic regression classifier on test set: 0.95
![](https://cdn.discordapp.com/attachments/753602758023839817/1055212623223869551/image.png)
