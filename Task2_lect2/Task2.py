import random;

def GetNumbers():
    a = random.randint(2, 99);
    b = random.randint(2, 99);
    #print("a:", a, "b:", b);

    return a, b;

def GetProd_and_Sum(a, b):
    prod = a * b;
    sum = a + b;

    return prod, sum;

def UnicProd(prod):
    numberForprod = []

    for numberOne in range(2, 100):

        if prod % numberOne == 0:
            numberTwo = prod/numberOne;

            if numberTwo < 100 and numberOne not in list(numberForprod):
                numberForprod.append(numberOne);
                numberForprod.append(numberTwo);

    if len(numberForprod) == 2:

        return False;

    return numberForprod;

def UnicSum(sum, numberForprod):

    numberForsum = [];
    i = 0;

    while(i < len(numberForprod)):

        numberOne = numberForprod[i];
        numberTwo = numberForprod[i+1];

        if sum == numberOne + numberTwo:
            print("First number:", numberOne, "Second number:", numberTwo);

        i = i + 2;
        numberForsum.append(numberOne);
        numberForsum.append(numberTwo);

    return numberForsum;

def Check(numberForsum, prod):
    return numberForsum[0] * numberForsum[1] == prod

def Start():
    a, b = GetNumbers();
    prod, sum = GetProd_and_Sum(a, b);

    numberForprod = UnicProd(prod);

    if numberForprod == False:
        Start()
    else:
        print("Prod: ", prod, "Sum: ", sum);
        print(numberForprod);

        numberForsum = UnicSum(sum, numberForprod);
        print(Check(numberForsum, prod));

