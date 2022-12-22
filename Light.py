Not_Light = [];

for i in range(1, 1001):
    iteration = 0;
    print("I", i)

    number = i
    for j in range(1, number+1):
        if number % j == 0:
            iteration +=1;

    if iteration % 2 != 0:
        Not_Light.append(i);
    print("Iteration", iteration);

print(Not_Light);
print(len(Not_Light));