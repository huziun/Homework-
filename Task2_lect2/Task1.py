import random
#S=pi*r^2

def Coordinate():
    count_in_circle = 0;
    all_Points = 100000;
    for i in range(all_Points):
        x = random.random();
        y = random.random();
        if x * x + y * y <= 1:
            count_in_circle += 1;

    return count_in_circle, all_Points;

def FindPi(count_in_circle, all_Points):
    pi = 4 * (count_in_circle / all_Points);
    return pi;

def Start():
    count_in_circle, all_Points = Coordinate();
    print("All points: ", all_Points);
    print("Points in circle: ", count_in_circle);
    print("Pi = ", FindPi(count_in_circle, all_Points));
    return 0;

