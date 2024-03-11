def example(x):
    if x > 0:        # Decision 1
        x += 1
    elif x < -10:    # Decision 2
        x -= 1
    else:
        x = 0
    
    for i in range(x):  # Decision 3
        print(i)

    return x