import radon.complexity

def simple_function():
    print("Hello World")

def complex_function():
    for i in range(10):
        for j in range(20):
            print(i, j)
    if x > 10:
        print("x is large")
    else:
        print("x is small")

simple_complexity = radon.complexity.cc_visit(simple_function)
complex_complexity = radon.complexity.cc_visit(complex_function)

print("Simple function complexity:", simple_complexity)  
print("Complex function complexity:", complex_complexity)
