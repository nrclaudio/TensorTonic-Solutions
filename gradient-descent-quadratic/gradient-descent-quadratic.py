def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    def func_d(a,b,c,x):
        return 2*a*x + b
    x = x0
    for _ in range(steps):
        x = x - lr * func_d(a,b,c,x)
    return x