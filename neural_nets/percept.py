
def main():
    x0 = 0.9
    x = [[0,0,x0],[0,1,x0],[1,0,x0],[1,1,x0]]
    D = [0,1,1,0]
    backpropagation(x,0.2,0.3,-0.9,0.1,D)

def backpropagation(x,w1, w2, w0, eta, D):
    itera = 0
    while(True): 
        suma,d_acum = 0,0
        for i in range(0, 4):
            x1,x2,x0 = x[i] 
            suma = (w0 * x0) + (w1 * x1) + (w2 * x2)
            y = 1 if suma > 0 else 0
            d = D[i] - y
            d_acum += abs(d) 
            # Gradient
            g_d = eta * d

            # Apply gradient
            g_d_x1 = g_d * x1
            g_d_x2 = g_d * x2
            g_d_x0 = g_d * x0
            
            # Recalculate weights
            w1 = w1 + g_d_x1
            w2 = w2 + g_d_x2
            w0 = w0 + g_d_x0
        if(d_acum == 0):
            print(f'The work is done with w values of w0-{w0}, w1-{w1}, w2-{w2} in {itera} iterations')
            break;
        itera =+ 1







if __name__ == '__main__':
    main()