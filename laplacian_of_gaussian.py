import math as m

def gaussian(x, y, s, u1, u2):
    """
    return the value of a gaussian function with the specified parameters
    x : (float) the x value to evaluate at
    y : (float) the y value to evaluate at
    s : (float) the standard deviation of the gaussian
    u1 : (float) the x mean of the gaussian
    u2 : (float) the y mean of the gaussian
    """
    
    output = m.e**(-((x - u1)**2 + (y - u2)**2)/(2*(s**2)))
    
    output /= ((2*m.pi)**2 * (s**4))**(1/2)
    
    return output


def laplacian(x, y, s, u1, u2):
    """
    return the value of a laplacian function of the gaussian with the specified parameters
    x : (float) the x value to evaluate at
    y : (float) the y value to evaluate at
    s : (float) the standard deviation of the gaussian
    u1 : (float) the x mean of the gaussian
    u2 : (float) the y mean of the gaussian
    """  
    
    output = gaussian(x, y, s, u1, u2)
    
    output *= (((x - u1)**2) + ((y - u2)**2))/(s**2) - 2
    
    output /= s**2
    
    return output


def laplacian_of_gaussian_kernel(side_length, domain, s, u1, u2,
                                 zero_center=True, max_normalize=True,
                                 absolute_l1_normalize=False):
    """
    this function returns the laplacian kernel of a guassian
    function with the parameters specified in the input
    the output kernel will be normalized to have a mean of zero
    and a max of 1
    
    side_length : (int) the number of rows and columns in the kernel
                  THIS SHOULD PROBABLY BE AN ODD NUMBER
    domain : (float) the x and y distance for the values to calculated over
    s : (float) the standard deviation of the gaussian
    u1 : (float) the x mean of the gaussian
    u2 : (float) the y mean of the gaussian
    zero_center : (boolean) centers the kernel around 0 meaning the sum of all the values
                  will be 1
    
    max_normalize : (boolean) renomalize the values so they are scaled to be less than 
                     than the max and the max will be scaled to 1 
                     
    absolute_l1_normalize : (boolean) rescale the values so they are divided by the sum
                             of their absolute values and their absolute values will sum
                             to 1 
                             
    NOTE : max_normalize and absolute_l1_normalize should probably not be True at the same time
    
    returns a list of lists of floats
    eg : [[0.075, 0.175, 0.075],
          [0.175, -1.0, 0.175],
          [0.075, 0.175, 0.075]]

    eg : laplacian_of_gaussian_kernel(5, 2, 1, 0, 0,
                                 zero_center=True, max_normalize=True,
                                 absolute_l1_normalize=False)
                                 
                                 creates the following kernel
                                 
    [[0.07502461545892744, 0.14450280626002782, 0.15694292954133218, 0.14450280626002782, 0.07502461545892744],
    [0.14450280626002782, 0.019031951727176533, -0.29000510924749184, 0.019031951727176533, 0.14450280626002782],
    [0.15694292954133218, -0.29000510924749184, -1.0, -0.29000510924749184, 0.15694292954133218],
    [0.14450280626002782, 0.019031951727176533, -0.29000510924749184, 0.019031951727176533, 0.14450280626002782],
    [0.07502461545892744, 0.14450280626002782, 0.15694292954133218, 0.14450280626002782, 0.07502461545892744]]    
    """
    
    #calculate the laplacian values for the kernel and
    #keep track of other values
    kernel = [[None for b in range(side_length)] for a in range(side_length)]
    mean = 0
    
    #fill the kernel with the laplacian values
    for i in range(side_length):
        y = -domain*(2*(i/(side_length - 1)) - 1)
        for j in range(side_length):
            
            x = domain*(2*(j/(side_length - 1)) - 1)
            
            L = laplacian(x, y, s, u1, u2)
            
            kernel[i][j] = L
            
            mean += L

    mean /= side_length**2
    
    #normalize the kernel so the values add up to zero
    if zero_center:
        for i in range(side_length):
            for j in range(side_length):
                kernel[i][j] -= mean
    
    #normalize based on the absolute maximum value 
    if max_normalize:
        max_value = 0
        for i in range(side_length):
            for j in range(side_length):
                max_value = max(max_value, abs(kernel[i][j]))
                
        for i in range(side_length):
            for j in range(side_length):
                kernel[i][j] /= max_value        
    
    #normalize based on the sum of absolute values
    if absolute_l1_normalize:
        total = 0
        for i in range(side_length):
            for j in range(side_length):
                total += abs(kernel[i][j])
                
        for i in range(side_length):
            for j in range(side_length):
                kernel[i][j] /= total        
              
    return kernel


def view():
    #plot the kernel
    import matplotlib.pyplot as plt
    
    kernel = laplacian_of_gaussian_kernel(201, 8, 2, 0, 0,
                                          zero_center=True, max_normalize=True,
                                          absolute_l1_normalize=False)
 
    plt.imshow(kernel)
    plt.show()
    
    
def find_reference():
    """
    atempting to find parameters of original
    kernel, which was
    [[0.066, 0.184, 0.066],
    [0.184, -1.0, 0.184],
    [0.066, 0.184, 0.066]]
    the sum of values in the kernel is zero
    which implies zero centering
    
    
    
    closest so far
    domain : 1.422811355677839, sigma : 0.6944471735867934
    with no normalization or zero centering
    and only accounting for the edge values
    [[0.066, 0.184, 0.066],
    [0.184, -1.369, 0.184],
    [0.066, 0.184, 0.066]]
    
    
    domain : 1.6689344172086042, sigma : 0.7424711855927963
    with zero centering and no normalization
    but accounting for all values
    [[0.075, 0.175, 0.075],
    [0.175, -1.0, 0.175],
    [0.075, 0.175, 0.075]]
    """
    import matplotlib.pyplot as plt
    
    values = ""
    new_kernel = None

    min_diff = 100000
    
    search = 2000
    for i in range(search):
        for j in range(search):
            
            kernel_domain = 0.0001 + 4*(j/(search-1))
            kernel_sigma = 0.0001 + 4*(i/(search-1))
            kernel = laplacian_of_gaussian_kernel(3, kernel_domain, kernel_sigma, 0, 0,
                                                  zero_center=True, max_normalize=False,
                                              absolute_l1_normalize=False)
            #current_diff = abs(0.066 - kernel[0][0]) + abs(0.184 - kernel[0][1])# + abs(1 + kernel[1][1])
            current_diff = abs(0.066 - kernel[0][0]) + abs(0.184 - kernel[0][1]) + abs(1 + kernel[1][1])
            
            #print(current_diff)
            
            if current_diff < min_diff:
                values = "domain : {}, sigma : {}".format(kernel_domain, kernel_sigma)
                min_diff = current_diff
                new_kernel = kernel  
    print(values)
    plt.imshow(new_kernel)
    plt.show() 



