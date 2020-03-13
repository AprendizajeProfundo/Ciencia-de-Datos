import numpy as onp

class optim():
    ### learning rate schedules
    def make_schedule(self,scalar_or_schedule):
        #if callable(scalar_or_schedule):
        if scalar_or_schedule == 'exponential_decay':
            return self.exponential_decay
        elif scalar_or_schedule =='inverse_time_decay':
            return self.inverse_time_decay
        elif scalar_or_schedule =='polynomial_decay':
            return self.polynomial_decay
        elif scalar_or_schedule =='piecewise_constant':
            return self.piecewise_constant
        #        
        elif onp.ndim(scalar_or_schedule) == 0:
            return self.constant(scalar_or_schedule)
        else:
            raise TypeError(type(scalar_or_schedule))
    
    def constant(self,step_size):
        def schedule(i):
            return step_size
        return schedule

    def exponential_decay(self,step_size, decay_steps, decay_rate):
        def schedule(i):
            return step_size * decay_rate ** (i / decay_steps)
        return schedule

    def inverse_time_decay(self,step_size, decay_steps, decay_rate, staircase=False):
        if staircase:
            def schedule(i):
                return step_size / (1 + decay_rate * onp.floor(i / decay_steps))
        else:
            def schedule(i):
                return step_size / (1 + decay_rate * i / decay_steps)
            return schedule


    def polynomial_decay(self, step_size, decay_steps, final_step_size, power=1.0):
        def schedule(step_num):
            step_num = onp.minimum(step_num, decay_steps)
            step_mult = (1 - step_num / decay_steps) ** power
            return step_mult * (step_size - final_step_size) + final_step_size
        return schedule


    def piecewise_constant(self, boundaries, values):
        boundaries = onp.array(boundaries)
        values = onp.array(values)
        if not boundaries.ndim == values.ndim == 1:
            raise ValueError("boundaries and values must be sequences")
        if not boundaries.shape[0] == values.shape[0] - 1:
            raise ValueError("boundaries length must be one longer than values length")
            
        def schedule(i):
            return values[onp.sum(i > boundaries)]
        return schedule


    ## main methods of the class
    def __init__(self,initial_param, step_size = 1):
        self.param = initial_param
        self.step_size = self.make_schedule(step_size)
        self.state = [self.param, self.step_size]
        
    def set_params(self):
        # to be overloaded
        pass
    
    def update(self,i, gradient):
        # to be overload
        pass
        
    def get_params(self):
        return self.param
    
    def get_state(self):
        #to be overload
        return self.state
       
     
    
    
    
    
# end class optimizer
    
    
    

class adam(optim):
    def __init__(self,initial_param,alpha=0.001):
        optim.__init__(self,initial_param, alpha)
        self.alpha = self.step_size
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.eps =1.0e-8
        #self.param = initial_param
        self.m = onp.zeros_like(initial_param)
        self.v = onp.zeros_like(initial_param)
        
    def set_params(self, alpha=0.001,beta_1=0.9, beta_2=0.999, eps =1.0e-8):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
    
    def update(self,i, gradient):
        self.m = (1 - self.beta_1) * gradient + self.beta_1 * self.m  # First  moment estimate.
        self.v = (1 - self.beta_2) * (gradient ** 2) + self.beta_2 * self.v  # Second moment estimate.
        mhat = self.m / (1 - self.beta_1 ** (i + 1))  # Bias correction.
        vhat = self.v / (1 - self.beta_2 ** (i + 1))
        self.param = self.param - self.alpha * mhat / (onp.sqrt(vhat) + self.eps)
                
        
    def get_state(self):
        self.state = [self.param,self.m,self.v, self.step_size]
        return self.state
    
# end class adam