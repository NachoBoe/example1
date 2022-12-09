import numpy as np

class model_free_primal_dual:

    def __init__(self,phi,f,alpha,epochs=1000,lr=0.1,batch_size=1):
        self.phi = phi
        self.f = f
        self.alpha = alpha
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self,training_data):
        theta = self.phi.params
        for i in range(self.epochs):
            theta_sampled, h_sampled = self.sample(training_data,theta.shape,self.batch_size)
            grad_f = self.calculate_grad(theta, theta_sampled, h_sampled,self.alpha)
            theta = self.act_theta(theta,grad_f,self.lr)
            if (i%500==0):
                print(i)
                print(theta)

        self.phi.params = theta
        return self.phi


    def sample(self,training_data,shape_theta,batch_size):
        h_sampled = []
        theta_sampled = []
        for i in range(batch_size):
            h_sampled.append(training_data[np.random.randint(0,high=training_data.shape[0])])
            th = np.random.random(size=(shape_theta))
            theta_sampled.append(th/np.linalg.norm(th))
        return theta_sampled, h_sampled 
    
    def calculate_grad(self,theta, theta_sampled, h_sampled,alpha):
        grad_f = 0
        for i in range(len(h_sampled)):
            t_sample = theta_sampled[i]
            h_sample = h_sampled[i]
            self.phi.params = theta + alpha*t_sample
            f1 = self.f(self.phi(h_sample), h_sample)
            self.phi.params = theta
            f2 = self.f(self.phi(h_sample), h_sample)
            grad_f += (f1 - f2)/alpha * t_sample
        grad_f /= len(h_sampled)
        return grad_f

    def act_theta(self,theta,grad_f,lr):
        theta_new = theta - lr*grad_f
        return theta_new
