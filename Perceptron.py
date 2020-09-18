import numpy as np

class Perceptron(object):

    def __init__(self, learn_rate, epochs):
        '''
        Learning rate --> small number between 0 to 1
        Epochs --> number of iterations perceptron will train on data
        Weights --> vector consisting of small random numbers (coefficients) between 0 and 1
          - W vector size: # of dimensions + 1
          - W = [w0, w1, w2]
        '''
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.weights = np.random.rand(3)

    def train(self, x_input, correct_label):
        '''
        x_input --> input vector containing [x1, x2] pairs
          - X = [1, x1, x2]
          - size = # of dimensions + 1
          - The first item in the vector is the constant term; we will need to add this constant later
        correct_label --> the label associated with the x vector
          - is either 1 or -1
          - this is what we will be comparing perceptron's output to
        errors --> keeps track of how many times our perceptron made the wrong prediction in each epoch
        '''
        self.errors = []
        for _ in range(self.epochs):
            error_count = 0
            for xi, target in zip(x_input, correct_label):
                ''' 
                Here, we update the weights according to the weight update rule:
                    weights = weights + change_in_weights
                and the perceptron learning rule:
                    change_in_weights = learn_rate * (target - prediction) * xi 
                '''
                prediction = self.predict(xi)
                update = self.learn_rate * (target - prediction)
                w0 = self.weights[0]
                w1_w2 = self.weights[1:]

                w0 += update                        # Really, it's w0 += update * 1
                w1_w2 += update * xi

                ''' 
                If our update is anything other than 0.0, then our algorithm made a 
                mistake; add it to our cost list
                '''
                if update != 0.0:
                    error_count += 1

            self.errors.append(error_count)
        return self

    def dot_product(self, x_input):
        # print("Self.w[1:] "+str(self.weights[1:]))
        # print("Self.w[0] "+str(self.weights[0]))
        '''
        x_input contains [x1, x2], and weights contains [w0, w1, w2];
        we need to add the constant '1' to x_input to do the dot product:
        w * x = w0, w1, w2 * [1, x1, x2]
              = (w0)(1) + (w1)(x1) + (w2)(x2)
              = w0 + (w1)(x1) + (w2)(x2)
        '''
        w0 = self.weights[0]
        w1_w2 = self.weights[1:]

        ''' Calculating (w1)(x1) + (w2)(x2): '''
        dot_prod = np.dot(x_input, w1_w2)

        ''' Calculating  w0 + (w1)(x1) + (w2)(x2): '''
        dot_prod = w0 + dot_prod
        return dot_prod

    def predict(self, x_input):
        ''' This is where we use our step function '''
        prediction = self.dot_product(x_input)
        if prediction >= 0:
            return 1
        else:
            return -1

