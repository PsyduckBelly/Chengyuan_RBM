class RBM:  # Define the class RBM
        def __init__(self, visible_dim, hidden_dim, learning_rate, number_of_iterations):
                # Constructor method to initialize the RBM with given parameters
                self.num_iter = number_of_iterations  # Number of training iterations
                # Initialize visible biases with random values between 0 and 1
                self.visible_biases = np.random.uniform(low=0, high=1, size=(1, visible_dim))
                # Initialize hidden biases with random values between 0 and 1
                self.hidden_biases = np.random.uniform(low=0, high=1, size=(1, hidden_dim))
                # Initialize weights with small random values from a normal distribution
                self.weights = np.random.normal(loc=0.0, scale=0.01, size=(visible_dim, hidden_dim))
                self.learning_rate = learning_rate  # Learning rate for weight updates
 
        def sample_prob(self, probs):
                # Given a probability, sample a binary outcome (0 or 1)
                return (probs > np.random.random(probs.shape)).astype(np.float32)

        def sigmoid(self, x):
 
                return 1 / (1 + np.exp(-x))
 
        def step(self, v0):
                # Perform one step of Gibbs sampling
                # Ensure input v0 is 2D for matrix operations
                v0 = v0.reshape(1, -1) if len(v0.shape) == 1 else v0

                # Compute hidden unit probabilities and sample binary states
                h0_prob = self.sigmoid(np.dot(v0, self.weights) + self.hidden_biases)
                h0 = self.sample_prob(h0_prob)
                # Compute visible unit probabilities for reconstruction and sample binary states
                v1_prob = self.sigmoid(np.dot(h0, self.weights.T) + self.visible_biases)
                v1 = self.sample_prob(v1_prob)
                # Again compute hidden unit probabilities for the reconstructed visible states
                h1_prob = self.sigmoid(np.dot(v1, self.weights) + self.hidden_biases)
                h1 = self.sample_prob(h1_prob)
                return v0, h0, v1, h1  # Return the initial, sampled, reconstructed, and resampled states

        def update_weights(self, v0):
                # Update the weights based on the difference between the sampled and resampled states
                v0, h0, v1, h1 = self.step(v0)  # Perform Gibbs sampling step
                # Calculate the gradient for weight update
                positive_grad = np.dot(v0.T, h0) - np.dot(v1.T, h1)
                # Update weights with gradient scaled by learning rate and averaged over batch size
                dw = self.learning_rate * (positive_grad / v0.shape[0])
                self.weights += dw
                # Update visible and hidden biases based on the difference between original and reconstructed
                db_v = self.learning_rate * np.mean(v0 - v1, axis=0)
                self.visible_biases += db_v.reshape(1, -1)
                db_h = self.learning_rate * np.mean(h0 - h1, axis=0)
                self.hidden_biases += db_h.reshape(1, -1)

        def train(self, data):
                # Train the RBM over the specified number of iterations
                for epoch in range(self.num_iter):
                        np.random.shuffle(data)  # Shuffle the training data in each epoch
                        for v0 in data:  # Update weights for each sample in the dataset
