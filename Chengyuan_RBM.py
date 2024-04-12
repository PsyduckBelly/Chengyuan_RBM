from __future__ import print_function
import numpy as np

class RBM:
    
    def __init__(self, num_visible, num_hidden):
        # Initialization
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.debug_print = True
        
        # Create a random number generator for weight initialization
        # TODO: probably here for the random seed
        np_rng = np.random.RandomState(317)
        
        # Normalize initiation to prevent symmetry
        #  if weights are initialized identically, 
        # neurons in the same layer may learn the same features, 
        # hindering diverse and effective learning.
        self.weights = np.asarray(np_rng.uniform(
            low = -0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high = 0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size = (num_visible, num_hidden)))
        
        # Add bias units to weights (first row and first column)
        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)
        
    def train(self, data, max_iterations = 5000, learning_rate = 0.01):
        # Initialization
        num_examples = data.shape[0]
        # Add bias units to data (first column)
        data = np.insert(data, 0, 1, axis = 1)

        # Training loop
        for i in range(max_iterations):
            # Compute activations of hidden units (positive phase)
            pos_hid_activations = np.dot(data, self.weights)
            # Compute probabilities of activating hidden units
            pos_hid_probs = self.energy(pos_hid_activations)
            # Bias units are always activated
            pos_hid_probs[:,0] = 1
            # Sample hidden states based on probabilities
            pos_hid_states = pos_hid_probs > np.random.rand(num_examples, self.num_hidden + 1)
            
            # correlations between the visible and hidden units given the current state of the network.
            pos_associations = np.dot(data.T, pos_hid_probs)

            # Compute activations of visible units (negative phase)
            neg_vis_activations = np.dot(pos_hid_states, self.weights.T)
            # Compute probabilities of activating visible units
            neg_vis_probs = self.energy(neg_vis_activations)
            # Bias units are always activated
            neg_vis_probs[:,0] = 1
            # Compute activations of hidden units based on visible probabilities
            neg_hid_activations = np.dot(neg_vis_probs, self.weights)
            # Compute probabilities of activating hidden units
            neg_hid_probs = self.energy(neg_hid_activations)
            
            # correlations the network predicts based on its current understanding (weights)
            neg_associations = np.dot(neg_vis_probs.T, neg_hid_probs)
            
            # updates the weights of the RBM in a direction that reduces the reconstruction error 
            # (difference between positive and negative associations), 
            # escaled by the learning rate and averaged over all training examples.
            self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)
            
            # how well the RBM is able to reconstruct the input data after a forward and backward pass
            error = np.sum((data - neg_vis_probs) ** 2)
            if self.debug_print and i % 100 == 0:
                print(f"Iteration: {i}, Error: {error}")
    
    # Energy function calculation
    def energy(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    ## Return Both hidden probability and hidden states
    # Alternating steps from visible layer to hidden layer
    def run_visible(self, data):
        num_examples = data.shape[0]
        # Initialize hidden states with bias unit activated
        hidden_states = np.ones((num_examples, self.num_hidden + 1))
        # Add bias units to data
        data = np.insert(data, 0, 1, axis = 1)

        # Compute hidden activations and probabilities
        hidden_activations = np.dot(data, self.weights)
        # Compute probabilities of activating hidden units
        hidden_probs = self.energy(hidden_activations)
        # Sample hidden states based on probabilities
        hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)

        return hidden_states[:,1:]
    
    # Alternating steps from hidden layer to visible layer
    def run_hidden(self, data):
        num_examples = data.shape[0]
        # Initialize visible states with bias unit activated
        visible_states = np.ones((num_examples, self.num_visible + 1))
        # Add bias units to data
        data = np.insert(data, 0, 1, axis = 1)

        # Compute visible activations and probabilities
        visible_activations = np.dot(data, self.weights.T)
        # Compute probabilities of activating visible units
        visible_probs = self.energy(visible_activations)
        # Sample visible states based on probabilities
        visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)

        return visible_states[:,1:]
    
    # Gibbs sampling step Randomly assign and do the gibbs sampling for the RBM
    def gibbs(self, num_samples):
        samples = np.ones((num_samples, self.num_visible + 1))
        samples[0, 1:] = np.random.rand(self.num_visible)

        for i in range(1, num_samples):
            visible = samples[i-1,:]

            # Calculate the activations of the hidden units.
            hidden_activations = np.dot(visible, self.weights)      
            # Calculate the probabilities of turning the hidden units on.
            hidden_probs = self.energy(hidden_activations)
            # Turn the hidden units on with their specified probabilities.
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            # Always fix the bias unit to 1.
            hidden_states[0] = 1

            # Recalculate the probabilities that the visible units are on.
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self.energy(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i,:] = visible_states

        # Ignore the bias units (the first column), since they're always set to 1.
        return samples[:,1:]        
