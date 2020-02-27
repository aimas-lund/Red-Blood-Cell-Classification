import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from lib.data_handling import images_to_array_list


class CNN:
    def __init__(self):
        self.training_data = None
        self.validation_data = None
        self.testing_data = None
        self.data_pre_processing = None
        self.data_augmentation = None
        self.network = None
        self.model = None
        self.layer_overview = []
        self.layer_dict = self.init_layer_dict()

    @staticmethod
    def init_layer_dict():
        keys = list(range(3))
        layer_type = ["Convolutional Layer", "Fully Connected Layer", "Max Pooling Layer"]

        return dict(zip(keys, layer_type))

    def pre_processing(self):  # normalizes the training data
        image_prep = ImagePreprocessing()
        image_prep.add_featurewise_zero_center()
        image_prep.add_featurewise_stdnorm()

        self.data_pre_processing = image_prep

    def augmentation(self, max_angle=5., sigma_max=3.,
                     flip_left_right=True,
                     random_rotation=True,
                     random_blur=True):

        if any([flip_left_right, random_blur, random_rotation]):
            pass
        else:
            raise ValueError

        try:
            image_aug = ImageAugmentation()
            if flip_left_right:
                image_aug.add_random_flip_leftright()  # adds left- and right flipped images to the training data

            if random_rotation:
                image_aug.add_random_rotation(
                    max_angle=max_angle)  # rotates random training data by a specified angle (
            # degrees)

            if random_blur:
                image_aug.add_random_blur(sigma_max=sigma_max)  # blurs random training data by a specified sigma
            self.data_augmentation = image_aug

        except ValueError:
            print("No augmentation selected!")

    def get_training_data(self, path):
        self.training_data = images_to_array_list(path)

    def get_validation_data(self, path):
        self.validation_data = images_to_array_list(path)

    def get_testing_data(self, path):
        self.testing_data = images_to_array_list(path)

    def init_network(self, shape=None, name='InputData'):
        """
        Initializes the network attribute for the CNN class
        :param shape: (list(int)) A list of integers specifying batch size, height-, width and depth of the input array,
         respectively.
        :param name: (str) Given name to the network. Will be called 'InputData' by default.
        :return: An initialized network attribute for the CNN class.
        """
        if shape is None:
            shape = [None, 150, 800, 3]

        self.network = input_data(shape=shape,
                                  data_preprocessing=self.data_pre_processing,
                                  data_augmentation=self.data_augmentation,
                                  name=name)
        self.layer_overview = [shape]

    def network_add_cl(self, nb_filter, filter_size,
                       activation='relu',
                       padding='same'):
        """
        Adds a convolutional layer to the specified network.
        :param nb_filter: (int) Specifying the number of convolutional filters
        :param filter_size: (int or list(int)) Specifying the size of the filters
        :param activation: (str) Specifying which type of activation function should be applied to the weights in the
         given layer. Default is set to 'rely' (Rectified Linear Unit).
        :param padding: (str) Specifying padding algorithm ('same' or 'valid')
        :return: An added convolutional layer to the network attribute in the CNN class.
        """
        if self.network is None:
            print("Network not initialized. Initialize network first using the 'init_network' method!")
        else:
            self.network = conv_2d(self.network, nb_filter, filter_size,
                                   activation=activation, padding=padding)
            self.layer_overview = self.layer_overview.append(
                [0, nb_filter, filter_size, activation, padding]
            )  # appends a list of specifying values of the convolutional layer to the overview list

    def network_add_fcl(self, n_units, activation='relu'):
        """
        Adds a fully connected layer to the specified network.
        :param n_units: (int) Number of units of the layer
        :param activation: (str) Specifying which type of activation function should be applied to the weights in the
         given layer. Default is set to 'rely' (Rectified Linear Unit).
        :return: An added fully connected layer to the network attribute in the CNN class.
        """
        if self.network is None:
            print("Network not initialized. Initialize network first using the 'init_network' method!")
        else:
            self.network = fully_connected(self.network, n_units, activation=activation)
            self.layer_overview = self.layer_overview.append(
                [1, n_units, activation]
            )  # appends a list of specifying values of the fully connected layer to the overview list

    def network_add_mpl(self, kernel_size, padding='same'):
        """
        Adds a max pooling layer to the specified network.
        :param kernel_size: (int) Pooling kernel size
        :param padding: (str) Specifying padding algorithm ('same' or 'valid')
        :return: An added max pooling layer to the network attribute in the CNN class.
        """
        if self.network is None:
            print("Network not initialized. Initialize network first using the 'init_network' method!")
        else:
            self.network = max_pool_2d(self.network, kernel_size, padding=padding)
            self.layer_overview = self.layer_overview.append(
                [2, kernel_size, padding]
            )  # appends a list of specifying values of the max pooling layer to the overview list

    def fit_model(self):
        # TODO: Fill this method out
        x = 1

    def save_model(self, filename='cnn_model'):
        if self.model is None:
            print("There is no model to save!")
        else:
            self.model.save(filename + ".tfl")
            print("Model successfully saves as: " + filename + ".tfl")

    def _display_model(self):
        layers = self.layer_overview[:1]
        if self.layer_overview[0][0] is None:
            batch_number = 'Not specified'
        else:
            batch_number = str(self.layer_overview[0][0])

        print("\n------------------------------------------")
        print("Deep Neural Network Overview")
        print("")

        for layer in layers:
            print("------------------------------------------")
            if layer[0] == 'cl':
                print("Layer {}: Convolutional Layer".format(layer.index + 1))
                print("Number of convolutional filters: {}".format(layer[1]))
                print("Size of filter(s): {}".format(layer[2]))
                print("Type of activation function: " + layer[3])
                print("Padding algorithm: " + layer[4])
            elif layer[0] == 'fcl':
                print("Layer {}: Fully Connected Layer".format(layer.index + 1))
                print("Number of units in layer: {}".format(layer[1]))
                print("Type of activation function: " + layer[3])
            elif layer[0] == 'mpl':
                print("Layer {}: Max Pooling Layer".format(layer.index + 1))
                print("Pooling kernel size: {}".format(layer[1]))
                print("Padding algorithm: " + layer[2])
            else:
                print("Layer {} - Layer type not detected".format(layer.index + 1))