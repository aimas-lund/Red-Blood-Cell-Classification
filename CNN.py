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

    def pre_processing(self):   # normalizes the training data
        image_prep = ImagePreprocessing()
        image_prep.add_featurewise_zero_center()
        image_prep.add_featurewise_stdnorm()

        self.data_pre_processing = image_prep

    def augmentation(self, max_angle=5., sigma_max=3.,
                     flip_left_right=True,
                     random_rotation=True,
                     random_blur=True):
        image_aug = ImageAugmentation()
        if flip_left_right:
            image_aug.add_random_flip_leftright()  # adds left- and right flipped images to the training data

        if random_rotation:
            image_aug.add_random_rotation(max_angle=max_angle)  # rotates random training data by a specified angle (
        # degrees)

        if random_blur:
            image_aug.add_random_blur(sigma_max=sigma_max)  # blurs random training data by a specified sigma

        if any([flip_left_right, random_blur, random_rotation]):
            self.data_augmentation = image_aug
        else:
            print("No augmentation selected")

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
        :param name: (string) Given name to the network. Will be called 'InputData' by default.
        :return: An initialized network attribute for the CNN class.
        """
        if shape is None:
            shape = [None, 150, 800, 3]

        self.network = input_data(shape=shape,
                                  data_preprocessing=self.data_pre_processing,
                                  data_augmentation=self.data_augmentation,
                                  name=name)
        self.layer_overview = [shape]

    def network_add_conv_layer(self, activation='relu'):
        """
        :param activation: (string) Specifying which type of activation should be applied to the weights in the given
        layer. Default is set to 'rely' (Rectified Linear Unit).
        :return: An added layer to the network attribute in the CNN class.
        """
        if self.network is None:
            print("Network not initialized. Initialize network first using the 'init_network' method!")
        else:
            self.network = conv_2d(self.network, activation=activation)
