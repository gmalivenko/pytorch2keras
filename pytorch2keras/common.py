import random
import string


def random_string(length):
    """
    Generate a random string for the layer name.
    :param length: a length of required random string
    :return: generated random string
    """
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))