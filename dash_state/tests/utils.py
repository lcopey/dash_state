import random
from string import ascii_lowercase


def random_string():
    n = random.randint(2, 10)
    return "".join(random.sample(ascii_lowercase, k=n))
