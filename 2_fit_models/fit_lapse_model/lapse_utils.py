# Fit lapse model to IBL data
import autograd.numpy as np
import autograd.numpy.random as npr

def load_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    session = data[2]
    return inpt, y, session

def load_session_fold_lookup(file_path):
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    session_fold_lookup_table = data[0]
    return session_fold_lookup_table

def load_animal_list(list_file):
    container = np.load(list_file, allow_pickle=True)
    data = [container[key] for key in container]
    animal_list = data[0]
    return animal_list

# Parameter grid to search over when using multiple initializations
def get_parmax(i, M):
    if i <= M:
        return 10
    else:
        return 1


def get_parmin(i, M):
    if i <= M:
        return -10
    else:
        return 0


def get_parstart(i, M):
    if i <= M:
        return 2 * npr.randn(1)
    else:
        gamma = np.maximum(0.05 + 0.03 * npr.rand(1), 0)
        gamma = np.minimum(gamma, 1)
        return gamma


# Reshape hessian and calculate its inverse
def calculate_std(hessian):
    # Calculate inverse of Hessian (this is what we will actually use to
    # calculate variance)
    inv_hessian = np.linalg.inv(hessian)
    # Take diagonal elements and calculate square root
    std_dev = np.sqrt(np.diag(inv_hessian))
    return std_dev
