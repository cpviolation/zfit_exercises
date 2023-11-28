import tensorflow as tf

def PX(momentum):
    """A function to extrapolate the x component of the momentum in a 4-vector defined with tensorflow

    Args:
        momentum (Tensor): a tensor of shape (n, 4) where n is the number of 4-vectors

    Returns:
        Tensor: a tensor of shape (n,) with the x component of the momentum
    """    
    return momentum[:, 0]

def PY(momentum):
    """A function to extrapolate the y component of the momentum in a 4-vector defined with tensorflow

    Args:
        momentum (Tensor): a tensor of shape (n, 4) where n is the number of 4-vectors

    Returns:
        Tensor: a tensor of shape (n,) with the y component of the momentum
    """    
    return momentum[:, 1]

def PZ(momentum):
    """A function to extrapolate the z component of the momentum in a 4-vector defined with tensorflow

    Args:
        momentum (Tensor): a tensor of shape (n, 4) where n is the number of 4-vectors

    Returns:
        Tensor: a tensor of shape (n,) with the z component of the momentum
    """    
    return momentum[:, 2]

def E(momentum):
    """A function to extrapolate the t component of the momentum in a 4-vector defined with tensorflow

    Args:
        momentum (Tensor): a tensor of shape (n, 4) where n is the number of 4-vectors

    Returns:
        Tensor: a tensor of shape (n,) with the t component of the momentum
    """    
    return momentum[:, 3]

def P(momentum):
    """A function to extrapolate the absolute momentum of a 4-vector defined with tensorflow

    Args:
        momentum (Tensor): a tensor of shape (n, 4) where n is the number of 4-vectors

    Returns:
        Tensor: a tensor of shape (n,) with the absolute momentum
    """    
    return tf.sqrt(tf.square(PX(momentum)) + tf.square(PY(momentum)) + tf.square(PZ(momentum)))

def P2(momentum):
    """A function to extrapolate the squared absolute momentum of a 4-vector defined with tensorflow

    Args:
        momentum (Tensor): a tensor of shape (n, 4) where n is the number of 4-vectors

    Returns:
        Tensor: a tensor of shape (n,) with the squared absolute momentum
    """    
    return tf.square(PX(momentum)) + tf.square(PY(momentum)) + tf.square(PZ(momentum))

def PT(momentum):
    """A function to extrapolate the transverse momentum in a 4-vector defined with tensorflow

    Args:
        momentum (Tensor): a tensor of shape (n, 4) where n is the number of 4-vectors

    Returns:
        Tensor: a tensor of shape (n,) with the transverse momentum
    """    
    return tf.sqrt(tf.square(PX(momentum)) + tf.square(PY(momentum)))

def M(momentum):
    """A function to extrapolate the mass of a 4-vector defined with tensorflow

    Args:
        momentum (Tensor): a tensor of shape (n, 4) where n is the number of 4-vectors

    Returns:
        Tensor: a tensor of shape (n,) with the mass
    """    
    return tf.sqrt(tf.square(E(momentum)) - P2(momentum))

def M2(momentum):
    """A function to extrapolate the squared mass of a 4-vector defined with tensorflow

    Args:
        momentum (Tensor): a tensor of shape (n, 4) where n is the number of 4-vectors

    Returns:
        Tensor: a tensor of shape (n,) with the squared mass
    """    
    return tf.square(E(momentum)) - P2(momentum)