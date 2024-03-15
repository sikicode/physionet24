import numpy as np

def tanh_sat(x, param, mode='ksigma'):
    """
    Saturates outlier samples using a tanh shape function

    Args:
    - x: Input data, can be a vector or a matrix (channels x time)
    - param: Scaling factor for the saturation level:
        - If mode == 'ksigma' or no mode defined: k times the standard deviation of each channel
        - If mode = 'absolute': a vector of absolute thresholds used to
          saturate each channel. If param is a scalar, the same value is
          used for all channels
    - mode (optional): 'ksigma' or 'absolute'. Default is 'ksigma'

    Returns:
    - y: Saturated data with outliers replaced by the saturation level
    """
    if mode == 'ksigma':
        alpha = param * np.std(x, axis=1, keepdims=True)  # Compute the scaling factor based on the standard deviation of each channel
    elif mode == 'absolute':
        if np.isscalar(param):
            alpha = param * np.ones((1, x.shape[0]))
        elif np.isvector(param):
            if len(param) == x.shape[0]:
                alpha = np.array(param)[np.newaxis, :]
            else:
                raise ValueError('Parameter must be a scalar or a vector with the same number of elements as the data channels')
    else:
        raise ValueError('Undefined mode')

    y = np.dot(np.diag(alpha.flatten()), np.tanh(np.dot(np.diag(1.0 / alpha.flatten()), x)))  # Scale the input data and apply the tanh function to saturate outliers
    return y
