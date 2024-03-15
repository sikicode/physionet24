import numpy as np

def ecg_grid_size_from_paper(img, paper_width, unit):
    """
    Calculates the grid size of an ECG image based on the physical dimensions
    (width) of the paper.

    Parameters:
    img (numpy.ndarray): A 2D matrix representing the ECG image.
    paper_width (float or tuple): The paper width, corresponding to the second dimension of the input image.
    unit (str): A string specifying the unit of paper_width ('cm' or 'in').

    Returns:
    tuple: Estimated coarse and fine grid sizes (in pixels).
    """

    # Function implementation
    width = img.shape[1]

    # Convert paper size to inches if it's in centimeters
    if unit.lower() == 'cm':
        paper_width_in_inch = paper_width / 2.54  # 1 inch = 2.54 cm
    else:
        paper_width_in_inch = paper_width  # Already in inches

    # Calculating pixels per inch
    pxls_per_inch = width / paper_width_in_inch

    # Standard coarse ECG grid 5mm x 5mm (0.5mV x 0.2s)
    # Converting from mm to inches (1 inch = 25.4 mm)
    coarse_grid_res = pxls_per_inch * 5 / 25.4  # coarse grid size in pixels

    # Standard fine ECG grid 5mm x 5mm (0.1mV x 40ms)
    fine_grid_res = coarse_grid_res / 5  # fine grid size in pixels

    return coarse_grid_res, fine_grid_res
