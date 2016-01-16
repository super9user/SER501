import pylab as pylab
from skimage import img_as_float
import numpy as np
from copy import deepcopy
import doctest


# Function calculates energy of each pixel in the image
# Asymptotic Complexity = rows X cols
def dual_gradient_energy(img):
    # rows and cols are the values of the matrix dimensions starting from 1
    """
    >>> img = np.array([[[0.5, 0.1, 0.45], [0.1, 0.3, 0.99], [0.21, 0.22, 0.12]], [[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]], [[0.9, 0.8, 0.7], [0.1, 0.2, 0.3], [0.4, 0.55, 0.67]]])
    >>> dual_gradient_energy(img)
    array([[ 2.97284516,  0.92661476,  0.90004052],
           [ 4.50765625,  4.23629321,  4.20025625],
           [ 0.62973661,  0.33451277,  1.54585625]])


    :rtype : array
    """
    rows, cols = img.shape[:2]

    # Initializing an empty array
    energy_matrix = np.empty(shape=(rows, cols))

    for r in range(0, rows):
        for c in range(0, cols):
            if cols - 1 > c > 0:
                # Not a Boundary condition
                r_x = img[r][c + 1][0] - img[r][c - 1][0]
                g_x = img[r][c + 1][1] - img[r][c - 1][1]
                b_x = img[r][c + 1][2] - img[r][c - 1][2]
            else:
                # Boundary conditions
                if c == 0:
                    # extreme left column
                    r_x = img[r][c + 1][0] - img[r][cols - 1][0]
                    g_x = img[r][c + 1][1] - img[r][cols - 1][1]
                    b_x = img[r][c + 1][2] - img[r][cols - 1][2]
                else:
                    # extreme right column
                    r_x = img[r][0][0] - img[r][c - 1][0]
                    g_x = img[r][0][1] - img[r][c - 1][1]
                    b_x = img[r][0][2] - img[r][c - 1][2]

            delta_x = np.square(r_x) + np.square(g_x) + np.square(b_x)

            if rows - 1 > r > 0:
                # Not a Boundary condition
                r_y = img[r + 1][c][0] - img[r - 1][c][0]
                g_y = img[r + 1][c][1] - img[r - 1][c][1]
                b_y = img[r + 1][c][2] - img[r - 1][c][2]
            else:
                # Boundary conditions
                if r == 0:
                    # extreme top row
                    r_y = img[r + 1][c][0] - img[rows - 1][c][0]
                    g_y = img[r + 1][c][1] - img[rows - 1][c][1]
                    b_y = img[r + 1][c][2] - img[rows - 1][c][2]
                else:
                    # extreme bottom row
                    r_y = img[0][c][0] - img[rows - 1][c][0]
                    g_y = img[0][c][1] - img[rows - 1][c][1]
                    b_y = img[0][c][2] - img[rows - 1][c][2]
            delta_y = np.square(r_y) + np.square(g_y) + np.square(b_y)

            energy = np.square(delta_x) + np.square(delta_y)
            energy_matrix[r][c] = energy
    return energy_matrix


# Find and return Min Energy Seam using Dynamic Programming approach
# Asymptotic Complexity = rows X cols
def find_seam(img):
    """
    >>> img = np.array([[7, 6, 8, 9, 3], [4, 5, 2, 1, 2], [7, 2, 3, 4, 9], [2, 1, 6, 4, 5], [8, 10, 1, 2, 3]])
    >>> find_seam(img)
    [2, 1, 2, 3, 4]


    :rtype : array
    """
    seam = []
    rows, cols = img.shape[:2]

    # Initializing an empty array
    cost_matrix = np.empty(shape=(rows, cols))

    for r in range(0, rows):
        for c in range(0, cols):
            current_energy = img[r][c]
            min_energy = 0
            if r != 0:
                # r is not the top row
                if cols - 1 > c > 0:
                    # Not a Boundary condition
                    north_east_energy = cost_matrix[r - 1][c + 1]
                    north_west_energy = cost_matrix[r - 1][c - 1]
                    north_energy = cost_matrix[r - 1][c]
                else:
                    # Boundary conditions
                    if c == 0:
                        # extreme left column
                        north_east_energy = cost_matrix[r - 1][c + 1]
                        north_west_energy = float("inf")
                    else:
                        # extreme right column
                        north_east_energy = float("inf")
                        north_west_energy = cost_matrix[r - 1][c - 1]

                    north_energy = cost_matrix[r - 1][c]

                # Find min out of the three energies
                min_energy = min(north_energy, north_east_energy, north_west_energy)

            cost_matrix[r][c] = min_energy + current_energy

    # Backtrack and find the min-energy seam
    col_index = -1
    for r in reversed(range(0, rows)):
        if r == (rows - 1):
            # Extreme Bottom Row
            min_energy_pixel = min(cost_matrix[r])
        else:
            if cols - 1 > col_index > 0:
                north_west_energy = cost_matrix[r][col_index - 1]
                north_east_energy = cost_matrix[r][col_index + 1]
            else:
                if col_index == 0:
                    # extreme left column
                    north_west_energy = float("inf")
                    north_east_energy = cost_matrix[r][col_index + 1]
                else:
                    # extreme right column
                    north_west_energy = cost_matrix[r][col_index - 1]
                    north_east_energy = float("inf")

            north_energy = cost_matrix[r][col_index]
            min_energy_pixel = min(north_west_energy, north_east_energy, north_energy)

        current_row = cost_matrix[r].tolist()
        col_index = current_row.index(min_energy_pixel)
        seam.append(col_index)

    return seam


# Plot seam on with red pixel and return img
# Asymptotic Complexity = rows X cols
def plot_seam(img, seam):
    """
    >>> img = np.array([[[0.5, 0.1, 0.45], [0.1, 0.3, 0.99], [0.21, 0.22, 0.12]], [[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]], [[0.9, 0.8, 0.7], [0.1, 0.2, 0.3], [0.4, 0.55, 0.67]]])
    >>> seam = [2, 1, 2, 3, 4]
    >>> plot_seam(img, seam)
    array([[[ 0.5 ,  0.1 ,  0.45],
            [ 0.1 ,  0.3 ,  0.99],
            [ 1.  ,  0.  ,  0.  ]],
    <BLANKLINE>
           [[ 0.  ,  0.  ,  1.  ],
            [ 1.  ,  0.  ,  0.  ],
            [ 0.  ,  1.  ,  0.  ]],
    <BLANKLINE>
           [[ 0.9 ,  0.8 ,  0.7 ],
            [ 0.1 ,  0.2 ,  0.3 ],
            [ 1.  ,  0.  ,  0.  ]]])

    :rtype : array
    """
    rows, cols = img.shape[:2]
    red_pixel = []
    for i in range(0, len(img[0][0])):
        if i == 0:
            red_pixel.append(1.0)
        else:
            red_pixel.append(0.0)
    for r in (range(0, rows)):
        for c in range(0, cols):
            if seam[r] == c:
                img[(rows - 1) - r][c] = red_pixel
    return img


# Remove seam from image and return updated image
# Asymptotic Complexity = rows X cols
def remove_seam(img, seam):
    """
    >>> img = np.array([[[0.5, 0.1, 0.45], [0.1, 0.3, 0.99], [0.21, 0.22, 0.12]], [[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]], [[0.9, 0.8, 0.7], [0.1, 0.2, 0.3], [0.4, 0.55, 0.67]]])
    >>> seam = [2, 1, 2, 3, 4]
    >>> remove_seam(img, seam)
    array([[[ 0.5 ,  0.1 ,  0.45],
            [ 0.1 ,  0.3 ,  0.99]],
    <BLANKLINE>
           [[ 0.  ,  0.  ,  1.  ],
            [ 0.  ,  1.  ,  0.  ]],
    <BLANKLINE>
           [[ 0.9 ,  0.8 ,  0.7 ],
            [ 0.1 ,  0.2 ,  0.3 ]]])

    :rtype : array
    """
    rows, cols = img.shape[:2]
    size_of_pixel = len(img[0][0])
    new_img = np.empty(shape=(rows, cols - 1, size_of_pixel))
    for r in reversed(range(0, rows)):
        new_cols = 0
        for c in range(0, cols):
            if seam[(rows - 1) - r] != c:
                new_img[r][new_cols] = deepcopy(img[r][c])
                new_cols = new_cols + 1

    return new_img


def display_input_output(input_img, output_img, energy_matrix, seam_plot):
    pylab.figure()
    pylab.gray()

    pylab.subplot(2, 2, 1)
    pylab.imshow(input_img)
    pylab.title("Input")

    pylab.subplot(2, 2, 2)
    pylab.imshow(output_img)
    pylab.title("Compressed Output")

    pylab.subplot(2, 2, 3)
    pylab.imshow(energy_matrix)
    pylab.title("Initial Energy Function")

    pylab.subplot(2, 2, 4)
    pylab.imshow(seam_plot)
    pylab.title("1st Seam Identified")
    pylab.show()


# Reduce width of image
def reduce_by_width(img, reduce_by_pixels):
    img = img_as_float(img)
    original_image = deepcopy(img)
    initial_energy, initial_seam = None, None
    for i in range(0, reduce_by_pixels):
        print "Removing Seam # ", i
        energy_matrix = dual_gradient_energy(img)
        seam = find_seam(energy_matrix)
        if i == 0:
            # storing initial values for later usage
            initial_energy = energy_matrix
            initial_seam = plot_seam(img, seam)
        img = remove_seam(img, seam)

    display_input_output(original_image, img, initial_energy, initial_seam)


# Reduce height of image
def reduce_by_height(img, reduce_by_pixels):
    img = img_as_float(img)
    original_image = deepcopy(img)
    img = img.transpose(1, 0, 2)
    initial_energy_matrix, initial_seam = None, None
    for i in range(0, reduce_by_pixels):
        print "Removing Seam # ", i
        energy_matrix = dual_gradient_energy(img)
        seam = find_seam(energy_matrix)
        if i == 0:
            # storing initial values for later usage
            initial_energy_matrix = energy_matrix.transpose(1, 0)
            initial_seam = plot_seam(img, seam).transpose(1, 0, 2)
        img = remove_seam(img, seam)
    img = img.transpose(1, 0, 2)
    display_input_output(original_image, img, initial_energy_matrix, initial_seam)


def main():
    img = pylab.imread("images/image1.png")
    
    reduce_by_width(img, 150)
    # reduce_by_height(img, 50)

    # run doctests
    doctest.testmod()


if __name__ == '__main__':
    main()
