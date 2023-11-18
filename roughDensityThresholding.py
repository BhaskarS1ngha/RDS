# matplotlib.use('TkAgg')
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sys import argv
import multiprocessing as mp
import os
import time

ix = 0
iy = 0
coords = []
args = argv
Theta = 40
NUM_WORKERS = os.cpu_count() - 1  # optimal is 6 workers
Visited = None
CLUSTERS = None


def onclick(event):  # callback for matplotlib imshow
    global ix, iy
    ix, iy = event.xdata, event.ydata
    global coords
    coords.append((int(iy), int(ix)))
    print(iy, ix)
    return coords


def Choose_initiator(CI_Visited, ROI_pixels):
    Pixel_array = []
    if np.all(CI_Visited) != 1:
        for i in range(len(CI_Visited)):
            if CI_Visited[i] != 1:
                Pixel_array.append(ROI_pixels[i])
    return Pixel_array


def eight_Neighbor(image, x, y, ROI_pixels):
    neighbor = [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1),
                (x - 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1)]
    row = []
    col = []
    neighbor_final = []
    neighbourhood = np.dstack(neighbor)

    for u in neighbourhood[0][0]:
        if image.shape[0] > u >= 0:
            row.append(u)
    for u in neighbourhood[0][1]:
        if image.shape[1] > u >= 0:
            col.append(u)

    if len(row) >= 4 and len(col) >= 4:
        for v in range(0, len(neighbor)):
            if neighbor[v] in ROI_pixels:
                neighbor_final.append(neighbor[v])
        return neighbor_final
    else:
        return 0


def initiator(image, x, y, B, D, ROI_pixels):
    global Theta
    neighbourhood_n = eight_Neighbor(image, x, y, ROI_pixels)
    if neighbourhood_n != 0:
        X = False
        for j in range(0, len(neighbourhood_n)):
            variable = int(image[B, D]) - int(image[neighbourhood_n[j]])
            if np.abs(variable) <= Theta:
                X = True
            else:
                break
        if X == True:
            return (x, y)
        else:
            return 0
    else:
        return 0


def new_initiator(image, x1, y1, B, D, m, ROI_pixels):  # m = element of N to be returned if X is true
    global Theta
    neighbourhood = eight_Neighbor(image, x1, y1, ROI_pixels)
    # print("new neighbor",neighbourhood_n)
    if neighbourhood != 0:
        X = False
        for j in range(0, len(neighbourhood)):
            if np.abs(int(image[B, D]) - int(image[neighbourhood[j]])) <= Theta:
                X = True
            else:
                break
        if X:
            return m
        else:
            return 0
    else:
        return 0


def find_clusters(image, B, D, m, ROI_pixels):
    global Theta, CLUSTERS, Visited
    (x1, y1) = m
    index = ROI_pixels.index((x1, y1))
    if Visited[index] != 1:
        Visited[index] = 1
    neighbourhood_new = eight_Neighbor(image, x1, y1, ROI_pixels)
    New_initiator = new_initiator(image, x1, y1, B, D, m, ROI_pixels)
    result_dict = {
        'Neighbourhood': [],
        'indices': [],
        'index': -1
    }
    if New_initiator != 0:
        result_dict['Neighbourhood'] = neighbourhood_new
        if (CLUSTERS[ROI_pixels.index((x1, y1))]) != 0:
            result_dict['index'] = ROI_pixels.index((x1, y1))
        for i in range(0, len(neighbourhood_new)):
            if (CLUSTERS[ROI_pixels.index(neighbourhood_new[i])]) == 0:
                result_dict['indices'].append(ROI_pixels.index(neighbourhood_new[i]))

    return result_dict


def cluster(IMAGE_PATH, Threshold=70):
    global Theta, Visited, CLUSTERS
    image = cv2.imread(IMAGE_PATH, 0)

    ROI_pixels = []
    # n_workers = mp.cpu_count()-1        # use all but 1 available processors
    n_workers = 7  # optimal processor count as per experiments
    print(f"Starting segmentation with {n_workers} processes")
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if Threshold <= image[i, j] <= 255:
                ROI_pixels.append((i, j))
            else:
                image[i, j] = 0
    Visited = mp.Array('i', len(ROI_pixels))
    CLUSTERS = mp.Array('i', len(ROI_pixels))

    print("roi", len(ROI_pixels))
    print("Processing, please wait...")
    c = 0
    k = 0
    #    Pixel_Array = []
    t0 = time.time_ns()
    while np.all(Visited) != 1:

        Pixel_Array = Choose_initiator(Visited, ROI_pixels)
        (x, y) = random.choice(Pixel_Array)
        Visited[ROI_pixels.index((x, y))] = 1

        (B, D) = (x, y)
        index = ROI_pixels.index((x, y))

        Initiator = initiator(image, x, y, B, D, ROI_pixels)

        neighbourhood_n = eight_Neighbor(image, x, y, ROI_pixels)

        if Initiator != 0:
            c = c + 1
            Visited[index] = 1
            CLUSTERS[index] = c
            for i in range(0, len(neighbourhood_n)):
                CLUSTERS[ROI_pixels.index(neighbourhood_n[i])] = c
                Visited[ROI_pixels.index(neighbourhood_n[i])] = 1

            N = neighbourhood_n
            Array = set(neighbourhood_n)

            try:

                while N:

                    m = 0
                    arg_list = []
                    for i in range(0, len(N)):
                        Visited[ROI_pixels.index(N[m])] = 1
                        arg_list.append((image, B, D, N[i], ROI_pixels))
                    while N:
                        N.pop(0)

                    with mp.Pool(processes=n_workers) as pool:
                        results = pool.starmap(find_clusters, arg_list)

                    for result in results:
                        if result['index'] != -1:
                            Visited[result['index']] = 1
                            CLUSTERS[result['index']] = c

                        for x in result['Neighbourhood']:
                            if x not in Array and CLUSTERS[ROI_pixels.index(x)] == 0:
                                N.append(x)
                                Array.add(x)
                        for indx in result['indices']:
                            Visited[indx] = 1
                            CLUSTERS[indx] = c

            except IndexError:
                print(f'hit index error at {m}')
                pass

        k = k + 1

    t1 = (time.time_ns() - t0) / 1000000000

    for i in range(0, len(CLUSTERS)):
        image[ROI_pixels[i]] = CLUSTERS[i]
    sum = 0

    dstImage = np.zeros(image.shape)
    dstImage = cv2.normalize(image, dstImage, 0, 255, cv2.NORM_MINMAX)

    fig = plt.figure()
    plt.suptitle(IMAGE_PATH, fontsize=16)
    implot = fig.add_subplot(111)
    implot.imshow(dstImage, cmap='gray')
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.axis('off')
    plt.show()

    clusterids = []
    for coord in coords:
        clusterids.append(image[coord[0], coord[1]])

    new_image = np.zeros(image.shape)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            for cluster in clusterids:
                if image[i, j] == cluster:
                    new_image[i, j] = 255

    print("sum", sum)
    print("image", image.size, len(CLUSTERS))

    print(f'Time taken: {t1} seconds')
    return new_image


if __name__ == '__main__':
    args = argv
    num_args = len(args)
    if num_args < 2 or num_args > 3:
        print("Usage: python3 roughDensityThresholding.py <image_path> <threshold>\n--Threshold is optional and "
              "defaults to 70")
        exit(1)
    elif num_args == 2:
        mask = cluster(args[1])
    else:
        # check if args[2] is an integer
        try:
            int(args[2])
        except ValueError:
            print("Usage: python3 roughDensityThresholding.py <image_path> <threshold>\n#Threshold is optional and "
                  "defaults to 70")
            exit(1)
        mask = cluster(args[1], int(args[2]))
    cv2.imshow("result", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
