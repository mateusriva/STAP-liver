"""
Tool for computing and displaying the color histogram of livers in patients.
"""
import io_liver

import matplotlib.pyplot as plt
import numpy as np

io_liver.debug = True

# Load patient data
#dataset = [(io_liver.load_data("data/{}".format(patient)), patient) for patient in ["Caso 2"]]
dataset = [(io_liver.load_data("data/{}".format(patient)), patient) for patient in ["Caso 1","Caso 2","Caso 3","Nl 1","Nl 2","Nl 3"]]

# Compute liver histogram for each patient
for pack, patient in dataset:
    volumes, masks = pack
    
    # Aligning mask
    masks['dixon']['liver']['data'] = io_liver.align_segmentation(volumes['Dixon1']['data'], masks['dixon']['liver']['data'], patient)

    # creating histogram dictionaries
    print("Computing {} histograms...".format(patient))
    liver_histograms = ({},{},{},{})
    non_liver_histograms = ({},{},{},{})

    data_shape = volumes["Dixon1"]["data"].shape
    for x in range(0,data_shape[0]):
      print("{:.2f}%\r".format(x*100/data_shape[0]), end="")
      for y in range(0,data_shape[1]):
        for z in range(0,data_shape[2]):
            # If voxel is liver, add histograms
            if masks["dixon"]["liver"]["data"][x,y,z] == 1:
                # Dixon1
                if volumes["Dixon1"]["data"][x,y,z] in liver_histograms[0]:
                    liver_histograms[0][volumes["Dixon1"]["data"][x,y,z]] += 1
                else:
                    liver_histograms[0][volumes["Dixon1"]["data"][x,y,z]] = 1
                # Dixon2
                if volumes["Dixon2"]["data"][x,y,z] in liver_histograms[1]:
                    liver_histograms[1][volumes["Dixon2"]["data"][x,y,z]] += 1
                else:
                    liver_histograms[1][volumes["Dixon2"]["data"][x,y,z]] = 1
                # Dixon3
                if volumes["Dixon3"]["data"][x,y,z] in liver_histograms[2]:
                    liver_histograms[2][volumes["Dixon3"]["data"][x,y,z]] += 1
                else:
                    liver_histograms[2][volumes["Dixon3"]["data"][x,y,z]] = 1
                # Dixon4
                if volumes["Dixon4"]["data"][x,y,z] in liver_histograms[3]:
                    liver_histograms[3][volumes["Dixon4"]["data"][x,y,z]] += 1
                else:
                    liver_histograms[3][volumes["Dixon4"]["data"][x,y,z]] = 1
            # If not, add histograms
            else:
                # Ignore black and black noise pixels
                if volumes["Dixon1"]["data"][x,y,z] < 10:
                    continue
                # Dixon1
                if volumes["Dixon1"]["data"][x,y,z] in non_liver_histograms[0]:
                    non_liver_histograms[0][volumes["Dixon1"]["data"][x,y,z]] += 1
                else:
                    non_liver_histograms[0][volumes["Dixon1"]["data"][x,y,z]] = 1
                # Dixon2
                if volumes["Dixon2"]["data"][x,y,z] in non_liver_histograms[1]:
                    non_liver_histograms[1][volumes["Dixon2"]["data"][x,y,z]] += 1
                else:
                    non_liver_histograms[1][volumes["Dixon2"]["data"][x,y,z]] = 1
                # Dixon3
                if volumes["Dixon3"]["data"][x,y,z] in non_liver_histograms[2]:
                    non_liver_histograms[2][volumes["Dixon3"]["data"][x,y,z]] += 1
                else:
                    non_liver_histograms[2][volumes["Dixon3"]["data"][x,y,z]] = 1
                # Dixon4
                if volumes["Dixon4"]["data"][x,y,z] in non_liver_histograms[3]:
                    non_liver_histograms[3][volumes["Dixon4"]["data"][x,y,z]] += 1
                else:
                    non_liver_histograms[3][volumes["Dixon4"]["data"][x,y,z]] = 1

    print("100%  ")
    # Displaying histograms for this patient
    plt.subplot(2,2,1)
    plt.bar(liver_histograms[0].keys(), liver_histograms[0].values(), color='g')
    plt.title("{} Dixon1 liver".format(patient))
    plt.xlim((0,1000))
    plt.subplot(2,2,2)
    plt.bar(liver_histograms[1].keys(), liver_histograms[1].values(), color='g')
    plt.title("{} Dixon2 liver".format(patient))
    plt.xlim((0,1000))
    plt.subplot(2,2,3)
    plt.bar(liver_histograms[2].keys(), liver_histograms[2].values(), color='g')
    plt.title("{} Dixon3 liver".format(patient))
    plt.xlim((0,1000))
    plt.subplot(2,2,4)
    plt.bar(liver_histograms[3].keys(), liver_histograms[3].values(), color='g')
    plt.title("{} Dixon4 liver".format(patient))
    plt.xlim((0,1000))

    plt.savefig("histograms/liver_{}.png".format(patient))

    plt.clf()

    plt.subplot(2,2,1)
    plt.bar(non_liver_histograms[0].keys(), non_liver_histograms[0].values(), color='g')
    plt.title("{} Dixon1 non_liver".format(patient))
    plt.xlim((0,1500))
    plt.subplot(2,2,2)
    plt.bar(non_liver_histograms[1].keys(), non_liver_histograms[1].values(), color='g')
    plt.title("{} Dixon2 non_liver".format(patient))
    plt.xlim((0,1500))
    plt.subplot(2,2,3)
    plt.bar(non_liver_histograms[2].keys(), non_liver_histograms[2].values(), color='g')
    plt.title("{} Dixon3 non_liver".format(patient))
    plt.xlim((0,1500))
    plt.subplot(2,2,4)
    plt.bar(non_liver_histograms[3].keys(), non_liver_histograms[3].values(), color='g')
    plt.title("{} Dixon4 non_liver".format(patient))
    plt.xlim((0,1500))

    plt.savefig("histograms/nonliver_{}.png".format(patient))
    plt.close()

    # Printing modes and stddevs...
    print("id\tmode\tstddev")
    for x in range(4):
        mode = max(liver_histograms[x].keys(), key=(lambda key: liver_histograms[x][key]))
        stddev = np.std([item for sublist in [[key]* liver_histograms[x][key] for key in liver_histograms[x].keys()] for item in sublist])
        print("Dixon{}-Liver\t{}\t{}".format(x+1,mode, stddev))

        mode = max(non_liver_histograms[x].keys(), key=(lambda key: non_liver_histograms[x][key]))
        stddev = np.std([item for sublist in [[key]* non_liver_histograms[x][key] for key in non_liver_histograms[x].keys()] for item in sublist])
        print("Dixon{}-NonLiver\t{}\t{}".format(x+1,mode, stddev))