import io_liver
import nrrd

dataset = [(io_liver.load_data("data/{}".format(patient)), patient) for patient in ["Caso 1","Caso 2","Caso 3","Nl 1","Nl 2","Nl 3"]]

for data, patient in dataset:
    volumes, masks = data
    masks['dixon']['liver']['data'] = io_liver.align_segmentation(volumes['Dixon1']['data'], masks['dixon']['liver']['data'], patient)
    nrrd.write("{}-liver-dixon-aligned.nrrd".format(patient), masks['dixon']['liver']['data'], options=masks['dixon']['liver']['header'])