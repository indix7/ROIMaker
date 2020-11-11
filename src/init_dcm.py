import numpy as np
import pandas as pd
import os
# import dicom
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pydicom


class InitDcm:
    def __init__(self):
        pass

    def test_show_one_image(self, path):
        ds = pydicom.dcmread(path)
        plt.figure(figsize=(10, 10))
        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
        plt.show()

    @classmethod
    def load_scan(cls, path):
        slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        for s in slices:
            s.SliceThickness = slice_thickness
        return slices

    @classmethod
    def get_pixels_hu(cls, slices):
        image = np.stack([s.pixel_array for s in slices])
        image = image.astype(np.int16)

        # 设置边界外的元素为0
        image[image == -2000] = 0

        # 转换为HU单位
        for slice_number in range(len(slices)):

            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope

            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)

            image[slice_number] += np.int16(intercept)

        return np.array(image, dtype=np.int16)

    @classmethod
    def plot_3d(cls, image, threshold=-300):
        p = image.transpose(2, 1, 0)
        # verts, faces = measure.marching_cubes_classic(p, threshold)
        verts, faces = measure.marching_cubes(p, threshold, method="_lorensen")

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(verts[faces], alpha=0.70)
        face_color = [0.45, 0.45, 0.75]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

        ax.set_xlim(0, p.shape[0])
        ax.set_ylim(0, p.shape[1])
        ax.set_zlim(0, p.shape[2])
        plt.show()

    # @classmethod
    # def resample(cls, image, scan, new_spacing=[1,1,1]):
    #     spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    #
    #     resize_factor = spacing / new_spacing
    #     new_real_shape = image.shape * resize_factor
    #     new_shape = np.round(new_real_shape)
    #     real_resize_factor = new_shape / image.shape
    #     new_spacing = spacing / real_resize_factor
    #
    #     image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    #
    #     return image, new_spacing
    #
    # pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
    # print("Shape before resampling\t", first_patient_pixels.shape)
    # print("Shape after resampling\t", pix_resampled.shape)

    @classmethod
    def largest_label_volume(cls, im, bg=-1):
        vals, counts = np.unique(im, return_counts=True)

        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None

    @classmethod
    def segment_lung_mask(cls, image, fill_lung_structures=True):

        binary_image = np.array(image > -320, dtype=np.int8) + 1
        labels = measure.label(binary_image)

        background_label = labels[0, 0, 0]

        binary_image[background_label == labels] = 2

        if fill_lung_structures:
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = cls.largest_label_volume(labeling, bg=0)

                if l_max is not None:
                    binary_image[i][labeling != l_max] = 1

        binary_image -= 1
        binary_image = 1 - binary_image

        labels = measure.label(binary_image, background=0)
        l_max = cls.largest_label_volume(labels, bg=0)
        if l_max is not None:
            binary_image[labels != l_max] = 0

        return binary_image
