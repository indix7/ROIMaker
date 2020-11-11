from src.init_dcm import InitDcm
import matplotlib.pyplot as plt


def process_show_bone(file_path):
    patient = InitDcm.load_scan(file_path)
    patient_pixels = InitDcm.get_pixels_hu(patient)
    InitDcm.plot_3d(patient_pixels, 400)


def process_show_lung(file_path):
    patient = InitDcm.load_scan(file_path)
    patient_pixels = InitDcm.get_pixels_hu(patient)
    lung = InitDcm.segment_lung_mask(patient_pixels, False)
    lung_fill = InitDcm.segment_lung_mask(patient_pixels, True)
    InitDcm.plot_3d(lung_fill, 0)


def process_test_one_image(file_path):
    patient = InitDcm.load_scan(file_path)
    patient_pixels = InitDcm.get_pixels_hu(patient)
    plt.hist(patient_pixels.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()
    plt.imshow(patient_pixels[80], cmap=plt.cm.gray)
    plt.show()


def show_all(file_path):
    patient = InitDcm.load_scan(file_path)
    patient_pixels = InitDcm.get_pixels_hu(patient)
    InitDcm.plot_3d(patient_pixels)


if __name__ == "__main__":
    file_path = "D:\\indix\\Code\\PythonProject\\Hospital\\resource\\mount\\mount\\CT3103586-Yang Jie"
    # process_test_one_image(file_path)
    # show_all(file_path)
    # process_show_bone(file_path)
    process_show_lung(file_path)




