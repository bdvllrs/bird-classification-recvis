from os import path, curdir, listdir, remove, rename, makedirs
from shutil import rmtree


def clean_segmentation():
    segment_file = path.abspath(path.join(curdir, '..', 'bird_dataset', 'segmentations'))
    train_files = path.abspath(path.join(curdir, '..', 'bird_dataset', 'train_images'))
    val_images = path.abspath(path.join(curdir, '..', 'bird_dataset', 'val_images'))
    k = 0
    for directory in listdir(segment_file):
        # If not in train
        if directory not in listdir(train_files):
            rmtree(segment_file + '/' + directory)
        else:
            for file in listdir(segment_file + '/' + directory):
                file_as_jpg = file[:-4] + '.jpg'
                if file_as_jpg not in listdir(train_files + '/' + directory):
                    if file_as_jpg not in listdir(val_images + '/' + directory):
                        k += 1
                        remove(segment_file + '/' + directory + '/' + file)
                    else:
                        makedirs(segment_file + '/' + 'val_images' + '/' + directory, exist_ok=True)
                        rename(segment_file + '/' + directory + '/' + file,
                               segment_file + '/' + 'val_images' + '/' + directory + '/' + file)
                else:
                    print('oui')
                    makedirs(segment_file + '/' + 'train_images' + '/' + directory, exist_ok=True)
                    rename(segment_file + '/' + directory + '/' + file,
                           segment_file + '/' + 'train_images' + '/' + directory + '/' + file)
    print(k, 'files deleted')


if __name__ == '__main__':
    clean_segmentation()
