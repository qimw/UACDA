all_file_paths = {
    # cityscapes
    'city_day_imgs_txt':      'data/cityscapes/leftImg8bit/cityscapes_day.txt',
    'city_night_imgs_txt':    'data/cityscapes/cyclegan_unet256_night/cityscapes_night.txt',
    'city_twilight_imgs_txt': 'data/cityscapes/cyclegan_unet256_twilight/cityscapes_twilight.txt',
    'city_lbls_txt': 'data/cityscapes/gtFine/gts.txt',

    # dark zurich
    'zurich_day_imgs_txt':  'data/dark_zurich/zurich_day.txt',
    'zurich_day_plbls_txt': './file_lists/zurich_day_plbls.txt', # auto generate

    'zurich_night_imgs_txt':  'data/dark_zurich/zurich_night.txt',
    'zurich_night_plbls_txt': './file_lists/zurich_night_plbls.txt', # auto generate

    'zurich_twilight_imgs_txt':  'data/dark_zurich/zurich_twilight.txt',
    'zurich_twilight_plbls_txt': './file_lists/zurich_twilight_plbls.txt', # auto generate

    'zurich_test_imgs_txt': 'data/dark_zurich/zurich_night_test.txt',
    'zurich_val_imgs_txt': 'data/dark_zurich/zurich_night_val.txt',
}
