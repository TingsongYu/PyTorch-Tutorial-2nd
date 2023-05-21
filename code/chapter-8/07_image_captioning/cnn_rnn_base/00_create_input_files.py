from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path=r'G:\deep_learning_data\coco_2014\image-caption-json\dataset_coco.json',
                       image_folder=r'G:\deep_learning_data\coco_2014\images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=r'G:\deep_learning_data\coco_2014\data-created',
                       max_len=50)
