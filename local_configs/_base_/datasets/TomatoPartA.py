from .. import *

# Dataset config
"""Dataset Path"""
C.dataset_name = "TomatoPartA"
C.dataset_path = osp.join(C.root_dir, "TomatoPartA")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".jpg"
C.gt_root_folder = osp.join(C.dataset_path, "label.json")
C.gt_format = ".json"
C.gt_transform = False
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = (
    True  # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
)
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True
C.num_train_imgs = 266
C.num_eval_imgs = 54
C.num_classes = 4
C.class_names = [
    "tomato",
]

"""Image Config"""
C.background = 0
C.image_height = 640
C.image_width = 640
C.rgb_norm_mean = np.array([0.0201, 0.0202, 0.0107])
C.rgb_norm_std = np.array([0.0835, 0.0765, 0.0484])
C.x_norm_mean = np.array([0.0612, 0.0612, 0.0612])
C.x_norm_std = np.array([0.2295, 0.2295, 0.2295])
