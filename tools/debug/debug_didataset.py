from datasets.dynamicImage_dataset import DynamicImageDataset

def debug_didataset(cfg, make_fn, train_set):
    dataset = DynamicImageDataset(cfg, make_fn, train_set)
    
    print(dataset[3])