from Diffusion.Train import train, eval


def main(model_config=None):
    modelConfig = {
        "state": "eval",  # train or eval
        "epoch": 200,
        "batch_size": 80,  # 80
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda",
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_last_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8,
        "cifar10_dir": './cifar10'
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    # 注意修改：
    # 1. state: 训练和推理状态
    # 2. cifar10_dir，根据自己的情况设置文件夹路径
    # 3. batch_size， 12G的1080ti，可以设置80
    main()
