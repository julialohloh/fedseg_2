"""
Pipeline Steps
1. Centralised data analysis
    - 
2. Centralised model training and analysis
    - Pixel accuracy for epoch
    - Mean pixel accuracy across all epochs
    - IoU(Jaccard) for epoch
    - IoU(Jaccard) across all epochs
    - F1 for epoch
    - F1 across all epochs
    - Cross stats

3. Partition simulation and analysis
    - 
4. Partition model training and anlalysis
5. Federated Training
6. Federated analysis - Global model
7. Federated analysis - Local models
8. Establish and present comparative metrics
"""
import data_utils_ray
import train_utils
import analyse_utils
import ray

configfile: "snakeconfig.yaml"

rule prep_data:
    input:
        # "msd_data/splitting_edited_test_edited.csv",
        # "msd_data/splitting_edited_train_edited.csv",
        # "msd_data/splitting_edited_val_edited.csv"
        f"{config['data']['train_path']}",
        f"{config['data']['val_path']}",
        f"{config['data']['test_path']}"
    output:
        # directory("data_snakemake")
        directory(f"{config['data']['dest_folder']}/data/"),
        directory(f"{config['data']['dest_folder']}/display/"),
        directory(f"{config['data']['dest_folder']}/segmentations/"),
        directory(f"{config['data']['dest_folder']}/weights/")
    params:
        source_root_folder = config["data"]["source_root_folder"],
        train_path = config["data"]["train_path"],
        val_path = config["data"]["val_path"],
        test_path = config["data"]["test_path"],
        source_folder= config["data"]['source_folder'],
        dest_folder= config["data"]["dest_folder"],
        w1 = config["data"]["w1"],
        w2 = config["data"]["w2"],
    run:
        ray.init()
        print(f"Hello world! prep_data rule is running.{params.source_root_folder} is the source")
        combined_files = data_utils_ray.get_files(train_path=params.train_path,val_path=params.val_path,test_path=params.test_path,source_folder=params.source_folder)
        dest_folder = params.dest_folder
        data_utils_ray.prep_msd_data_ray(combined_files=combined_files,dest=dest_folder,w1=params.w1,w2=params.w2)
        ray.shutdown()
        
rule train_centralised:
    input:
    # The input has to match the output of the rule data_prep
        f"{config['data']['dest_folder']}/weights"
    output:
        # This rule will output a directory called checkpointns
        # directory("checkpoints")
        f"{config['model_params']['checkpoint_path']}"
    params:
        root_dir = config["data"]["dest_folder"],
        num_channels = config["model_params"]["num_channels"],
        num_filters = config["model_params"]["num_filters"],
        kernel_h = config["model_params"]["kernel_h"],
        kernel_w = config["model_params"]["kernel_w"],
        kernel_c = config["model_params"]["kernel_c"],
        stride_conv = config["model_params"]["stride_conv"],
        pool = config["model_params"]["pool"],
        stride_pool = config["model_params"]["stride_pool"],
        num_class = config["model_params"]["num_class"],
        epochs = config["model_params"]["epochs"],
        batch_size = config["model_params"]["batch_size"],
        patience = config["model_params"]["patience"],
        checkpoint_path = config["model_params"]["checkpoint_path"],
        optim_type = config["optimizer"]["optim_type"],
        lr = config["optimizer"]["lr"],
        momentum = config["optimizer"]["momentum"],
        weight_decay = config["optimizer"]["weight_decay"],
        sched_type = config["scheduler"]["sched_type"],
        step_size = config["scheduler"]["step_size"],
        gamma = config["scheduler"]["gamma"],
        w1 = config["data"]["w1"],
        w2 = config["data"]["w2"],
        

    run:
        import os
        train_utils.run_training(root_dir=params.root_dir,num_channels=params.num_channels,num_filters=params.num_filters,\
        kernel_h=params.kernel_h,kernel_w=params.kernel_w,kernel_c=params.kernel_c,stride_conv=params.stride_conv,\
        pool=params.pool,stride_pool=params.stride_pool,num_class=params.num_class,epochs=params.epochs,lr=params.lr,\
        momentum=params.momentum,weight_decay=params.weight_decay,step_size=params.step_size,gamma=params.gamma,w1=params.w1,w2=params.w2,\
        checkpoint_path=params.checkpoint_path,batch_size=params.batch_size,patience=params.patience)

rule run_analysis:
    input:
        f"{config['model_params']['checkpoint_path']}"
    output:
        f"{config['analysis']['output_dir_name']}/metrics.txt"
    params:
        num_channels = config["model_params"]["num_channels"],
        num_filters = config["model_params"]["num_filters"],
        kernel_h = config["model_params"]["kernel_h"],
        kernel_w = config["model_params"]["kernel_w"],
        kernel_c = config["model_params"]["kernel_c"],
        stride_conv = config["model_params"]["stride_conv"],
        pool = config["model_params"]["pool"],
        stride_pool = config["model_params"]["stride_pool"],
        num_class = config["model_params"]["num_class"],
        epochs = config["model_params"]["epochs"],
        # analysis
        root_dir = config["analysis"]["root_dir"],
        state_dict_path = config["analysis"]["state_dict_path"],
        mode = "test",
        w1 = config["data"]["w1"],
        w2 = config["data"]["w2"],
        output_dir_name = config["analysis"]["output_dir_name"],
        batch_size = config["analysis"]["batch_size"],
        shuffle = config["analysis"]["shuffle"],
        # TBC - include transform as a param?
        transform = None,
        ind_dice_filename = config["analysis"]["ind_dice_filename"]

    run:
        # create dataset
        dataset = analyse_utils.MsdTestDataset(root_dir=params.root_dir,mode=params.mode,w1=params.w1,w2=params.w2,transform=params.transform)
        dataloader = analyse_utils.create_dataloader(dataset,batch_size=params.batch_size,shuffle=params.shuffle)
        # instantiate model
        model_params = {
            'num_channels': params.num_channels,
            'num_filters': params.num_filters,
            'kernel_h': params.kernel_h,
            'kernel_w': params.kernel_w,
            'kernel_c': params.kernel_c,
            'stride_conv': params.stride_conv,
            'pool': params.pool,
            'stride_pool': params.stride_pool,
            'num_class': params.num_class,
            # 'epochs': epochs
        }
        # run validation
        print(f"Running analysis...")
        print("Printing dirs")
        # for root,dirs,files in os.walk("./"):
        #     print(dirs)
        # returns avg test loss and avg test dice
        test_loss,test_dice_0,test_dice_1,test_dice_2,test_dice_avg = analyse_utils.test_model(model_params=model_params,state_dict_path=params.state_dict_path,data_loader=dataloader,output_path=params.output_dir_name)
        # Calculate dice score/metrics
        # test_dice_score = analyse_utils.calc_dice(root_dir=params.output_dir_name)
        with open(f"{params.output_dir_name}/metrics.txt","w") as f:
            f.write("Test loss:" + str(test_loss))
            f.write("\n")
            f.write("Test Dice score[Class 0]:" + str(test_dice_0))
            f.write("\n")
            f.write("Test Dice score[Class 1]:" + str(test_dice_1))
            f.write("\n")
            f.write("Test Dice score[Class 2]:" + str(test_dice_2))
            f.write("\n")
            f.write("Test Dice score:" + str(test_dice_avg))
        df = analyse_utils.calc_individual_dice(params.output_dir_name)
        df_output_path = os.path.join(params.output_dir_name,params.ind_dice_filename)
        df.to_csv(df_output_path,index=False)

       







