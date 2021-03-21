- Train and Pred
    - python train.py --data_dir UCLA-protest/ --batch_size 32 --lr 0.002 --print_freq 100 --epochs 100 --cuda
    - python pred.py --img_dir path/to/some/image/directory/ --output_csvpath result.csv --model model_best.pth.tar --cuda


- rec
- python train.py --data_dir "/home/Data/image_data/fine_grain_img_traintest" --batch_size 32 --lr 0.002 --print_freq 100 --epochs 100 --cuda

--data_dir "/home/Data/image_data/fine_grain_img_traintest" --batch_size 8 --lr 0.02 --print_freq 20 --epochs 100 --cuda