# PYTHONIOENCODING=utf-8 python main.py --cuda
# python -u Wavelet/factornn/main.py --cuda --batch_size=128 --epochs=100 --data_path=Wavelet/factornn/260all_crossval/

nohup python -u main.py --cuda --batch_size=256 --epochs=200 --data_path=/liyuejin/Wavelet/factornn/260all_crossval/ > train0121.log 2>&1 &