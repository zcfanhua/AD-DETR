import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('AD-DETR.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=4, 
                workers=2, 
                project='runs/train',
                name='exp',
                )