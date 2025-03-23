import argparse # terminal에서 실행시키기 위한 것
import training
import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chest_radiograph_PCD") # 프로그램 설명
    parser.add_argument('--batch_size', default=4, type=int, help="batch size") # help는 인수설명
    parser.add_argument('--epoch', default=40, type=int, help='training epoch')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--l2', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--model_name', default='efficientnet', type=str, help='model name')
    parser.add_argument('--pretrained', default=None, type=str, help='pretrained model')
    parser.add_argument('--train', default='train', type=str, help='train and eval')
    args = parser.parse_args()

    # 데이터 불러오기

    train_dataloader, validation_dataloader = datasets.dataloader(args.batch_size)
    print('Completed loading your datasets.')

    # 모델 불러오기 및 학습하기
    learning = training.SuperviedLearning(train_dataloader, validation_dataloader, args.model_name)

    if args.train == 'train':
        learning.train(args.epoch, args.lr, args.l2)
    else:
        print('test')