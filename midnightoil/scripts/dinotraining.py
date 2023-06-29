import argparse
import tensorflow as tf

from midnightoil.models.dino import Dino, DinoHead, MultiCropWrapper, load_base
from midnightoil.io.dataset import load_dataset, unravel_dataset, DataGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-epoch", "--epochs", type=int, metavar="", default=100)
    parser.add_argument("-b", "--batch_size", type=int, metavar="", default=2)
    parser.add_argument("-ct", "--crop_teacher", type=int, metavar="", default=128)
    parser.add_argument("-cs", "--crop_student", type=int, metavar="", default=64)
    parser.add_argument(
        "-d_train",
        "--dataset_train",
        type=str,
        metavar="",
        default="datasets/unwrapped/train",
    )
    parser.add_argument(
        "-d_test",
        "--dataset_test",
        type=str,
        metavar="",
        default="datasets/unwrapped/test",
    )
    parser.add_argument(
        "-s_weights",
        "--student_weights_path",
        type=str,
        metavar="",
        default="student_weights",
    )
    parser.add_argument(
        "-t_weights",
        "--teacher_weights_path",
        type=str,
        metavar="",
        default="teacher_weights",
    )

    args = parser.parse_args()
    return args

import yaml
cfg = yaml.safe_load(open('/home/ferreira/src/midnightoil/midnightoil/scripts/setups/SWIN_OPT/swinv_ultratiny.yml'))
import numpy as np
def main():
    args = parse_args()

    head = DinoHead()
    
    teacher = load_base(cfg['model'], args.crop_teacher)
    cfg['model']['window_size'] = 2
    student = load_base(cfg['model'], args.crop_student)



    
    student = MultiCropWrapper(backbone=student, head=head)
    #student.build(input_shape=(10, args.crop_teacher, args.crop_teacher, 1))
    teacher = MultiCropWrapper(backbone=teacher, head=head)

    #print(student.summary())
    #print(teacher.summary())

    model = Dino(teacher, student)
    

   
        
    #test_dataset = test_dataset.with_options(ignore_order) 


 
    train_dataset = DataGenerator(
        mode="train",
        dataset_path=args.dataset_train,
        batch_size=args.batch_size,
        local_image_size=args.crop_student,
        global_image_size=args.crop_teacher,
    )

    val_dataset = DataGenerator(
        mode="val",
        dataset_path=args.dataset_test,
        batch_size=args.batch_size,
        local_image_size=args.crop_student,
        global_image_size=args.crop_teacher,
    )

    learning_rate = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[args.epochs / 2], values=[0.0001, 0.00001]
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.teacher_weights_path,
            monitor="loss",
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
        )
    ]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    model.build(input_shape=(10, args.crop_teacher, args.crop_teacher, 1))
    
    model.fit_generator(
        generator=train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    model.student_model.save_weights(args.student_weights_path)
    model.teacher_model.save_weights(args.teacher_weights_path)
 

if __name__ == "__main__":
    main()