#CUDA_VISIBLE_DEVICES=0 python generate_test_data.py --causal_model big_causal_graph --load_path './logs/celebA_0501_182150' --model_type 'began'
from __future__ import print_function
import pandas as pd
import numpy as np
import tensorflow as tf
from copy import deepcopy
from IPython.core import debugger
import pickle
import PIL.Image as Image
import os
debug = debugger.Pdb().set_trace

from main import get_trainer, main
from CausalGAN.generate_training_data import CPD_0, sim_labels, pack_labels

def shift(CPD, child, parent, shift_size):
    for i, (c, b, p) in enumerate(CPD):
        if c == child:
            if parent == "Base":
                CPD[i][1] += shift_size
            elif parent in p.keys():
                CPD[i][2][parent] += shift_size
    return CPD



if __name__ == "__main__":
    # Initialize model
    trainer = get_trainer()
    sess = trainer.sess
    cc=trainer.cc
    if hasattr(trainer,'model'):
        model=trainer.model
    main(trainer)
    # tf.logging.set_verbosity(tf.logging.ERROR)

    # Generate training data for simultaneous shifts
    N = 1000
    if not os.path.exists("../data/celeb_gan/Test_Dist/images/"):
        os.makedirs("../data/celeb_gan/Test_Dist/images")

    df = pd.read_csv('../experiments/celeb_gan/models/simultaneous_delta.csv')
    cpd = deepcopy(CPD_0)
    for child, delta in zip(df['A_names'], df['delta']):
        cpd = shift(deepcopy(cpd), child, "Base", delta)
    labels = sim_labels(N, cpd)
    feed_dict={cc.label_dict[k]: v for k,v in labels.iteritems()}
    feed_dict[trainer.batch_size]=N
    images=sess.run(model.G,feed_dict)
    with open('../data/celeb_gan/Test_Dist/label_names.pkl', 'wb') as f:
        pickle.dump(labels.keys(), f)

    # Save training data
    np.save("../data/celeb_gan/Test_Dist/test_labels.npy".format(child), pack_labels(labels))

    
    # Save images using PIL
    for i, image in enumerate(images):
        Image.fromarray(image.astype(np.uint8)).save("../data/celeb_gan/Test_Dist/images/image_{}.png".format("%04d" % i))

    # np.save('../data/celeb_gan/Test_Dist/images/test_images.npy'.format(child), images)

    
    
    sess.close()