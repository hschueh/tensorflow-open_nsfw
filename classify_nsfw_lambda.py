#!/usr/bin/env python
import sys
import requests
import json
import os
import tensorflow as tf

from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader

import numpy as np


IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"

def classify_nsfw_lambda(imgs):
    model = OpenNsfwModel()

    with tf.Session() as sess:

        def classity_nsfw(url):
            if 'jpg' not in url or '/a/' in url:
                return -1
            print("Downloading from '{}'".format(url))
            local_filename = url.split('/')[-1]
            try:
                r = requests.get(url, stream=True)
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024): 
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
                image = fn_load_image(local_filename)
                predictions = \
                    sess.run(model.predictions,
                            feed_dict={model.input: image})
            
                print("Results for '{}'".format(local_filename))
                print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]))
                return predictions[0][1]
            except requests.exceptions.ConnectionError:
                print("Connection Err")
                return -1
            except tf.errors.InvalidArgumentError:
                print("Argument Err")
                return -1
            except (OSError, IOError) as e:
                print("FIle Not Found Err")
                return -1
            finally:
                if os.path.isfile(local_filename):
                    os.remove(local_filename)

        model.build(weights_path='data/open_nsfw-weights.npy', input_type=InputType[InputType.TENSOR.name.upper()])

        fn_load_image = create_tensorflow_image_loader(sess)

        sess.run(tf.global_variables_initializer())

        outputObj = {'values': []}
        for img in imgs:
            p = classity_nsfw(img)
            outputObj['values'].append(p)
        return str(outputObj)


if __name__ == "__main__":
    print(str(classify_nsfw_lambda(["http://i.imgur.com/ypZm0KB.jpg","http://i.imgur.com/gHGY33o.jpg"])))