from flask import Flask,request,jsonify
import classify_nsfw_lambda
import json
import simplejson
app = Flask(__name__)

@app.route('/classify_nsfw')
def classidy_nsfw():
    w = request.args.get('imgs', '')
    print(w)
    imgs = w.strip("[]").split(',')
    r = classify_nsfw_lambda.classify_nsfw_lambda(imgs)
    r = r.replace('\'', '\"')
    return r
