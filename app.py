import os
from flask import Flask, jsonify, make_response, request
from flask_restful import Api, Resource
from flask_cors import CORS

from flask_prediction import load_model, predict
from counter import addLog
import datetime

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

api = Api(app, prefix="/api/to_back_end")

labels = ["Iderodcomedian", "enkhbat"]
model, tokenizer, processor, output_mode, device = load_model()

class Predictor(Resource):

    def get(self):

        print("request",request)
        text = request.args['text']
        url = request.base_url
        print("text:",text)

        index_, score_ = predict(text, model, tokenizer, processor, output_mode, device)
        date_now = str(datetime.datetime.now())
        req_count = addLog(date_now, url, text, index_, score_)

        return make_response(jsonify(
                username = labels[index_],
                score = str(score_),
                req_count = req_count
        ), 200)

    def post(self):
        # content = request.args['error']
        # print("content:",content)
        return make_response(jsonify(
            "post"
        ), 200)

api.add_resource(Predictor, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
