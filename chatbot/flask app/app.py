from flask import Flask, Request, jsonify, render_template

from utils import prediction_class, get_response

app = Flask(__name__, template_folder= 'templates')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_message', methods = ['POST'])
def handle_message():
    message = Request.json['message']
    intents_list = prediction_class(message)
    response = get_response(intents_list)

    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug= True)

