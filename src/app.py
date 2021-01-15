import sys
from flask import Flask, render_template, request

sys.path.append('./backend')
app = Flask(__name__, template_folder='./frontend/templates', static_folder='./frontend/static')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get")
def neural_network():
    from bot import bot
    return bot(request.args.get('msg'))


if __name__ == '__main__':
    app.run()
