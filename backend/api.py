from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS, cross_origin
from chatgpt import make_a_question

app = Flask(__name__, static_folder='./')
CORS(app, support_credentials=True)

# Rota para a raiz da API
@app.route('/')
def hello_world():
    return send_from_directory(app.static_folder, 'index.html')

# Rota para retornar dados em formato JSON com parâmetro na URL
@app.route('/api/question', methods=['GET'])
def exemplo_api():
    q = request.args.get('q', 'usuário')
    answer = make_a_question(q)
    dados = {'message': answer[0], 'document': answer[1], 'status': 'sucesso'}
    return jsonify(dados)

if __name__ == '__main__':
    app.run(debug=True)
