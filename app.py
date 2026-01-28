from flask import Flask, render_template, request, jsonify, url_for
import data as dt
from plot import gerar_grafico, gerar_grafico_radar
import os

app = Flask(__name__)

# Estados e anos disponíveis
estados = dt.capturar_estado()
anos = dt.capturar_ano()

@app.route("/")
def index():
    estados = dt.capturar_estado()
    anos = dt.capturar_ano()
    resultado_comparacoes = dt.carregar_resultado_valores_comuns()
    return render_template("index.html", estados=estados, anos=anos, comparacoes=resultado_comparacoes)
    #return render_template("index.html", estados=estados, anos=anos)

@app.route("/grafico", methods=["POST"])
def grafico():
    estado = request.form.get("estado")
    estado2 = request.form.get("estado2")

    # Tenta converter o ano 1. Se falhar, assume 0 (ou o valor que preferir)
    try:
        ano = int(request.form.get("ano"))
    except:
        ano = 0 

    # Tenta converter o ano 2. Se falhar, define como None
    try:
        ano2 = int(request.form.get("ano2"))
    except:
        ano2 = None 

    # Agora as variáveis 'ano' e 'ano2' existem garantidamente.
    fig_html, angulos = gerar_grafico(estado, ano, estado2, ano2)
    
    return jsonify({"grafico": fig_html, **angulos})

@app.route("/gif/<estado>")
def gif(estado):
    gif_path = f"assets/{estado}.gif"
    if os.path.exists(os.path.join("static", gif_path)):
        return jsonify({"src": url_for('static', filename=gif_path)})
    return jsonify({"src": ""})

@app.route("/grafico_valor/<valor>")
def grafico_valor(valor):
    resultado = dt.carregar_resultado_valores_comuns()
    try:
        valor_float = float(valor)
    except ValueError:
        return "Valor inválido", 400

    info = resultado.get(valor_float)
    if not info or len(info["estados"]) != 2:
        return f"Valor {valor} precisa ter exatamente dois estados", 404

    from plot import gerar_grafico_comparativo_estados
    html = gerar_grafico_comparativo_estados(valor_float, info["estados"])
    return html

@app.route("/tabela_valores")
def tabela_valores():
    dados = dt.carregar_resultado_valores_comuns()
    return render_template("tabela_valores.html", comparacoes=dados)

@app.route("/grafico_radar", methods=["POST"])
def grafico_radar():
    estado1 = request.form.get("estado")
    ano1 = request.form.get("ano")
    estado2 = request.form.get("estado2")
    ano2 = request.form.get("ano2")

    fig_html = gerar_grafico_radar(estado1, ano1, estado2, ano2)
    return jsonify({"grafico": fig_html})

if __name__ == "__main__":
    app.run(debug=True)
