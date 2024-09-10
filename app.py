from flask import Flask, render_template, request, redirect, url_for, session
from baseDatos import DBConnection
from generador_nombres import generate_names
#base
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Necesario para utilizar la sesión

# URL de conexión a MySQL
db_url = 'mysql+pymysql://root:Alonso181437@localhost:3306/generador_nombres'
db = DBConnection(db_url)

@app.route('/', methods=['GET', 'POST'])  # Aceptar tanto GET como POST
def index():
    categories = db.get_categories()  # Obtener todas las categorías

    if request.method == 'POST':  # Verificar si el método es POST
        brand_name = request.form.get('brand_name')  # Obtener el nombre de marca o palabra favorita
        category_id = request.form.get('category_id')  # Obtener la categoría seleccionada

        # Obtener prefijos y sufijos de la categoría seleccionada
        prefixes = db.get_prefixes_by_category(category_id)
        suffixes = db.get_suffixes_by_category(category_id)

        # Generar nombres combinando el nombre de marca y los prefijos y sufijos
        names = generate_names(brand_name, prefixes, suffixes, num_names=50)

        # Limpiar la sesión antes de guardar nuevos datos
        session.pop('names', None)  # Eliminar cualquier nombre previo en la sesión
        session['names'] = names  # Guardar los nombres generados en la sesión

        # Redirigir para limpiar la página
        return redirect(url_for('generated'))

    return render_template('index.html', categories=categories)

@app.route('/generated', methods=['GET'])  # Solo aceptar GET
def generated():
    # Obtener los nombres de la sesión y luego limpiar la sesión
    names = session.get('names', [])
    session.pop('names', None)  # Limpiar los nombres de la sesión después de mostrarlos
    categories = db.get_categories()  # Obtener todas las categorías nuevamente

    return render_template('index.html', names=names, categories=categories)

if __name__ == '__main__':
    app.run(debug=True)
